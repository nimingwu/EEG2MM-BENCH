import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .dataset import EEGDataset
from .utils import gather_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = (
                    logit_scale * image_features @ all_text_features.T
                )
                logits_per_text = (
                    logit_scale * text_features @ all_image_features.T
                )
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss


def train_epoch(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    device,
    mse_loss_scale,
    retrieval_img_loss_scale,
    retrieval_txt_loss_scale,
    retrieval_only,
):
    """Train loop for one epoch."""
    mse = nn.MSELoss()
    InfoNCE = ClipLoss()

    for batch in dataloader:
        loss = 0
        eeg_data = batch.eeg.to(device, non_blocking=True)
        labels = batch.class_id
        img_features = batch.img_vector.to(device, non_blocking=True)
        img_features_norm = batch.img_vector_norm.to(device, non_blocking=True)
        txt_features_norm = batch.txt_vector_norm.to(device, non_blocking=True)
        subjects = batch.subject
        if retrieval_only:
            logit_scale = model.logit_scale
        else:
            logit_scale = 1

        backbone = model(eeg_data, subjects)

        # Backbone loss MSE
        mse_loss = mse(backbone, img_features) * mse_loss_scale
        loss += mse_loss

        # Backbone loss Contrastive
        contrastive_loss_img = (
            InfoNCE(backbone, img_features_norm, logit_scale)
            * retrieval_img_loss_scale
        )
        contrastive_loss_txt = (
            InfoNCE(backbone, txt_features_norm, logit_scale)
            * retrieval_txt_loss_scale
        )
        loss += contrastive_loss_img
        loss += contrastive_loss_txt

        # Backpropogate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


def evaluate_epoch(
    model,
    dataloader,
    device,
    mse_loss_scale,
    retrieval_img_loss_scale,
    retrieval_txt_loss_scale,
    retrieval_only,
):
    mse = nn.MSELoss()
    InfoNCE = ClipLoss()
    loss_total = 0
    for batch in dataloader:
        loss = 0

        eeg_data = batch.eeg.to(device, non_blocking=True)
        labels = batch.class_id
        img_features = batch.img_vector.to(device, non_blocking=True)
        img_features_norm = batch.img_vector_norm.to(device, non_blocking=True)
        txt_features_norm = batch.txt_vector_norm.to(device, non_blocking=True)
        subjects = batch.subject
        if retrieval_only:
            logit_scale = model.logit_scale
        else:
            logit_scale = 1

        backbone = model(eeg_data, subjects)

        # Backbone loss MSE
        mse_loss = mse(backbone, img_features) * mse_loss_scale
        loss += mse_loss

        # Backbone loss Contrastive
        contrastive_loss_img = (
            InfoNCE(backbone, img_features_norm, logit_scale)
            * retrieval_img_loss_scale
        )
        contrastive_loss_txt = (
            InfoNCE(backbone, txt_features_norm, logit_scale)
            * retrieval_txt_loss_scale
        )
        loss += contrastive_loss_img
        loss += contrastive_loss_txt
        loss_total += loss.item()
    print(f"Loss: {loss.item()}")


def get_dataloaders(
    config_name,
    subjects,
    batch_size,
    *,
    h5_dir=None,
    profile_id=None,
    feature_store=None,
    text_features=None,
    test_batch_size=200,
):
    """EEG2X-Benchmark: ``config_name`` is replaced by the benchmark wiring.

    ENIGMA upstream reads its own ``preprocessed_eeg_*_flat.npy`` arrays; the
    benchmark hands the dataset a protocol-v2 HDF5 directory, a registered
    profile id and a frozen CLIP feature store instead.  One subject per
    model, matching the benchmark's per-subject runs.
    """
    common = dict(
        subject=subjects[0], h5_dir=h5_dir, profile_id=profile_id,
        feature_store=feature_store, text_features=text_features)
    # Train dataset
    train_dataset = EEGDataset(split="train", **common)
    # EEG2X-Benchmark: upstream always drops the last partial batch, which
    # would leave zero batches (and a zero-step OneCycleLR schedule) on the
    # benchmark's smaller splits -- ds004018 trains on 180 stimuli and
    # ds005106 on ~200, both below the paper's 512 batch size.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        drop_last=(len(train_dataset) > batch_size),
        pin_memory=False,
        persistent_workers=True,
    )
    # Test dataset
    test_dataset = EEGDataset(split="test", **common)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return train_dataloader, test_dataloader


def prepare_optimizer(
    model,
    max_lr,
    num_epochs,
    train_dataloader,
    pct_start,
    final_div_factor,
):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Separate out subject-wise parameters so they don't appear in both groups
    backbone = [(n, p) for n, p in model.named_parameters()]

    # Split backbone params into decay/no_decay groups, excluding subject-wise
    backbone_params_decay = [
        p for n, p in backbone if not any(nd in n for nd in no_decay)
    ]
    backbone_params_no_decay = [
        p for n, p in backbone if any(nd in n for nd in no_decay)
    ]

    opt_grouped_parameters = [
        {
            "params": backbone_params_decay,
            "weight_decay": 1e-2,
            "lr": max_lr,
        },
        {
            "params": backbone_params_no_decay,
            "weight_decay": 0.0,
            "lr": max_lr,
        },
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

    # Match max_lr to the number of param groups so each group keeps its scaled rate
    param_group_lrs = []
    for group in opt_grouped_parameters:
        param_group_lrs.append(group["lr"])

    total_steps = int(num_epochs * len(train_dataloader))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=param_group_lrs,
        total_steps=total_steps,
        final_div_factor=final_div_factor,
        pct_start=pct_start,
        anneal_strategy="cos",
    )
    return optimizer, lr_scheduler
