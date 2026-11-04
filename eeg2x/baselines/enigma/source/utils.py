import os
import gc
import numpy as np
import torch
import torch.functional as F
# EEG2X-Benchmark: upstream logs to W&B at import time.  The benchmark records
# its provenance in each run's run_config.json instead, so the dependency is
# optional here and only the ``gather_features`` helper is consumed.
try:
    import wandb
except ImportError:  # pragma: no cover - exercised when wandb is absent
    wandb = None
import yaml
from types import SimpleNamespace
import subprocess
import random
import torch.nn as nn
from typing import Tuple

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))
    return trainable


def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch for single GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch for multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures determinism in cuDNN
    torch.backends.cudnn.benchmark = (
        False  # Avoids nondeterministic algorithms in cuDNN
    )


def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


def parse_config(config_file="experiment.yml"):
    """Parse the configuration from a YAML file."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return dict_to_namespace(config)


# Does not accept a model wrapped in DDP
def get_eegfeatures(model, dataloader, device, mode, output_path):
    model.eval()
    features_list = []  # List to store features
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            eeg_data = batch.eeg.to(device, non_blocking=True)
            img_features = batch.img_vector.to(device, non_blocking=True)
            subjects = batch.subject

            backbone = model(eeg_data, subjects)
            features_list.append(backbone.float())

        features_tensor = torch.cat(features_list, dim=0)

    return features_tensor.cpu()


def compute_retrieval_metrics(
    decoded_vectors: torch.Tensor,
    ground_truth_vectors: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Compute Top1, Top5, and Top10 retrieval metrics.

    """
    n = len(ground_truth_vectors)
    k_values = range(100)
    labels = np.arange(n)
    topk_counts = {}
    for k in k_values:
        topk_counts[k] = 0
    similarity = decoded_vectors @ ground_truth_vectors.T
    for k in k_values:
        _, candidates = torch.topk(similarity, k=k)
        candidates = labels[candidates]
        topk_counts[k] += sum(
            [labels[i] in candidates[i] for i in range(len(labels))]
        )
    topk_accuracies = {}
    for k in k_values:
        topk_accuracies[k] = topk_counts[k] / n

    return topk_accuracies


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
