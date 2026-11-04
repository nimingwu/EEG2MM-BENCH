import os
import shutil
import sys
import warnings  # Added to handle warnings in the main block
import argparse
from PIL import Image
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from source.dataset import EEGDataset
from source.models import ENIGMA
from source.utils import get_eegfeatures, set_seed, compute_retrieval_metrics
import pandas as pd

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True


def recon_inference(
    model_path,
    config_name,
    cache_path,
    output_path,
    model_name,
    subj_ids,
    retrieval_only,
    repetitions,
    seed,
):
    # Set random seeds for reproducibility
    if seed is not None:
        set_seed(seed)

    # Set device
    device = torch.device("cuda")

    subjects = [f"sub-{subj:02d}" for subj in subj_ids]
    torch.hub.set_dir(cache_path)
    output_path = os.path.join(output_path, model_name)
    model_path = os.path.join(model_path, model_name)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    test_dataset = EEGDataset(
        config_name,
        subjects=[subjects[0]],
        split="test",
    )

    # Transformer backbone
    num_channels, num_timepoints = (
        test_dataset.eeg_data.shape[-2],
        test_dataset.eeg_data.shape[-1],
    )
    if not retrieval_only:
        from source.models import SDXL_Reconstructor

        embed_dim = 1024
        generator = SDXL_Reconstructor(device=device, cache_dir=cache_path)

    model = ENIGMA(
        num_channels,
        num_timepoints,
        subjects=subjects,
        embed_dim=1024,
        retrieval_only=retrieval_only,
    )
    # Assert that the checkpoint exists
    assert os.path.exists(f"{model_path}/last.pth"), (
        "checkpoint not found at"
        f" {model_path}/last.pth"
        " if using best.pth, did you train with a validation set?"
    )
    model_weights = torch.load(f"{model_path}/last.pth", map_location=device)
    # Load the checkpoint
    model.load_state_dict(model_weights, strict=False)
    model = model.to(device)
    model.eval()

    for subject in subjects:
        # Load the data
        test_dataset = EEGDataset(
            config_name,
            subjects=[subject],
            split="test",
        )

        subject_dataloader = DataLoader(
            test_dataset,
            batch_size=200,
            shuffle=False,
            num_workers=0,  # Adjust based on CPU cores
            pin_memory=False,
            drop_last=False,  # Ensure all batches have the same size
        )
        # Extract EEG features
        eeg_test_features = get_eegfeatures(
            model, subject_dataloader, device, "test", output_path
        )

        # retrieval grid creation and benchmarking
        ground_truth_embeds = test_dataset.get_image_features()
        ground_truth_embeds /= ground_truth_embeds.norm(dim=-1, keepdim=True)
        ground_truth_images = test_dataset.get_images()

        if retrieval_only:
            # Compute retrieval metrics
            topk_accuracy = compute_retrieval_metrics(
                eeg_test_features, ground_truth_embeds
            )
            df = pd.DataFrame(
                [{f"Top {k} Accuracy": v for k, v in topk_accuracy.items()}]
            )
            df.to_csv(
                os.path.join(
                    model_path, f"retrieval_statistics_{subject}.csv"
                ),
                index=False,
            )
        else:

            # Configure DataLoader
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,  # Adjust based on CPU cores
                pin_memory=False,
                drop_last=False,  # Ensure all batches have the same size
                persistent_workers=True,
            )
            # Inference Loop
            with torch.no_grad():
                for batch in tqdm(
                    test_dataloader,
                    desc=f"Subject {subject} Reconstruction loop",
                    file=sys.stdout,
                ):
                    stimuli = int(batch.class_id[0])
                    sample_path = os.path.join(
                        output_path, f"reconstructions_{subject}", str(stimuli)
                    )
                    os.makedirs(sample_path, exist_ok=True)

                    # Save ground truth image
                    gt_image_path = batch.img_path[0]
                    Image.open(gt_image_path).save(
                        os.path.join(sample_path, "gt_image.jpg"),
                        format="JPEG",  # Explicitly set format
                        quality=75,  # Quality from 1 (worst) to 95 (best); lower means more compression
                        optimize=True,  # Optimize the Huffman tables
                        progressive=True,
                    )
                    torch.save(
                        eeg_test_features[stimuli].cpu(),
                        os.path.join(sample_path, "predicted_embeds.pt"),
                    )

                    # Generate the images
                    images = []

                    for rep in range(repetitions):
                        image = generator.reconstruct(
                            c_i=eeg_test_features[stimuli].unsqueeze(0),
                            n_samples=1,
                        )
                        # Save the PIL Image
                        image.save(
                            os.path.join(sample_path, f"{rep}.jpg"),
                            format="JPEG",  # Explicitly set format
                            quality=75,  # Quality from 1 (worst) to 95 (best); lower means more compression
                            optimize=True,  # Optimize the Huffman tables
                            progressive=True,
                        )  # Create a progressive JPEG)

            final_recons = torch.zeros(
                (len(test_dataset), repetitions, 3, 224, 224)
            )
            final_embeds = torch.zeros((len(test_dataset), embed_dim))

            for stimulus in range(len(test_dataset)):
                embeds_path = os.path.join(
                    output_path,
                    f"reconstructions_{subject}",
                    f"{stimulus}",
                    "predicted_embeds.pt",
                )

                for rep in range(repetitions):
                    recon_path = os.path.join(
                        output_path,
                        f"reconstructions_{subject}",
                        f"{stimulus}",
                        f"{rep}.jpg",
                    )
                    if os.path.exists(recon_path):
                        recon_image = Image.open(recon_path)
                        final_recons[stimulus, rep] = transforms.ToTensor()(
                            recon_image.resize((224, 224))
                        )
                    else:
                        print(
                            "Reconstruction not found for"
                            f" stimulus {stimulus}, repetition {rep}"
                        )

            # Save the aggregated reconstructions and embeddings
            torch.save(
                final_recons,
                os.path.join(output_path, f"final_recons_{subject}.pt"),
            )


def main():
    """
    Main function to parse arguments and spawn distributed processes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Reconstruction Inference with Distributed Data Parallel"
            " (DDP)."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="train_logs",
        help="Path to where model weights and training metadata are stored.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="cache",
        help=(
            "Path to where misc. files downloaded from HuggingFace or Torch"
            " Hub are stored. Defaults to shared directory."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Path to where the features and reconstructions are stored.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="things_eeg2",
        help=(
            "Name of the config to load for the dataset (looks in configs"
            " directory)."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ENIGMA",
        help="Name of model, used for checkpoint saving",
    )
    parser.add_argument(
        "--subj_ids",
        nargs="+",
        type=int,
        default=[1],
        help="List of subject IDs to train on.",
    )
    parser.add_argument(
        "--retrieval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only perform retrieval grid creation, no reconstructions.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions to sample for each sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generators.",
    )
    args = parser.parse_args()
    # Print arguments to the slurm log for reference
    print("recon_inference.py ARGUMENTS:\n-----------------------")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-----------------------")

    # Multi-gpu will use all devices available to it, this is intended to be controlled via the devices you allocate with SLURM
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    if world_size < 1:
        raise ValueError("No GPUs available for reconstruction.")

    recon_inference(
        model_path=args.model_path,
        config_name=args.config_name,
        cache_path=args.cache_path,
        output_path=args.output_path,
        model_name=args.model_name,
        subj_ids=args.subj_ids,
        retrieval_only=args.retrieval_only,
        repetitions=args.repetitions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
