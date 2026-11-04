import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
import scipy as sp
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

from source.dataset import EEGDataset

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True


def evaluate_recons(
    model_path,
    config_name,
    cache_path,
    output_path,
    model_name,
    subj_ids,
):
    device = torch.device("cuda")
    # Refine input parameters
    subjects = [f"sub-{subj:02d}" for subj in subj_ids]
    torch.hub.set_dir(cache_path)
    output_path = os.path.join(output_path, model_name)
    model_path = os.path.join(model_path, model_name)

    for subj in subjects:
        # Load the recons
        final_recons = torch.load(
            os.path.join(output_path, f"final_recons_{subj}.pt")
        )
        test_dataset = EEGDataset(config_name, subjects=[subj], split="test")

        ground_truth_images = transforms.Resize((224, 224))(
            test_dataset.get_images()
        )

        @torch.no_grad()
        def two_way_identification(
            all_recons,
            all_images,
            model,
            preprocess,
            feature_layer=None,
            return_avg=False,
        ):
            preds = model(
                torch.stack(
                    [preprocess(recon) for recon in all_recons], dim=0
                ).to(device)
            )
            reals = model(
                torch.stack(
                    [preprocess(indiv) for indiv in all_images], dim=0
                ).to(device)
            )
            if feature_layer is None:
                preds = preds.float().flatten(1).cpu().numpy()
                reals = reals.float().flatten(1).cpu().numpy()
            else:
                preds = preds[feature_layer].float().flatten(1).cpu().numpy()
                reals = reals[feature_layer].float().flatten(1).cpu().numpy()

            # Compute correlation matrix
            # Each row: features of an image
            # Transpose to have variables as columns
            reals_T = reals.T
            preds_T = preds.T
            r = np.corrcoef(reals_T, preds_T, rowvar=False)

            # Extract correlations between reals and preds
            N = len(all_images)
            r = r[:N, N:]  # Shape (N, N)

            # Get congruent correlations (diagonal elements)
            congruents = np.diag(r)

            # For each reconstructed image, compare its correlation with the correct original image
            # vs. other original images
            success_counts = []
            total_comparisons = N - 1  # Exclude self-comparison

            for i in range(N):
                # Correlations of reconstructed image i with all original images
                correlations = r[:, i]
                # Correlation with the correct original image
                congruent = congruents[i]
                # Count how many times the correlation with other images is less than the congruent correlation
                successes = (
                    np.sum(correlations < congruent) - 1
                )  # Subtract 1 to exclude the self-comparison
                success_rate = successes / total_comparisons
                success_counts.append(success_rate)

            if return_avg:
                # Return the average success rate
                return np.mean(success_counts)
            else:
                # Return the list of success rates per reconstructed image
                return success_counts

        preprocess_pixcorr = transforms.Compose(
            [
                transforms.Resize(
                    512, interpolation=transforms.InterpolationMode.BILINEAR
                ),
            ]
        )

        def get_pix_corr(all_images, all_recons, return_avg=False):
            # Flatten images while keeping the batch dimension
            all_images_flattened = (
                preprocess_pixcorr(all_images)
                .reshape(len(all_images), -1)
                .cpu()
            )
            all_recons_flattened = (
                preprocess_pixcorr(all_recons).view(len(all_recons), -1).cpu()
            )

            correlations = []
            for i in range(len(all_images)):
                correlations.append(
                    np.corrcoef(
                        all_images_flattened[i], all_recons_flattened[i]
                    )[0][1]
                )
            if return_avg:
                return np.mean(correlations)
            else:
                return correlations

        preprocess_ssim = transforms.Compose(
            [
                transforms.Resize(
                    512, interpolation=transforms.InterpolationMode.BILINEAR
                ),
            ]
        )

        def get_ssim(all_images, all_recons, return_avg=False):

            # convert image to grayscale with rgb2grey
            img_gray = rgb2gray(
                preprocess_ssim(all_images).permute((0, 2, 3, 1)).cpu()
            )
            recon_gray = rgb2gray(
                preprocess_ssim(all_recons).permute((0, 2, 3, 1)).cpu()
            )

            ssim_score = []
            for im, rec in zip(img_gray, recon_gray):
                ssim_score.append(
                    structural_similarity(
                        rec,
                        im,
                        multichannel=True,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                        data_range=1.0,
                    )
                )
            if return_avg:
                return np.mean(ssim_score)
            else:
                return ssim_score

        alex_weights = AlexNet_Weights.IMAGENET1K_V1

        alex_model = create_feature_extractor(
            alexnet(weights=alex_weights),
            return_nodes=["features.4", "features.11"],
        ).to(device)
        alex_model.eval().requires_grad_(False)
        preprocess_alexnet = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def get_alexnet(all_images, all_recons, return_avg=False):
            # AlexNet(2)
            alexnet2 = two_way_identification(
                all_recons.to(device).float(),
                all_images,
                alex_model,
                preprocess_alexnet,
                "features.4",
                return_avg=return_avg,
            )

            # AlexNet(5)
            alexnet5 = two_way_identification(
                all_recons.to(device).float(),
                all_images,
                alex_model,
                preprocess_alexnet,
                "features.11",
                return_avg=return_avg,
            )
            return alexnet2, alexnet5

        weights = Inception_V3_Weights.DEFAULT
        inception_model = create_feature_extractor(
            inception_v3(weights=weights), return_nodes=["avgpool"]
        ).to(device)
        inception_model.eval().requires_grad_(False)
        preprocess_inception = transforms.Compose(
            [
                transforms.Resize(
                    342, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def get_inceptionv3(all_images, all_recons, return_avg=False):

            inception = two_way_identification(
                all_recons.float(),
                all_images.float(),
                inception_model,
                preprocess_inception,
                "avgpool",
                return_avg=return_avg,
            )

            return inception

        import clip

        clip_model, preprocess = clip.load(
            "ViT-L/14", device=device, download_root=cache_path
        )
        preprocess_clip = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        def get_clip(all_images, all_recons, return_avg=False):
            clip_2way = two_way_identification(
                all_recons,
                all_images,
                clip_model.encode_image,
                preprocess_clip,
                None,
                return_avg=return_avg,
            )  # final layer
            return clip_2way

        @torch.no_grad()
        def get_clip_cosine(all_images, all_recons, return_avg=False):
            # Get the cosine similarity between the clip embeddings
            # of the final recons and the ground truth images
            final_embeds = clip_model.encode_image(
                torch.stack(
                    [preprocess_clip(recon) for recon in all_recons], dim=0
                ).to(device)
            )
            gt_embeds = clip_model.encode_image(
                torch.stack(
                    [preprocess_clip(indiv) for indiv in all_images], dim=0
                ).to(device)
            )

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = [float(value) for value in cos(final_embeds, gt_embeds)]
            if return_avg:
                return np.mean(cos_sim)
            else:
                return cos_sim

        weights = EfficientNet_B1_Weights.DEFAULT
        eff_model = create_feature_extractor(
            efficientnet_b1(weights=weights), return_nodes=["avgpool"]
        )
        eff_model.eval().requires_grad_(False)
        preprocess_efficientnet = transforms.Compose(
            [
                transforms.Resize(
                    255, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def get_efficientnet(all_images, all_recons, return_avg=False):
            # see weights.transforms()

            gt = eff_model(preprocess_efficientnet(all_images))["avgpool"]
            gt = gt.reshape(len(gt), -1).cpu().numpy()
            fake = eff_model(preprocess_efficientnet(all_recons))["avgpool"]
            fake = fake.reshape(len(fake), -1).cpu().numpy()

            effnet = [
                sp.spatial.distance.correlation(gt[i], fake[i])
                for i in range(len(gt))
            ]
            if return_avg:
                return np.mean(effnet)
            else:
                return effnet

        swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")

        swav_model = create_feature_extractor(
            swav_model, return_nodes=["avgpool"]
        )
        swav_model.eval().requires_grad_(False)
        preprocess_swav = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def get_swav(all_images, all_recons, return_avg=False):
            gt = swav_model(preprocess_swav(all_images))["avgpool"]
            gt = gt.reshape(len(gt), -1).cpu().numpy()
            fake = swav_model(preprocess_swav(all_recons))["avgpool"]
            fake = fake.reshape(len(fake), -1).cpu().numpy()

            swav = [
                sp.spatial.distance.correlation(gt[i], fake[i])
                for i in range(len(gt))
            ]
            if return_avg:
                return np.mean(swav)
            else:
                return swav

        metrics_data = {
            "sample": [],
            "repetition": [],
            "PixCorr": [],
            "SSIM": [],
            "AlexNet(2)": [],
            "AlexNet(5)": [],
            "InceptionV3": [],
            "CLIP": [],
            "CLIP_Cosine": [],
            "EffNet-B": [],
            "SwAV": [],
        }

        # Iterate over each sample and compute metrics with tqdm and suppressed output
        for repetition in tqdm(
            range(final_recons.shape[1]),
            desc="Processing samples",
            file=sys.stdout,
        ):
            # with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            rep_recons = final_recons[:, repetition]

            pixcorr = get_pix_corr(ground_truth_images, rep_recons)
            ssim = get_ssim(ground_truth_images, rep_recons)
            alexnet2, alexnet5 = get_alexnet(ground_truth_images, rep_recons)
            inception = get_inceptionv3(ground_truth_images, rep_recons)
            clip = get_clip(ground_truth_images, rep_recons)
            clip_cosine = get_clip_cosine(ground_truth_images, rep_recons)
            effnet = get_efficientnet(ground_truth_images, rep_recons)
            swav = get_swav(ground_truth_images, rep_recons)

            # Append each result to its corresponding list, and store the image index

            metrics_data["sample"].extend(list(range(final_recons.shape[0])))
            metrics_data["repetition"].extend(
                [repetition for _ in range(final_recons.shape[0])]
            )
            metrics_data["PixCorr"].extend(pixcorr)
            metrics_data["SSIM"].extend(ssim)
            metrics_data["AlexNet(2)"].extend(alexnet2)
            metrics_data["AlexNet(5)"].extend(alexnet5)
            metrics_data["InceptionV3"].extend(inception)
            metrics_data["CLIP"].extend(clip)
            metrics_data["CLIP_Cosine"].extend(clip_cosine)
            metrics_data["EffNet-B"].extend(effnet)
            metrics_data["SwAV"].extend(swav)

        # Check that all lists have the same length before creating DataFrame
        lengths = [len(values) for values in metrics_data.values()]
        if len(set(lengths)) != 1:
            print("Error: Not all metric lists have the same length")
            for metric, values in metrics_data.items():
                print(f"{metric}: {len(values)} items")
        else:
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(metrics_data)
            # Save the table to a CSV file
            df.to_csv(os.path.join(model_path, f"recon_statistics_{subj}.csv"))


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Reconstruction Inference with Specified Configurations."
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
        help="Comma-separated list of subject IDs to train on.",
    )

    args = parser.parse_args()
    # Print model arguments to the slurm log for reference
    print("evaluate_recons.py ARGUMENTS:\n-----------------------")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-----------------------")

    evaluate_recons(
        model_path=args.model_path,
        config_name=args.config_name,
        cache_path=args.cache_path,
        output_path=args.output_path,
        model_name=args.model_name,
        subj_ids=args.subj_ids,
    )


if __name__ == "__main__":
    main()
