import os
import sys
import math
import numpy as np
from PIL import Image
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from torchvision import transforms
from huggingface_hub import hf_hub_download
from einops.layers.torch import Rearrange
from torch import Tensor
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class CLIPEncoder:
    def __init__(
        self,
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        precision="fp32",
        batch_size: int = 20,
        device="cuda",
        **kwargs,
    ):
        self.batch_size = batch_size
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained, precision, device=device, **kwargs
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

    def encode_text(self, text, normalize=False):
        features = []
        for i in tqdm(
            range(0, len(text), self.batch_size), desc="CLIP Encoding text..."
        ):
            batch_text = text[i : min(i + self.batch_size, len(text))]
            inputs = self.tokenizer(batch_text).to(self.device)
            with torch.no_grad():
                batch_features = self.model.encode_text(inputs)
                if normalize:
                    batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features)
        features = torch.cat(features, dim=0)
        return features.detach().cpu()

    def encode_image(self, image):
        if isinstance(image, Image.Image):
            image = [image]
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = [image]
            elif image.ndim != 4:
                raise ValueError("Invalid tensor shape for image encoding.")

        elif isinstance(image, list) and all(
            isinstance(img, Image.Image) for img in image
        ):
            image = [self.preprocess(img.convert("RGB")) for img in image]
        elif isinstance(image, list) and all(
            isinstance(img, torch.Tensor) for img in image
        ):
            image = [
                img.unsqueeze(0) if img.ndim == 3 else img for img in image
            ]
        elif isinstance(image, list) and all(
            isinstance(img, str) for img in image
        ):
            print("Preprocessing images...")
            preprocessed_image = []
            for i in tqdm(
                range(0, len(image)),
                desc="CLIP Preprocessing images...",
            ):
                preprocessed_image.append(
                    self.preprocess(Image.open(image[i]).convert("RGB"))
                )
            image = preprocessed_image
        else:
            raise ValueError("Unsupported image type for encoding.")

        features = []
        for i in tqdm(
            range(0, len(image), self.batch_size),
            desc="CLIP Encoding images...",
        ):
            batch_images = image[i : min(i + self.batch_size, len(image))]
            if isinstance(batch_images, list):
                batch_images = torch.stack(batch_images)
            with torch.no_grad():
                batch_features = self.model.encode_image(
                    batch_images.to(self.device)
                )
            features.append(batch_features)

        features = torch.cat(features, dim=0)
        return features.detach().cpu()


class Spatio_Temporal_CNN(nn.Module):
    """Configurable Spatio_Temporal_CNN"""

    def __init__(
        self,
        emb_size=4,
        conv1_kernel=(1, 5),
        pool1_kernel=(1, 17),
        pool1_stride=(1, 5),
        conv2_kernel=(63, 1),
    ):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, conv1_kernel, stride=(1, 1)),
            nn.AvgPool2d(pool1_kernel, pool1_stride),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, conv2_kernel, stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.contiguous().view(x.size(0), -1)
        return x


class ResidualAdd(nn.Module):
    """Residual addition."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    """Flatten."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class MLP_Projector(nn.Sequential):
    """EEG Projection."""

    def __init__(self, embedding_dim=184, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class ENIGMA(nn.Module):

    def __init__(
        self,
        num_channels,
        sequence_length,
        subjects,
        hidden_dim=184,
        embed_dim=1024,
        retrieval_only=False,
    ):
        super(ENIGMA, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.subject_wise_linear = nn.ModuleDict(
            {
                subject: nn.Linear(sequence_length, sequence_length)
                for subject in subjects
            }
        )
        self.tsencoder = Spatio_Temporal_CNN(
            emb_size=4,
            conv1_kernel=(1, 5),
            pool1_kernel=(1, 17),
            pool1_stride=(1, 5),
            conv2_kernel=(num_channels, 1),
        )
        self.mlp_proj = MLP_Projector(
            embedding_dim=hidden_dim, proj_dim=embed_dim, drop_proj=0.5
        )
        if retrieval_only:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, subjects):
        # Project each subject through its own linear layer
        x_eeg = torch.zeros_like(x)
        subjects = np.array(subjects)
        unique_subjects = np.unique(subjects)
        for subject in unique_subjects:
            subj_mask = torch.from_numpy(subjects == subject)
            x_eeg[subj_mask] = self.subject_wise_linear[subject](
                x[subj_mask]
            )

        z_eeg = self.tsencoder(x_eeg)
        c_eeg = self.mlp_proj(z_eeg)
        return c_eeg


class SDXL_Reconstructor:

    def __init__(
        self,
        num_inference_steps=4,
        device="cuda",
        cache_dir="/srv/eeg_reconstruction/shared/cache/",
    ):
        self.num_inference_steps = num_inference_steps
        self.dtype = torch.float16
        self.device = device

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
        )
        feature_extractor = CLIPImageProcessor(size=224, crop_size=224)

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            variant="fp16",
            cache_dir=cache_dir,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        pipe.to(device)
        # load ip adapter
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        # set ip_adapter scale (default is 1)
        pipe.set_ip_adapter_scale(1)
        self.pipe = pipe
        self.pipe.vae = self.pipe.vae.to(torch.float32)

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            # Handle the case where 'image' is a tensor
            if image.ndim == 3:
                image = [
                    transforms.ToPILImage()(image.cpu())
                ]  # Convert single tensor to PIL Image
            elif image.ndim == 4:
                image = [
                    transforms.ToPILImage()(img.cpu()) for img in image
                ]  # Convert batch of tensors to list of PIL Images
            else:
                raise ValueError(
                    "Unsupported tensor shape. Expected 3 or 4 dimensions."
                )
        elif isinstance(image, str):
            # Handle the case where 'image' is a string (assumed to be a file path)
            image = [
                Image.open(image).convert("RGB")
            ]  # Open image and convert to RGB
        elif isinstance(image, list) and all(
            isinstance(img, str) for img in image
        ):
            # Handle the case where 'image' is a list of strings (image file paths)
            image = [
                Image.open(path).convert("RGB") for path in image
            ]  # Open each image and convert to RGB
        elif isinstance(image, Image.Image):
            # Handle the case where 'image' is a PIL Image
            image = [image.convert("RGB")]  # Ensure image is in RGB format
        elif isinstance(image, list) and all(
            isinstance(img, Image.Image) for img in image
        ):
            # Handle the case where 'image' is a list of PIL Images
            image = [
                img.convert("RGB") for img in image
            ]  # Ensure all images are in RGB format
        else:
            raise ValueError(
                "Unsupported image format. Please provide a tensor, image"
                " path, list of image paths, a PIL Image, or a list of PIL"
                " Images."
            )

        with torch.no_grad():
            # Now 'image' is a list of PIL Images
            clip_image, _ = self.pipe.encode_image(
                image=image, device=self.device, num_images_per_prompt=1
            )
        return clip_image

    def encode_text(self, text, cropped=False):
        with torch.no_grad():
            clip_text, _, _, _ = self.pipe.encode_prompt(text)
            if cropped:
                clip_text = clip_text[:, :, :768]
        return clip_text

    def encode_latent(self, image):
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(
                "`image` has to be of type PIL.Image.Image but is"
                f" {type(image)}"
            )

        with torch.no_grad():
            image = self.pipe.image_processor.preprocess(image).to(
                "cuda", torch.float32
            )
            latents = self.pipe.vae.encode(image).latent_dist.sample(None)
            latents = latents * self.pipe.vae.config.scaling_factor
        return latents

    def reconstruct(
        self,
        image=None,
        latent=None,
        c_i=None,
        c_t=None,
        n_samples=1,
        textstrength=0.0,
        strength=1.0,
        cropped=False,
    ):
        with torch.no_grad():
            if c_i is not None and c_t is not None:
                c_i = [
                    c_i.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                ]
                if cropped:
                    c_t = c_t.reshape((-1, 77, 768))
                    c_t = F.pad(c_t, (0, 1280))
                else:
                    c_t = c_t.reshape((-1, 77, 2048))
                prompt = None
                pooled_prompt_embeds = torch.zeros((1, 1280))
            elif c_i is not None:
                c_i = [
                    c_i.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                ]
                prompt = ""
                pooled_prompt_embeds = None
                textstrength = 0.0
            elif c_t is not None:
                c_i = [torch.zeros((1, 1, 1024))]
                if cropped:
                    c_t = c_t.reshape((-1, 77, 768))
                    c_t = F.pad(c_t, (0, 1280))
                else:
                    c_t = c_t.reshape((-1, 77, 2048))
                prompt = None
                pooled_prompt_embeds = torch.zeros((1, 1280))
                textstrength = 1.0
            else:
                strength = 0.0
            if image is None and latent is None:
                strength = 1.0

            if latent is not None:
                latent = latent.reshape((-1, 4, 62, 62)).to(
                    self.device, torch.float32
                )
                latent = latent / self.pipe.vae.config.scaling_factor
                image = self.pipe.vae.decode(latent, return_dict=False)[0]
                images = self.pipe.image_processor.postprocess(
                    image, output_type="pil"
                )

            if strength != 0.0:
                self.pipe.set_ip_adapter_scale(1 - textstrength)
                images = self.pipe(
                    image=image,
                    strength=strength,
                    prompt=prompt,
                    prompt_embeds=c_t,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    ip_adapter_image_embeds=c_i,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=0.0,
                    generator=None,
                    num_images_per_prompt=n_samples,
                ).images

        if len(images) == 1:
            return images[0]
        else:
            return images
