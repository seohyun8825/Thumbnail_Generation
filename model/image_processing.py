from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F


def image_to_latent(image_path, vae, device, unet_embedding_dim=640, target_resolution=(64, 64)):
    """
    이미지를 로드, 전처리하고 VAE를 사용해 latent 벡터를 생성 및 UNet 입력 형태로 변환.

    Args:
        image_path (str): 이미지 파일 경로.
        vae (torch.nn.Module): VAE 모델.
        device (torch.device): 실행 장치 (e.g., "cuda" or "cpu").
        unet_embedding_dim (int): UNet 임베딩 차원 (기본값: 640).
        target_resolution (tuple): UNet이 기대하는 latent 해상도 (기본값: 64x64).

    Returns:
        torch.Tensor: UNet 입력 형태로 변환된 latent 벡터.
    """
    # 1. 이미지 로드 및 전처리
    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        preprocess = Compose([
            Resize((256, 256)),  # VAE 입력 크기에 맞게 조정
            ToTensor(),  # Tensor로 변환
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
        ])
        return preprocess(image).unsqueeze(0)  # 배치 차원 추가

    # 2. 이미지 전처리 및 Tensor로 변환
    image_tensor = preprocess_image(image_path).to(device)

    # 3. VAE 인코딩
    image_tensor = image_tensor.to(dtype=vae.dtype)  # VAE 데이터 타입에 맞추기
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()  # (1, 4, 32, 32)

    # 4. Latent 업샘플링: 32x32 → 64x64
    latents_upsampled = F.interpolate(latents, size=target_resolution, mode="bilinear", align_corners=False)  # (1, 4, 64, 64)

    # 5. Flatten 및 UNet 임베딩 크기로 확장
    batch_size, channels, height, width = latents_upsampled.shape
    latents_flat = latents_upsampled.view(batch_size, -1, channels)  # (1, 4096, 4)

    # 6. Embedding 확장
    embedding_resize = nn.Linear(channels, unet_embedding_dim).to(device, dtype=latents.dtype)
    latents_unet = embedding_resize(latents_flat)  # (1, 4096, 640)
    #latents_unet = latents_unet * self.scheduler.init_noise_sigma
    #print("applied initnosiseisigma")
    return latents_unet

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
    dtype = next(self.image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = self.feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_enc_hidden_states = self.image_encoder(
            torch.zeros_like(image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
def prepare_ip_adapter_image_embeds(
    self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
):
    if ip_adapter_image_embeds is None:
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]

        if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
            raise ValueError(
                f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
            )

        image_embeds = []
        for single_ip_adapter_image, image_proj_layer in zip(
            ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
        ):
            output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
            single_image_embeds, single_negative_image_embeds = self.encode_image(
                single_ip_adapter_image, device, 1, output_hidden_state
            )
            single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
            single_negative_image_embeds = torch.stack(
                [single_negative_image_embeds] * num_images_per_prompt, dim=0
            )

            if do_classifier_free_guidance:
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                single_image_embeds = single_image_embeds.to(device)

            image_embeds.append(single_image_embeds)
    else:
        repeat_dims = [1]
        image_embeds = []
        for single_image_embeds in ip_adapter_image_embeds:
            if do_classifier_free_guidance:
                single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                single_image_embeds = single_image_embeds.repeat(
                    num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                )
                single_negative_image_embeds = single_negative_image_embeds.repeat(
                    num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                )
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
            else:
                single_image_embeds = single_image_embeds.repeat(
                    num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                )
            image_embeds.append(single_image_embeds)

    return image_embeds