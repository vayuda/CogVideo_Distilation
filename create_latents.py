


# --- test inference ---
# from PIL import Image
# from diffusers import CogVideoXImageToVideoPipeline
# from diffusers.utils import export_to_video, load_image
# import torch



# Encode example video image and text
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline
from diffusers.video_processor import VideoProcessor
from diffusers.utils import load_image
import torch

device = "cuda"
model_path = "THUDM/CogVideoX1.5-5B-I2V"

pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

with open("prompt.txt", "r") as f:
    prompt = f.read().strip()

def encode_prompt(prompt, tokenizer, text_encoder, device):
    prompt_token_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=226,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(device)
    prompt_token_ids = prompt_token_ids.input_ids
    prompt_embedding = text_encoder(prompt_token_ids)[0]
    return prompt_embedding

# prompt_embedding = encode_prompt(prompt, tokenizer, text_encoder, device)
# torch.save(prompt_embedding, "horse_riding_text_prompt.pt")
# print(f"saved input prompt embed {prompt_embedding.shape}, {prompt_embedding.dtype}")
# [1, 226, 4096]

image = load_image("horse.png")
vae_scale_factor_temporal = 4
vae_scale_factor_spatial = 8
vp = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
image = vp.preprocess(image).to(device=device, dtype=torch.bfloat16)
latent_channels = pipe.transformer.config.in_channels // 2
print(f"latent_channels: {latent_channels}")
height = 480
width = 720
latents, image_latents = pipe.prepare_latents(
            image,
            batch_size = 1,
            num_channels_latents = latent_channels,
            num_frames = 48,
            height = height,
            width = width,
            device= device,
        )
print(f"latents: {latents.shape}, {latents.dtype}")
# torch.save(latents, "horse_riding_latents.pt")
# torch.save(image_latents, "horse_riding_image_latents.pt")
