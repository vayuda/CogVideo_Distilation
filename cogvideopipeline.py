from typing import Union, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler, AutoencoderKLCogVideoX, CogVideoXPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.video_processor import VideoProcessor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from tqdm import tqdm

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


class CogVideoXPipeline():
    def __init__(self,
        hf_name: str,
        video_shape: tuple = (49, 720, 480),
        pipe_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: int = 50,
        guidance_scale: float = 0,
    ):
        self.hf_name = hf_name
        self.pipe_dtype = pipe_dtype
        self.device = torch.device("cuda")
        self.dit = CogVideoXTransformer3DModel.from_pretrained(hf_name, subfolder="transformer", torch_dtype=pipe_dtype).to(self.device)
        self.dit.gradient_checkpointing = True
        self.vae = AutoencoderKLCogVideoX.from_pretrained(hf_name, subfolder="vae", torch_dtype=pipe_dtype).to(self.device)
        self.text_encoder = T5EncoderModel.from_pretrained(hf_name, subfolder="text_encoder", torch_dtype=pipe_dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, subfolder="tokenizer",)
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(hf_name, subfolder="scheduler")
        self.mem("init",print_=True)
        self.text_seq_len = 224 if "1.5" in hf_name else 226

        self.transformer_config = self.dit.config
        self.vae_config = self.vae.config
        self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.num_frames = video_shape[0]
        self.width = video_shape[1]
        self.height = video_shape[2]
        self.guidance_scale = guidance_scale

    def mem(self, place, print_=False):
        if print_:
            print(f"Mem usage at {place}: {torch.cuda.memory_allocated() / 1024**2}")

    def unload(self, component: str):
        if component == "dit":
            del self.dit
        elif component == "vae":
            del self.vae
        elif component == "text_encoder":
            del self.text_encoder
        elif component == "tokenizer":
            del self.tokenizer
        elif component == "scheduler":
            del self.scheduler
        torch.cuda.empty_cache()

    def encode_text(self, prompts: Optional[List[str]] = None, save_path: Optional[str] = None):
        if not prompts:
            return None
        assert self.text_encoder is not None, "Text encoder is not loaded"
        assert self.tokenizer is not None, "Tokenizer is not loaded"

        prompt_token_ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.text_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            embeddings = self.text_encoder(**prompt_token_ids)[0].to(self.pipe_dtype)

        if save_path is not None:
            torch.save(embeddings, save_path)

        return embeddings

    def encode_image(self, image_paths: Union[str, List[str]]):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        images = []
        for image_path in image_paths:
            images.append(load_image(image_path))
        images = self.video_processor.preprocess(images).to(self.device).to(self.pipe_dtype)
        return images

    def prepare_latents(
        self, batch_size: int,
        image_paths: Optional[Union[str, List[str]]]=None,
        save_path: Optional[str]=None
    ):
        n_channels = self.transformer_config.in_channels // 2 if image_paths is not None else self.transformer_config.in_channels

        shape = (
            batch_size,
            (self.num_frames - 1) // self.vae_scale_factor_temporal+1,
            n_channels,
            self.height // self.vae_scale_factor_spatial,
            self.width // self.vae_scale_factor_spatial
        )

        latent = torch.randn(shape, device=self.device, dtype=self.pipe_dtype)
        latent = latent * self.scheduler.init_noise_sigma
        if image_paths is not None:
            assert self.vae is not None, "VAE model is not loaded"
            images = self.encode_image(image_paths)
            images = images.unsqueeze(2) # [B,C,H,W] -> [B,C,F,H,W]
            image_latent_dist = self.vae.encode(images.to(dtype=self.vae.dtype)).latent_dist
            image_latents = image_latent_dist.sample() * self.vae.config.scaling_factor
            image_latents = image_latents.permute(0, 2, 1, 3, 4) # [B, F, C, H, W]
            padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
            latent_padding = image_latents.new_zeros(padding_shape)
            image_latents = torch.cat([image_latents, latent_padding], dim=1)
        else:
            image_latents = None



        if save_path is not None:
            torch.save(latent, save_path)
            if image_latents is not None:
                torch.save(image_latents, save_path.replace(".pt", "_image.pt"))
        return latent, image_latents

    def prepare_image_rotary_positional_embeddings(self, latent) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = self.height // (self.vae_scale_factor_spatial * self.transformer_config.patch_size)
        grid_width = self.width // (self.vae_scale_factor_spatial * self.transformer_config.patch_size)

        p = self.transformer_config.patch_size
        p_t = self.transformer_config.patch_size_t

        base_size_width = self.transformer_config.sample_width // p
        base_size_height = self.transformer_config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer_config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=latent.size(1),
                device=self.device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (latent.size(1) + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer_config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=self.device,
            )
        return freqs_cos, freqs_sin

    def denoise_one_step(self, latent, prompt_embeds, t, image_latent=None, image_rotary_embeds=None):
        assert self.dit is not None, "dit is not initialized"
        latent = self.scheduler.scale_model_input(latent, t)
        latent_model_input = torch.cat([latent] * 2) if self.guidance_scale > 1.0 else latent
        # concat the image latent along the channel dimension if provided
        if image_latent is not None:
            image_latent_input = torch.cat([image_latent] * 2) if self.guidance_scale > 1.0 else image_latent
            latent_model_input = torch.cat([latent_model_input, image_latent_input], dim=2)


        timestep = t.expand(latent.shape[0]).to(self.device)
        if "1.5" in self.hf_name:
            ofs_emb = latent.new_full((1,), fill_value=2.0).to(self.device, dtype=self.pipe_dtype)
        else:
            ofs_emb = None



        self.mem("before inference")
        with torch.inference_mode():
            noise_pred = self.dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_embeds if image_latent is not None else None,
                ofs=ofs_emb,
                return_dict=False,
            )[0]
            # self.mem("after inference", print_=True)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        return latent

    def generate(self, batch_size, prompt_embeds, image_paths=None, denoising_steps=50):
        latent, image_latent = self.prepare_latents(batch_size, image_paths)
        image_rotary_embeds = self.prepare_image_rotary_positional_embeddings(latent)
        self.scheduler.set_timesteps(num_inference_steps=denoising_steps)
        for t in tqdm(self.scheduler.timesteps, desc="denoising steps"):
            latent = self.denoise_one_step(latent, prompt_embeds, t, image_latent, image_rotary_embeds)
        return latent

    def export_video_from_latent(self, latents, video_fp):
        assert self.vae is not None, "VAE model is not initialized"
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents
        frames = self.vae.decode(latents).sample
        video = self.video_processor.postprocess_video(frames).squeeze(0)
        fps = 16 if "1.5" in self.hf_name else 8
        export_to_video(video, video_fp, fps=fps)

    def full_pipeline(
        self,
        prompts: List[str],
        video_fp: str,
        image_paths: Optional[List[str]] = None,
        negative_prompts: Optional[List[str]] = None,
        denoising_steps = 50
    ):
        if negative_prompts is not None:
            assert len(negative_prompts) == len(prompts), "Number of negative prompts must match number of prompts"
        if image_paths is not None:
            assert len(image_paths) == len(prompts), "Number of images must match number of prompts"

        prompt_embeds = self.encode_text(prompts)
        negative_prompt_embeds = self.encode_text(negative_prompts)
        if self.guidance_scale >= 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        self.mem("before unload")
        self.unload("text_encoder")
        self.mem("after unload")
        latent = self.generate(len(prompts), prompt_embeds, image_paths=image_paths, denoising_steps=denoising_steps)

        self.export_video_from_latent(latent, video_fp)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    pipeline = CogVideoXPipeline("THUDM/CogVideoX-2b", guidance_scale=7.5, pipe_dtype=torch.float16)

    latent = torch.load("checkpoints/gen_latents_step_latest.pt")
    for i in range(latent.shape[0]):
        pipeline.export_video_from_latent(latent[1].unsqueeze(0), f"samples/epoch{i}.mp4")


    # pipeline.encode_text([prompt], save_path="data/prompt_embed_2b.pt")
    # pipeline.encode_text([negative_prompt], save_path="data/negative_prompt_embed_2b.pt")
    # pipeline.prepare_latents(1, image_paths=["data/horse.png"], save_path="data/horse_2b_latent.pt")

    # with open("data/prompt.txt", "r") as f:
    #     prompt = f.read()
    # with open("data/negative_prompt.txt", "r") as f:
    #     negative_prompt = f.read()
    # pipeline.full_pipeline(
    #     [prompt],
    #     "2b-1.mp4",
    #     negative_prompts=[negative_prompt],
    #     denoising_steps=1
    #     # image_paths=["data/horse.png"]
    # )
