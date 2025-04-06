from multiprocessing import Pipe
from typing import Union, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.models import AutoencoderKLWan
from diffusers.models import WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_image
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm
import ftfy
import html
import re

import os


class WanPipeline():
    def __init__(self,
        hf_name: str,
        video_shape: tuple = (81, 832, 480),
        pipe_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: int = 50,
        guidance_scale: float = 5.0,
        load_modules: List = []
    ):
        self.hf_name = hf_name
        self.pipe_dtype = pipe_dtype
        self.device = torch.device("cuda")
        self.dit = WanTransformer3DModel.from_pretrained(hf_name, subfolder="transformer", torch_dtype=pipe_dtype).to(self.device)
        self.dit.gradient_checkpointing = True
        self.vae = AutoencoderKLWan.from_pretrained(hf_name, subfolder="vae", torch_dtype=pipe_dtype).to(self.device)
        if load_modules != [] and "text_encoder" in load_modules:
            self.text_encoder = UMT5EncoderModel.from_pretrained(hf_name, subfolder="text_encoder", torch_dtype=pipe_dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, subfolder="tokenizer",)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(hf_name, subfolder="scheduler")
        self.mem("init",print_=True)
        self.text_seq_len = 226

        self.transformer_config = self.dit.config
        self.vae_config = self.vae.config
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
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
        for prompt in prompts:
            prompt = ftfy.fix_text(prompt)
            prompt = html.unescape(html.unescape(prompt))
            prompt = re.sub(r"\s+", " ", prompt)
            prompt = prompt.strip()

        batch_size = len(prompts)
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.text_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), mask.to(self.device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.pipe_dtype, device=self.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(self.text_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)
        if save_path is not None:
            torch.save(prompt_embeds, save_path)

        return prompt_embeds

    def prepare_latents(
        self, batch_size: int = 1,
        image_paths: Optional[Union[str, List[str]]]=None,
        save_path: Optional[str]=None
    ):
        shape = (
            batch_size,
            self.dit.config.in_channels,
            (self.num_frames - 1) // self.vae_scale_factor_temporal+1,
            self.height // self.vae_scale_factor_spatial,
            self.width // self.vae_scale_factor_spatial
        )
        print(shape)

        latent = torch.randn(shape, device=self.device, dtype=self.pipe_dtype)
        if save_path is not None:
            torch.save(latent, save_path)

        return latent

    def denoise_one_step(self, latent, prompt_embeds, t):
        assert self.dit is not None, "dit is not initialized"
        latent_model_input = torch.cat([latent] * 2) if self.guidance_scale > 1.0 else latent
        timestep = t.expand(latent.shape[0]).to(self.device)

        self.mem("before inference")
        with torch.inference_mode():
            noise_pred = self.dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                return_dict=False,
            )[0]
            # self.mem("after inference", print_=True)
        if self.guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        return latent

    def generate(self, batch_size, prompt_embeds, denoising_steps=50):
        latent = self.prepare_latents(batch_size)
        self.scheduler.set_timesteps(num_inference_steps=denoising_steps)
        for t in tqdm(self.scheduler.timesteps, desc="denoising steps"):
            latent = self.denoise_one_step(latent, prompt_embeds, t)
        return latent

    def export_video_from_latent(self, latents, video_fp):
        assert self.vae is not None, "VAE model is not initialized"
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        frames = self.vae.decode(latents)[0]
        video = self.video_processor.postprocess_video(frames).squeeze(0)
        export_to_video(video, video_fp, fps=15)

    def full_pipeline(
        self,
        prompts: List[str],
        video_fp: str,
        negative_prompts: Optional[List[str]] = None,
        denoising_steps = 50
    ):
        if negative_prompts is not None:
            assert len(negative_prompts) == len(prompts), "Number of negative prompts must match number of prompts"

        prompt_embeds = self.encode_text(prompts)
        negative_prompt_embeds = self.encode_text(negative_prompts)
        if self.guidance_scale >= 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        self.mem("before unload")
        self.unload("text_encoder")
        self.mem("after unload")
        latent = self.generate(len(prompts), prompt_embeds, denoising_steps=denoising_steps)

        self.export_video_from_latent(latent, video_fp)


def run_diffusers():
    from diffusers import AutoencoderKLWan, WanPipeline

    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # prompt = "A cat walks on the grass, realistic"
    # negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # with open("data/prompt.txt", "r") as f:
    #     prompt = f.read()
    # with open("data/negative_prompt.txt", "r") as f:
    #     negative_prompt = f.read()
    prompt_embed = torch.load("data/prompt_embed_wan.pt", weights_only=True)
    negative_prompt_embed = torch.load("data/negative_prompt_embed_wan.pt", weights_only=True)

    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
    vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)

    shape = (
        1,
        16,
        (81 - 1) // vae_scale_factor_temporal+1,
        480 // vae_scale_factor_spatial,
        832 // vae_scale_factor_spatial
    )

    latent = torch.randn(shape, device="cuda", dtype=torch.float16)
    output = pipe(
        prompt_embeds=prompt_embed,
        negative_prompt_embeds=negative_prompt_embed,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0
    ).frames[0]
    export_to_video(output, "official-wan.mp4", fps=15)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # run_diffusers()
    pipeline = WanPipeline("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", pipe_dtype=torch.float32, load_modules=["vae", "dit"])
    # pipeline.prepare_latents()
    # pipeline.unload("vae")
    # pipeline.unload("text_encoder")
    # pipeline.unload("tokenizer")
    # pipeline.mem("after unloading", print_=True)
    latent = torch.load("checkpoints/gen_latents_step_latest.pt")
    for i in tqdm(range(latent.shape[0]), desc="Exporting videos"):
        pipeline.export_video_from_latent(latent[i].unsqueeze(0), f"samples/epoch{i}.mp4")
        print(f"exported video {i+1}/{latent.shape[0]}")



    # with open("data/prompt.txt", "r") as f:
    #     prompt = f.read()
    # with open("data/negative_prompt.txt", "r") as f:
    #     negative_prompt = f.read()

    # pipeline.encode_text([prompt], save_path="data/prompt_embed_wan.pt")
    # pipeline.encode_text([negative_prompt], save_path="data/negative_prompt_embed_wan.pt")

    # pipeline.full_pipeline(
    #     [prompt],
    #     "wan-pipeline-test-10.mp4",
    #     negative_prompts=[negative_prompt],
    #     denoising_steps=10
    # )
