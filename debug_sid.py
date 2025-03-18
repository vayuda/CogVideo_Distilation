import copy
import yaml
import time
import wandb
from tqdm import tqdm
import deepspeed
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os
from transformers import AutoTokenizer, Trainer, TrainingArguments, PreTrainedTokenizer
import transformers
from diffusers.video_processor import VideoProcessor
import pathlib
from diffusers.utils import export_to_video
import torch
from omegaconf import OmegaConf as om
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler, AutoencoderKLCogVideoX
from diffusers.training_utils import compute_snr
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint

def print_tensor_stats(name, tensor):
    if torch.distributed.get_rank() == 0:
        print(f"{name} [{tensor.min().item():.3f}, {tensor.max().item():.3f}]  "
            f"µ: {tensor.mean().item():.3f}, ∑: {tensor.std().item():.3f}")
        if torch.any(torch.isnan(tensor)):
            print(f"WARNING: NaN detected in {name}")
        if torch.any(torch.isinf(tensor)):
            print(f"WARNING: Inf detected in {name}")

class SiD_Loss(torch.nn.Module):
    def __init__(self):
        super(SiD_Loss, self).__init__()
        self.alpha = 1.2
        self.is_main_process = int(os.environ.get("RANK", 0)) == 0

    def forward(
        self,
        teacher_score,
        student_score: torch.Tensor,
        xg: torch.Tensor,
        xt: torch.Tensor,
        sigma_t: torch.Tensor,
        target=None,
        snr=None,
        loss_type='student',
        ):
        # nan_mask = torch.isnan(teacher_score) | torch.isnan(student_score) | torch.isnan(xg)
        # if torch.any(nan_mask):
        #     not_nan_mask = ~nan_mask
        #     teacher_score = teacher_score[not_nan_mask]
        #     student_score = student_score[not_nan_mask]
        #     xg = xg[not_nan_mask]
        #     print("removed nans")
        student_score = student_score.to(torch.float32)
        xg = xg.to(torch.float32)
        xt = xt.to(torch.float32)
        sigma_t = sigma_t.to(torch.float32)
        # print_tensor_stats("teacher_score", teacher_score)
        # print_tensor_stats("student_score", student_score)
        # print_tensor_stats("xg", xg)

        # student score loss
        if loss_type == 'student':
            student_weight = sigma_t[:, None, None, None, None]
            loss = (student_score-target)**2
            loss = loss * snr/(snr+1)
            student_loss = (student_score - xg) ** 2
            student_loss = student_loss.mean()
            return student_loss

        # generator loss
        else:
            teacher_score = teacher_score.to(torch.float32)
            print(teacher_score.shape)
            with torch.no_grad():
                generator_weight = abs(xg - teacher_score).to(torch.float32).mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)
            score_diff = teacher_score - student_score
            teacher_eval = teacher_score - xg
            generator_loss = (score_diff) * ((teacher_eval) - self.alpha * (score_diff)) / generator_weight
            # print_tensor_stats("generator loss", generator_loss)
            # print_tensor_stats("student teacher diff", score_diff)
            # print_tensor_stats("generator teacher diff", teacher_eval)

            if self.is_main_process:
                wandb.log({
                    "score diff": (score_diff**2).mean().item(),
                    "teacher eval": (teacher_eval**2).mean().item(),
                    "generator weight": (generator_weight**2).mean().item(),
                })
            generator_loss = generator_loss.mean()
            return generator_loss


class VideoTrainer(Trainer):
    def __init__(self, config):
        # Load configuration
        self.config = config
        self.batch_size = config.get("batch_size", 2)
        self.generator_lr = config.get("generator_lr", 1e-4)
        self.student_lr = config.get("student_lr", 1e-4)
        self.teacher_timesteps = config.get("teacher_timesteps", 50)
        self.student_timesteps = config.get("student_timesteps", 1)
        self.sigma_init = config.get("sigma_init", 0.25)
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.height = config.get("height", 256)
        self.width = config.get("width", 256)
        self.tmin = config.get("tmin", 20)
        self.tmax = config.get("tmax", 980)
        self.num_epochs = config.get("num_epochs", 10)
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.student_gen_train_ratio = config.get("student_gen_train_ratio", 1)
        self.ckpt_save_path = config.get("ckpt_save_path", None)
        self.pipeline_type = config.get("pipeline_type", "t2v") # or i2v
        # Setup distributed training
        self.setup_distributed()
        # Initialize models, optimizers, loss
        self.setup_models()
        self.training_dtype = torch.float16
        print(f"Training with {self.training_dtype}")
        self.sid_loss = SiD_Loss()

        # load hardcoded target data
        self.load_data()

        # Set up schedulers
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(config["model"], subfolder="scheduler")
        self.student_scheduler = copy.deepcopy(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps=self.teacher_timesteps)  # 50
        self.student_scheduler.set_timesteps(num_inference_steps=self.student_timesteps)

        alphas_cumprod = self.scheduler.alphas_cumprod
        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.sigmas = self.sigmas.to(device=self.device, dtype=self.training_dtype)
        self.generator_timesteps = self.student_scheduler.timesteps

        self.global_step = 0

    def setup_distributed(self):
        """
        Sets up distributed training environment.
        This version works with the DeepSpeed launcher which initializes process groups.
        """
        # When launched with deepspeed, LOCAL_RANK is set
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.is_main_process = (self.local_rank == 0)
            self.device = torch.device(f"cuda:{self.local_rank}")

            # Set the device for this process
            torch.cuda.set_device(self.local_rank)

        else:
            # Fallback for single-GPU operation or when not using DeepSpeed launcher
            self.local_rank = 0
            self.world_size = 1
            self.is_main_process = True
            self.device = torch.device("cuda:0")

        print(f"Process rank: {self.local_rank}/{self.world_size}, device: {self.device}")

    def setup_models(self):
        # Create models

        self.vae = AutoencoderKLCogVideoX.from_pretrained(self.config["model"], subfolder="vae", torch_dtype=torch.float16).to(self.device)
        self.vae_config = self.vae.config
        self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # i2v@@@model setup
        self.dit = CogVideoXTransformer3DModel.from_pretrained(
            self.config["model"], subfolder="transformer",
            torch_dtype=torch.float16
        )
        # Disable gradients for non-LoRA params
        for name, param in self.dit.named_parameters():
            param.requires_grad = False

        # Setup LoRA
        lora_config = LoraConfig(**self.config['lora'])
        self.dit = get_peft_model(self.dit, lora_config)
        self.dit.add_adapter("generator", lora_config)
        self.dit.add_adapter("student", lora_config)


        # Retrieve optimizable parameters
        self.dit.set_adapter("generator")
        generator_params = [p for p in self.dit.parameters() if p.requires_grad]

        self.dit.set_adapter("student")
        student_params = [p for p in self.dit.parameters() if p.requires_grad]

        student_param_size = sum(p.numel() * p.element_size() for p in student_params) / (1024 * 1024)
        generator_param_size = sum(p.numel() * p.element_size() for p in generator_params) / (1024 * 1024)


        # Configure DeepSpeed Zero2 Config
        ds_config = {
            "fp16": {
                "enabled": "False",
            },
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.generator_lr,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            }
        }

        ds_config = {
            "fp16": {
                "enabled": "True",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.generator_lr,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto"
            }
        }

        self.engine = deepspeed.initialize(
            model = self.dit,
            config = ds_config,
            model_parameters = student_params + generator_params
        )[0]

        if self.is_main_process:
            print(self.dit.active_adapters)
            print(f"Found {len(student_params)} student params that require grad: {student_param_size:.2f}MB")
            print(f"Found {len(generator_params)} generator params that require grad: {generator_param_size:.2f}MB")

            if self.config.get("use_wandb", False):
                wandb.init(
                    entity=self.config.get("wandb_entity", "SiD_pawan"),
                    project=self.config.get("wandb_project", "CogVideoX-SiD-Distillation"),
                    name=self.config.get("wandb_run_name", "test"),
                    group="multi-gpu",
                    config=self.config
                )
                wandb.watch(self.engine.module)

        deepspeed.comm.barrier()

    def load_data(self):
        video_latent = torch.load(
            "data/horse_2b_latent.pt",
            weights_only=True,
            map_location="cpu"
        ).to(self.device).to(self.training_dtype)

        image_latent = torch.load(
            "data/horse_2b_latent_image.pt",
            weights_only=True,
            map_location="cpu"
        ).to(self.device).to(self.training_dtype)

        positive_prompt = torch.load(
            "data/prompt_embed_2b.pt",
            weights_only=True,
            map_location="cpu"
        ).to(self.device).to(self.training_dtype)

        negative_prompt = torch.load(
            "data/negative_prompt_embed_2b.pt",
            weights_only=True,
            map_location="cpu"
        ).to(self.device).to(self.training_dtype)

        # Create dataset with repeated tensors
        self.batch = (video_latent, image_latent, positive_prompt, negative_prompt)

    def mem(self, place, verbose=False):
        if self.is_main_process and verbose:
            print(f"Mem usage at {place}: {torch.cuda.memory_allocated() / 1024**2}")

    def prepare_model_inputs_gen(self):
        video_latent, image_latent, positive_prompt, negative_prompt = self.batch
        video_latent = video_latent * self.student_scheduler.init_noise_sigma
        #t_index = torch.randint(0, len(self.generator_timesteps), (1,))

        t_init = torch.tensor(650, dtype=torch.long) # hardcode init_t
        generator_timestep = t_init.expand(video_latent.shape[0]).to(self.device)
        #generator_timestep = self.generator_timesteps[t_index].expand(video_latent.shape[0]).to(self.device)

        latent_model_input = self.student_scheduler.scale_model_input(video_latent, generator_timestep)
        # concat the image latent along the channel dimension if provided
        if self.pipeline_type == "i2v":
            latent_model_input = torch.cat([latent_model_input, image_latent], dim=2)

        timesteps = torch.randint(self.tmin, self.tmax, (video_latent.shape[0],)).to(self.device)

        if "1.5" in self.config.get("model", ""):
            ofs_emb = latent_model_input.new_full((1,), fill_value=2.0).to(self.device, dtype=self.training_dtype)
        else:
            ofs_emb = None
        return video_latent, image_latent, latent_model_input, positive_prompt, negative_prompt, timesteps, generator_timestep, ofs_emb

    def prepare_model_inputs(self):
        video_latent, image_latent, positive_prompt, negative_prompt = self.batch
        video_latent = video_latent * self.student_scheduler.init_noise_sigma
        t_index = torch.randint(0, len(self.generator_timesteps), (1,))
        generator_timestep = self.generator_timesteps[t_index].expand(video_latent.shape[0]).to(self.device)

        latent_model_input = self.student_scheduler.scale_model_input(video_latent, generator_timestep)
        # concat the image latent along the channel dimension if provided
        if self.pipeline_type == "i2v":
            latent_model_input = torch.cat([latent_model_input, image_latent], dim=2)
        timesteps = torch.randint(0, self.tmax, (video_latent.shape[0],)).to(self.device)

        if "1.5" in self.config.get("model", ""):
            ofs_emb = latent_model_input.new_full((1,), fill_value=2.0).to(self.device, dtype=self.training_dtype)
        else:
            ofs_emb = None
        return video_latent, image_latent, latent_model_input, positive_prompt, negative_prompt, timesteps, generator_timestep, ofs_emb

    def train_step(self):
        timing_stats = {}
        total_start = start = time.time()
        total_student_loss = 0
        (
            video_latent,
            image_latent,
            latent_model_input,
            positive_prompt,
            negative_prompt,
            timesteps,
            generator_timestep,
            ofs_emb
        ) = self.prepare_model_inputs_gen()

        self.engine.module.set_adapter("generator")
        self.mem("start of run models")

        # Generator forward pass
        noise_pred = self.engine(
            hidden_states=latent_model_input,
            encoder_hidden_states=positive_prompt,
            ofs=ofs_emb,
            timestep=generator_timestep,
            return_dict=False,
        )[0]
        timing_stats['gen fwd'] = time.time() - start
        start = time.time()
        self.mem("gen fwd")

        xg = self.student_scheduler.step(noise_pred, generator_timestep[0], video_latent, return_dict=False)[0]
        #noise = 0.5 * torch.ones_like(video_latent, dtype=self.training_dtype, device=self.device)
        noise = torch.rand_like(video_latent, dtype=self.training_dtype, device=self.device)


        xt = self.scheduler.add_noise(xg, noise, timesteps).detach()
        sigma_t = self.sigmas[timesteps]


        ### Update student fake diffusion model


        # Set student mode
        self.engine.module.set_adapter("student")
        student_score = self.engine(
            hidden_states=xt,
            encoder_hidden_states=positive_prompt,
            timestep=timesteps,
            ofs=ofs_emb,
            return_dict=False,
        )[0]

        timing_stats['student fwd'] = time.time() - start
        start = time.time()
        self.mem("student fwd")

        target = self.student_scheduler.get_velocity(xg, noise, timesteps)
        snr = compute_snr(self.student_scheduler, timesteps)
        student_loss = self.sid_loss(None, student_score, xg.detach(), xt, sigma_t, target, snr, loss_type="student")
        self.engine.backward(student_loss)
        total_student_loss = student_loss.detach().item()
        self.engine.step()
        timing_stats['student update'] = time.time() - start
        start = time.time()
        self.mem("student update")


        ### Update one-step generator

        # Teacher forward pass (always in eval mode)
        self.engine.module.disable_adapter()
        with torch.inference_mode():
            teacher_scores = self.engine(
                hidden_states=torch.cat([xt, xt], dim=0),
                encoder_hidden_states=torch.cat([negative_prompt, positive_prompt], dim=0),
                timestep=timesteps,
                ofs=ofs_emb,
                return_dict=False,
            )[0]
            teacher_score_uncond, teacher_score_text = teacher_scores.chunk(2)
            teacher_score = teacher_score_uncond + self.guidance_scale * (teacher_score_text - teacher_score_uncond)

        timing_stats['teacher fwd'] = time.time() - start
        start = time.time()
        self.mem("teacher fwd")

        # Student forward pass (always in eval mode)
        self.engine.module.set_adapter("student")
        with torch.inference_mode():
            student_score = self.engine(
                hidden_states=xt,
                encoder_hidden_states=positive_prompt,
                timestep=timesteps,
                ofs=ofs_emb,
                return_dict=False,
            )[0]
        timing_stats['student fwd2'] = time.time() - start
        start = time.time()
        self.mem("student fwd2")


        nan_mask_images = torch.isnan(xg).flatten(start_dim=1).any(dim=1)
        nan_mask_y_real = torch.isnan(teacher_score).flatten(start_dim=1).any(dim=1)
        nan_mask_y_fake = torch.isnan(student_score).flatten(start_dim=1).any(dim=1)
        nan_mask = nan_mask_images | nan_mask_y_real | nan_mask_y_fake

        # Check if there are any NaN values present
        if nan_mask.any():
            print("find nan!!!!!!")

        generator_loss = self.sid_loss(teacher_score, student_score, xg, xt, sigma_t, loss_type="generator")
        total_generator_loss = generator_loss.detach().item()
        self.engine.module.set_adapter("generator")
        self.engine.backward(generator_loss)
        self.engine.step()

        timing_stats['gen update'] = time.time() - start
        start = time.time()
        self.mem("gen update")

        self.global_step += 1
        timing_stats["total update time"] = time.time() - total_start

        return total_student_loss, total_generator_loss, timing_stats, teacher_score, student_score

    def train(self):
        generator_samples = []
        for epoch in range(self.num_epochs):
            if self.is_main_process:
                print(f"Starting epoch {epoch}")
                loop = tqdm(range(self.config.get("dataset_size")), desc=f"Epoch {epoch}")
            else:
                loop = range(self.config.get("dataset_size"))

            for batch_idx in loop:
                student_loss, generator_loss, timing_stats, teacher_score, student_score = self.train_step()

                if self.config.get("use_wandb", False) and self.is_main_process:
                    log_dict = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "student_loss": student_loss,
                        "student-teacher-diff": teacher_score - student_score
                    }

                    # Only log generator loss when it's actually updated
                    if self.global_step % self.student_gen_train_ratio == 0:
                        log_dict["generator_loss"] = generator_loss

                    wandb.log(log_dict)

            # On epoch end:
            if self.ckpt_save_path is not None:
                self.engine.save_checkpoint(self.ckpt_save_path, f"model")
                if self.is_main_process:
                    latent = self.generate_latent()
                    #generator_samples.append()
                    self.export_video_from_latent(latent, "samples/")

                    #torch.save(torch.cat(generator_samples, dim=0),
                          #f"{self.ckpt_save_path}/gen_latents_step_latest.pt")

    def export_video_from_latent(self, latents, video_fp):
        assert self.vae is not None, "VAE model is not initialized"
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents
        frames = self.vae.decode(latents).sample
        video = self.video_processor.postprocess_video(frames).squeeze(0)
        fps = 16 if "1.5" in self.hf_name else 8
        export_to_video(video, video_fp, fps=fps)

    def generate_latent(self):
        (
            video_latent,
            image_latent,
            latent_model_input,
            positive_prompt,
            negative_prompt,
            _,
            generator_timestep,
            ofs_emb
        ) = self.prepare_model_inputs()
        with torch.inference_mode():
            self.engine.module.set_adapter("generator")
            self.student_scheduler.set_timesteps(num_inference_steps=self.student_timesteps)
            prompt = torch.cat([negative_prompt, positive_prompt],dim=0)
            for t in tqdm(self.student_scheduler.timesteps, desc="generating validation video"):
                latent_model_input = self.scheduler.scale_model_input(video_latent, t)
                latent_model_input = torch.cat([latent_model_input,latent_model_input],dim=0)
                timestep = t.expand(latent_model_input.shape[0]).to(self.device)
                noise_pred = self.engine(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt,
                    ofs=ofs_emb,
                    timestep=timestep,
                    return_dict=False,
                )[0]
                a, b = noise_pred.chunk(2)
                noise_pred = a + self.guidance_scale * (b - a)
                video_latent = self.student_scheduler.step(noise_pred, t, video_latent, return_dict=False)[0]

        return video_latent

    def save_checkpoint(self, generator_samples):
        """Save model checkpoint"""
        # DeepSpeed has its own checkpointing mechanism
        checkpoint_dir = f"{self.ckpt_save_path}/ds_checkpoints_latest"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save DeepSpeed checkpoint states
        self.engine.save_checkpoint(checkpoint_dir, "model")

        # Save additional data
        torch.save({
            'global_step': self.global_step,
        }, f"{self.ckpt_save_path}/ckpt_latest_meta.pt")

        # Save generated samples
        torch.save(torch.cat(generator_samples, dim=0),
                  f"{self.ckpt_save_path}/gen_latents_step_latest.pt")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        # Load DeepSpeed checkpoints
        _, client_state = self.engine.load_checkpoint(f"{path}/ds_checkpoints", "model")

        # Load additional metadata
        meta = torch.load(f"{path}/ckpt_latest_meta.pt")
        self.global_step = meta['global_step']


def main():
    torch.set_float32_matmul_precision('high')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    # DeepSpeed will initialize the distributed environment
    trainer = VideoTrainer(config)
    trainer.train()

    # Cleanup wandb
    if trainer.config.get("use_wandb", False) and trainer.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
