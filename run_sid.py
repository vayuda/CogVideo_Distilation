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
import gc
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.training_utils import compute_snr
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint_sequential

class VideoTrainer:
    def __init__(self, config):
        # Load configuration
        self.config = config
        self.batch_size = config.get("batch_size", 1)
        self.generator_lr = config.get("generator_lr", 1e-5)
        self.student_lr = config.get("student_lr", 1e-5)
        self.teacher_timesteps = config.get("teacher_timesteps", 50)
        self.student_timesteps = config.get("student_timesteps", 1)
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.height = config.get("height", 480)
        self.width = config.get("width", 720)
        self.tmin = config.get("tmin", 20)
        self.tmax = config.get("tmax", 800)
        self.num_epochs = config.get("num_epochs", 10)
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.student_gen_train_ratio = config.get("student_gen_train_ratio", 1)
        self.ckpt_save_path = config.get("ckpt_save_path", None)
        self.pipeline_type = config.get("pipeline_type", "t2v") # or i2v
        # Set up schedulers
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(config["model"], subfolder="scheduler")
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.generator_timesteps = torch.tensor([int(config.get("generator_one_step_time", 999) * (1-i/self.config.get("student_timesteps",1))) for i in range(self.student_timesteps)],dtype=torch.int32)
        self.training_dtype = torch.float16
        print(f"Training with {self.training_dtype}")
        # Setup distributed training
        self.setup_distributed()
        # Initialize models, optimizers, loss
        self.setup_models()
        # load hardcoded target data
        self.load_data()
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
        # i2v@@@model setup
        self.dit = CogVideoXTransformer3DModel.from_pretrained(
            self.config["model"], subfolder="transformer",
            attn_implementation="flash_attention_2",
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
            "amp":{
                "enabled": True
            },
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
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

    def mem(self, place, verbose=True):
        if self.is_main_process and verbose:
            print(f"Mem usage at {place}: {torch.cuda.memory_allocated() / 1024**2}")

    def prepare_model_inputs(self):
        video_latent, image_latent, positive_prompt, negative_prompt = self.batch
        latent = torch.randn_like(video_latent)
        noise = torch.randn_like(video_latent)
        t_index = torch.randint(0, len(self.generator_timesteps), (1,))
        generator_timestep = self.generator_timesteps[t_index].expand(video_latent.shape[0]).to(self.device)
        timesteps = torch.randint(self.tmin, self.tmax, (video_latent.shape[0],)).to(self.device)
        ofs_emb = None
        return latent, noise, positive_prompt, negative_prompt, timesteps, generator_timestep, ofs_emb

    def student_loss(self, xg, noise, t, fake_score):
        if self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(xg, noise, t)
            loss = (fake_score-target)**2
            snr = compute_snr(self.scheduler, t)
            loss = loss * snr/(snr+1)
        else:
            loss = (fake_score-xg)**2

        loss = loss.mean()
        return loss

    def generator_loss(self, teacher_score: torch.Tensor, student_score: torch.Tensor, xg: torch.Tensor):
        alpha = 1.2
        nan_mask = torch.isnan(teacher_score) | torch.isnan(student_score) | torch.isnan(xg)
        if torch.any(nan_mask):
            not_nan_mask = ~nan_mask
            teacher_score = teacher_score[not_nan_mask]
            student_score = student_score[not_nan_mask]
            xg = xg[not_nan_mask]
            print("removed nans")
        teacher_score = teacher_score.to(torch.float32)
        student_score = student_score.to(torch.float32)
        xg = xg.to(torch.float32)

        with torch.no_grad():
            generator_weight = abs(xg - teacher_score).to(torch.float32).mean(dim=[1, 2, 3, 4], keepdim=True).clip(min=0.00001)
        score_diff = teacher_score - student_score
        teacher_eval = teacher_score - xg
        generator_loss = (score_diff) * (teacher_eval - alpha * score_diff) / generator_weight
        generator_loss = generator_loss.mean()

        return generator_loss

    def diffuse(self, model_name, set_grad, latent, prompt, time_step, cfg=False):
        self.mem(f"before {model_name} fwd")
        if model_name == "generator":
            self.engine.module.set_adapter("generator")
        elif model_name == "student"
            self.engine.module.set_adapter("student")
        else:
            self.engine.module.disable_adapter()
        if set_grad:
            output = self.engine(
                hidden_states=latent,
                encoder_hidden_states=prompt,
                timestep=time_step,
                return_dict=False,
            )[0]
        else:
            with torch.inference_mode():
                output = self.engine(
                    hidden_states=latent,
                    encoder_hidden_states=prompt,
                    timestep=time_step,
                    return_dict=False,
                )[0]
        if cfg:
            unc,cond = output.chunk(2)
            return unc + self.guidance_scale * (cond - unc)
        else:
            return output

    def train_step(self):
        total_student_loss = 0
        (
            latent,
            noise,
            positive_prompt,
            negative_prompt,
            timesteps,
            generator_timestep,
            ofs_emb
        ) = self.prepare_model_inputs()

        # Generator forward pass
        latent_model_input = torch.zeros_like(latent)
        latent_model_input = self.scheduler.add_noise(latent_model_input, noise, generator_timestep)
        noise_pred = self.diffuse("generator", False, latent_model_input, positive_prompt, generator_timestep)
        xg = self.scheduler.step(noise_pred, generator_timestep[0], latent, return_dict=False)[1]
        xt = self.scheduler.add_noise(xg.detach(), noise, timesteps)
        # Teacher forward pass (always in eval mode)
        teacher_score = self.diffuse(
            "teacher",
            False,
            torch.cat([xt, xt], dim=0),
            torch.cat([negative_prompt,positive_prompt],dim=0),
            timesteps,
            cfg=True
        )
        # student forward pass and update
        student_score = self.diffuse(
            "student",
            True,
            xt,
            positive_prompt,
            timesteps,
            cfg=False
        )
        student_loss = self.student_loss(xg.detach(), noise, timesteps, student_score)
        total_student_loss = student_loss.detach().item()

        self.engine.backward(student_loss)
        self.engine.step()
        self.mem("student update")

        # update generator
        noise_pred = self.diffuse("generator", True, latent_model_input, positive_prompt, generator_timestep)
        xg = self.scheduler.step(noise_pred, generator_timestep[0], latent, return_dict=False)[1]
        xt = self.scheduler.add_noise(xg.detach(), noise, timesteps)
        # Teacher forward pass (always in eval mode)
        teacher_score = self.diffuse(
            "teacher",
            False,
            torch.cat([xt, xt], dim=0),
            torch.cat([negative_prompt,positive_prompt],dim=0),
            timesteps,
            cfg=True
        )
        # student forward pass and update
        student_score = self.diffuse(
            "student",
            False,
            xt,
            positive_prompt,
            timesteps,
            cfg=False
        )
        generator_loss = self.generator_loss(teacher_score, student_score, xg)
        total_generator_loss = generator_loss.detach().item()
        self.engine.module.set_adapter("generator")
        self.engine.backward(generator_loss)
        self.engine.step()

        self.global_step += 1
        return total_student_loss, total_generator_loss

    def train(self):
        generator_samples = []
        generator_samples.append(self.generate_latent())
        torch.save(torch.cat(generator_samples, dim=0),
              f"{self.ckpt_save_path}/gen_latents_step_latest.pt")
        for epoch in range(self.num_epochs):
            if self.is_main_process:
                print(f"Starting epoch {epoch}")
                loop = tqdm(range(self.config.get("dataset_size")), desc=f"Epoch {epoch}")
            else:
                loop = range(self.config.get("dataset_size"))

            for batch_idx in loop:
                student_loss, generator_loss = self.train_step()

                if self.config.get("use_wandb", False) and self.is_main_process:
                    log_dict = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "student_loss": student_loss,
                    }

                    # Only log generator loss when it's actually updated
                    if self.global_step % self.student_gen_train_ratio == 0:
                        log_dict["generator_loss"] = generator_loss

                    wandb.log(log_dict)

            # On epoch end:
            if self.ckpt_save_path is not None:
                self.engine.save_checkpoint(self.ckpt_save_path, f"model")
                if self.is_main_process:
                    generator_samples.append(self.generate_latent())
                    torch.save(torch.cat(generator_samples, dim=0),
                          f"{self.ckpt_save_path}/gen_latents_step_latest.pt")

    def generate_latent(self):
        (
            latent,
            noise,
            positive_prompt,
            negative_prompt,
            timesteps,
            generator_timestep,
            ofs_emb
        ) = self.prepare_model_inputs()
        with torch.inference_mode():
            self.engine.module.set_adapter("generator")
            for t in tqdm(self.generator_timesteps, desc="generating validation video"):
                latent_model_input = torch.zeros_like(latent)
                latent_model_input = self.scheduler.add_noise(latent_model_input, noise, t * torch.ones(latent_model_input.shape[0], dtype=torch.long, device=self.device))
                timestep = t * torch.ones(latent_model_input.shape[0]).to(self.device)
                noise_pred = self.engine(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=positive_prompt,
                    ofs=ofs_emb,
                    timestep=timestep,
                    return_dict=False,
                )[0]
                out = self.scheduler.step(noise_pred, t, latent, return_dict=False)
                latent = out[0]
                image = out[1]

        return image

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
