import copy
from transformers.models.sam.image_processing_sam import F
import yaml
import time
import wandb
import os
import gc
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.fabric import Fabric
from lightning.fabric.strategies import FSDPStrategy
import wandb
from wandb.integration.lightning.fabric import WandbLogger
from torch.utils.checkpoint import checkpoint_sequential

from diffusers.models import WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from diffsynth import FlowMatchScheduler
from diffusers.training_utils import compute_snr
from peft import LoraConfig, get_peft_model

# Create a random latent dataset
class LatentDataset(Dataset):
    def __init__(self, num_samples, latent_shape, positive_prompt, negative_prompt, device=None):
        self.num_samples = num_samples
        self.latent_shape = latent_shape
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.device = device
        self.video_latents = [torch.randn(self.latent_shape) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random latents and prompts for each item
        return self.video_latents[idx]

class VideoTrainer:
    def __init__(self, config, fabric):
        # Load configuration
        self.config = config
        self.batch_size = config.get("batch_size", 1)
        self.dataset_size = self.config.get("dataset_size", 1000)
        self.generator_lr = config.get("generator_lr", 1e-5)
        self.fake_score_lr = config.get("student_lr", 1e-5)
        self.generator_timesteps = config.get("generator_timesteps", 1)
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
        # Set up scheduler
        self.scheduler = FlowMatchScheduler(shift=3.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_inference_steps=self.scheduler.num_train_timesteps)
        nt = self.config.get("generator_timesteps", 1)
        self.generator_timesteps = torch.tensor(
            [config.get("generator_one_step_time", 999) * (1.0-i/nt) for i in range(nt)],
            dtype=torch.float32
        )

        # Setup distributed training
        self.setup_distributed(fabric)

        # Initialize models, optimizers, loss
        self.setup_models()

        # load hardcoded target data
        self.load_data()
        self.global_step = 0

    def setup_distributed(self, fabric):
        self.device = fabric.device
        self.fabric = fabric

    def setup_models(self):
        # Create models
        self.dit = WanTransformer3DModel.from_pretrained(
            self.config["model"], subfolder="transformer",
            attn_implementation="flash_attention_2",
        )

        # Disable gradients for non-LoRA params
        for name, param in self.dit.named_parameters():
            param.requires_grad = False

        # Setup LoRA
        lora_config = LoraConfig(**self.config['lora'])
        self.dit = get_peft_model(self.dit, lora_config)
        self.dit.add_adapter("generator", lora_config) # one step generator
        self.dit.add_adapter("fsm", lora_config) # fake score model
        self.dit.enable_gradient_checkpointing()
        # Retrieve optimizable parameters
        self.dit.set_adapter("generator")
        generator_params = [p for p in self.dit.parameters() if p.requires_grad]
        generator_optimizer = torch.optim.AdamW(generator_params, lr=self.generator_lr)
        self.dit.set_adapter("fsm")
        fake_score_model_params = [p for p in self.dit.parameters() if p.requires_grad]
        fake_score_optimizer = torch.optim.AdamW(fake_score_model_params, lr=self.fake_score_lr)
        # setup model and optimizers for distributed training with fabric
        (
            self.dit,
            self.generator_optimizer,
            self.fsm_optimizer
        ) = self.fabric.setup(self.dit, generator_optimizer, fake_score_optimizer)

        fake_score_model_size = sum(p.numel() * p.element_size() for p in fake_score_model_params) / (1024 * 1024)
        generator_size = sum(p.numel() * p.element_size() for p in generator_params) / (1024 * 1024)
        self.fabric.loggers[0].watch(self.dit)
        if self.fabric.is_global_zero:
            print(f"Found {len(fake_score_model_params)} fake score model params that require grad: {fake_score_model_size:.2f}MB")
            print(f"Found {len(generator_params)} generator params that require grad: {generator_size:.2f}MB")

    def load_data(self):
        self.positive_prompt = torch.load(
            "data/prompt_embed_wan.pt",
            weights_only=True,
            map_location="cpu"
        ).to(self.device)

        self.negative_prompt = torch.load(
            "data/negative_prompt_embed_wan.pt",
            weights_only=True,
            map_location="cpu"
        ).to(self.device)
        self.cfg_prompt = torch.cat([self.negative_prompt,self.positive_prompt],dim=0).to(self.device)
        latent_shape = (16,21,60,104)
        dataset = LatentDataset(
            self.dataset_size,
            latent_shape,
            self.positive_prompt,
            self.negative_prompt,
            device=self.device
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        self.batch_iterator = iter(self.dataloader)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)

    def mem(self, place, verbose=False):
        if self.fabric.is_global_zero and verbose:
            print(f"Mem usage at {place}: {torch.cuda.memory_allocated() / 1024**2}")

    def fake_score_loss(self, xg, noise, t, fake_score):
        loss = (fake_score - xg)**2
        # if self.scheduler.config.prediction_type == "v_prediction":
        #     target = self.scheduler.get_velocity(xg, noise, t)
        #     loss = (fake_score-target)**2
        #     snr = compute_snr(self.scheduler, t)
        #     loss = loss * snr/(snr+1)
        # else:
        #     loss = (fake_score-xg)**2

        loss = loss.mean()
        return loss

    def generator_loss(self, real_score: torch.Tensor, fake_score: torch.Tensor, xg: torch.Tensor):
        alpha = 1.2
        nan_mask = torch.isnan(real_score) | torch.isnan(fake_score) | torch.isnan(xg)
        if torch.any(nan_mask):
            not_nan_mask = ~nan_mask
            real_score = real_score[not_nan_mask]
            fake_score = fake_score[not_nan_mask]
            xg = xg[not_nan_mask]
            print("removed nans")
        real_score = real_score.to(torch.float32)
        fake_score = fake_score.to(torch.float32)
        xg = xg.to(torch.float32)

        with torch.no_grad():
            generator_weight = abs(xg - real_score).to(torch.float32).mean(dim=[1, 2, 3, 4], keepdim=True).clip(min=0.00001)
        score_diff = real_score - fake_score
        teacher_eval = real_score - xg
        generator_loss = (score_diff) * (teacher_eval - alpha * score_diff) / generator_weight
        generator_loss = generator_loss.mean()

        return generator_loss

    def diffuse(self, model_name, set_grad, latent, prompt, time_step, cfg=False):
        self.mem(f"before {model_name} fwd")
        if model_name == "generator":
            self.dit.set_adapter("generator")
        elif model_name == "fake":
            self.dit.set_adapter("fsm")
        else:
            self.dit.disable_adapter()
        if set_grad:
            output = self.dit(
                hidden_states=latent,
                encoder_hidden_states=prompt,
                timestep=time_step,
                return_dict=False,
            )[0]
        else:
            with torch.inference_mode():
                output = self.dit(
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

    def train_step(self, latent):
        total_student_loss = 0
        noise = torch.randn_like(latent)
        # Generator forward pass
        xg = self.run_generator(latent, use_cfg=False)

        timesteps = torch.randint(self.tmin, self.tmax, (latent.shape[0],)).to(self.device)
        timesteps = torch.randint(self.tmin, self.tmax, (latent.shape[0],)).to(self.device)
        xt = self.scheduler.add_noise(xg.detach(), noise, timesteps.to(torch.float))

        # student forward pass and update
        fake_score = self.diffuse(
            "fake",
            True,
            xt,
            self.positive_prompt,
            timesteps,
            cfg=False
        )
        fake_model_loss = self.fake_score_loss(xg.detach(), noise, timesteps, fake_score)
        total_student_loss = fake_model_loss.detach().item()

        self.fsm_optimizer.zero_grad()
        self.fabric.backward(fake_model_loss)
        self.fsm_optimizer.step()
        self.mem("fake score model update")

        # update generator
        xg = self.run_generator(latent,train=True, use_cfg=False)
        timesteps = torch.randint(self.tmin, self.tmax, (latent.shape[0],)).to(self.device)
        xt = self.scheduler.add_noise(xg.detach(), noise, timesteps.to(torch.float))
        # Teacher forward pass (always in eval mode)
        real_score = self.diffuse(
            "real",
            False,
            torch.cat([xt, xt], dim=0),
            self.cfg_prompt,
            timesteps,
            cfg=True
        )
        # student forward pass and update
        fake_score = self.diffuse(
            "fake",
            False,
            xt,
            self.positive_prompt,
            timesteps,
            cfg=False
        )
        generator_loss = self.generator_loss(real_score, fake_score, xg)
        total_generator_loss = generator_loss.detach().item()
        self.dit.set_adaptmer("generator")
        self.generator_optimizer.zero_grad()
        self.fabric.backward(generator_loss)
        self.generator_optimizer.step()
        self.global_step += 1
        return total_student_loss, total_generator_loss

    def train(self):
        generator_samples = []
        if self.fabric.is_global_zero:
            generator_samples.append(self.run_generator())
            torch.save(torch.cat(generator_samples, dim=0),
                f"{self.ckpt_save_path}/gen_latents_step_latest.pt")
        self.fabric.barrier()
        for epoch in range(self.num_epochs):
            for latent in tqdm(self.dataloader):
                student_loss, generator_loss = self.train_step(latent)

                if self.config.get("use_wandb", False) and self.fabric.is_global_zero:
                    log_dict = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "student_loss": student_loss,
                        "generator_loss": generator_loss
                    }

                    wandb.log(log_dict)

            # On epoch end:
            if self.ckpt_save_path is not None and self.fabric.is_global_zero:
                self.save_checkpoint()
                generator_samples.append(self.run_generator())
                torch.save(torch.cat(generator_samples, dim=0), f"{self.ckpt_save_path}/gen_latents_step_latest.pt")
            self.fabric.barrier()

    def run_generator(self,latent=None,train=False, use_cfg=True):
        self.dit.set_adapter("generator")
        if latent is None:
            latent = torch.randn((1,16,21,60,104)).to(self.device)
        for t in self.generator_timesteps:
            timestep = t * torch.ones(latent.shape[0]).to(self.device)
            latent_model_input = torch.cat([latent] * 2) if use_cfg else latent
            prompt = self.cfg_prompt if use_cfg else self.positive_prompt
            if not train:
                with torch.inference_mode():
                    noise_pred = self.dit(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt,
                        timestep=timestep,
                        return_dict=False,
                    )[0]
            else:
                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt,
                    timestep=timestep,
                    return_dict=False,
                )[0]
            if use_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)

        return latent

    def save_checkpoint(self):
        checkpoint_dir = f"{self.ckpt_save_path}/checkpoints_latest"
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {"model": self.dit}
        self.fabric.save(checkpoint_dir, state)

    def load_checkpoint(self, path):
        state = {"model": self.dit}
        self.fabric.load(path, state)

def main():
    torch.set_float32_matmul_precision('high')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    logger = WandbLogger(
        project=config.get("wandb_project", "CogVideoX-SiD-Distillation"),
        entity=config.get("wandb_entity", "SiD_pawan"),
        name=config.get("wandb_run_name", "test"),
        group="multi-gpu",
        config=config
    )
    fabric = Fabric(
        loggers=logger,
        precision="bf16-mixed",
        accelerator="cuda",
        devices=4,
        strategy='ddp'
    )
    fabric.launch()

    trainer = VideoTrainer(config, fabric)
    trainer.train()

if __name__ == "__main__":
    main()
