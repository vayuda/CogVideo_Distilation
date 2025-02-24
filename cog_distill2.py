
import copy
import yaml
import time
import wandb
from tqdm import tqdm
from contextlib import nullcontext
import deepspeed
import torch
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from lightning.fabric import Fabric

from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from peft import LoraConfig, get_peft_model



class SiD_Loss(torch.nn.Module):
    def __init__(self):
        super(SiD_Loss, self).__init__()
        self.alpha = 1.2

    def forward(self,
                teacher_score: torch.Tensor,
                student_score: torch.Tensor,
                xg: torch.Tensor,
                xt: torch.Tensor,
                sigma_t: torch.Tensor,
                loss_type='student'):

        nan_mask = torch.isnan(teacher_score) | torch.isnan(student_score) | torch.isnan(xg)
        if torch.any(nan_mask):
            not_nan_mask = ~nan_mask
            teacher_score = teacher_score[not_nan_mask]
            student_score = student_score[not_nan_mask]
            xg = xg[not_nan_mask]
            print("removed nans")

        # student score loss
        if loss_type == 'student':
            student_weight = sigma_t[:, None, None, None, None]
            student_loss = student_weight * (student_score - xg) ** 2
            student_loss = student_loss
            return student_loss

        # generator loss
        else:
            with torch.no_grad():
                generator_weight = abs(xt - teacher_score).to(torch.float32).mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)
            generator_loss = (teacher_score - student_score) * ((teacher_score - xg) - self.alpha * (teacher_score - student_score)) / generator_weight
            generator_loss = generator_loss.sum()
            return generator_loss


class VideoTrainer:
    def __init__(self, fabric, config):
        # Load configuration
        self.config = config
        self.batch_size = config.get("batch_size", 2)
        self.generator_lr = config.get("generator_lr", 1e-4)
        self.student_lr = config.get("student_lr", 1e-4)
        self.teacher_timesteps = config.get("teacher_timesteps", 50)
        self.student_timesteps = config.get("student_timesteps", 1)
        self.sigma_init = config.get("sigma_init", 0.5)
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.height = config.get("height", 256)
        self.width = config.get("width", 256)
        self.tmax = config.get("tmax", 800)
        self.num_epochs = config.get("num_epochs", 10)
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Setup Fabric
        self.fabric = fabric

        # Initialize models, optimizers, loss
        self.setup_models()
        self.pt_dtype = torch.bfloat16
        print(f"using dtype {self.pt_dtype}")
        self.sid_loss = SiD_Loss()

        # Set up schedulers.
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(config["model"],subfolder="scheduler")
        self.student_scheduler = copy.deepcopy(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps=self.teacher_timesteps) # 50
        self.student_scheduler.set_timesteps(num_inference_steps=self.student_timesteps) # 1

        alphas_cumprod = self.student_scheduler.alphas_cumprod
        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.sigmas = self.sigmas.to(device = self.fabric.device, dtype= self.pt_dtype)
        distances = torch.abs(self.sigmas - self.sigma_init)
        self.generator_one_step_time = torch.argmin(distances)

        # Set up wandb
        if self.config.get("use_wandb", False) and self.fabric.is_global_zero:
            wandb.init(
                project=self.config.get("wandb_project", "cogvideo-distillation"),
                name=self.config.get("wandb_run_name", "test"),
                config=self.config
            )


    def setup_models(self):
        # Create models
        self.teacher = CogVideoXTransformer3DModel.from_pretrained(
            self.config["model"], subfolder="transformer",
            torch_dtype = torch.bfloat16
        )
        self.teacher = self.teacher.to(self.fabric.device)
        self.student = copy.deepcopy(self.teacher)
        self.generator = copy.deepcopy(self.teacher)
        self.student.gradient_checkpointing = True
        # self.generator.gradient_checkpointing = True

        # Setup LoRA
        lora_config = LoraConfig(**self.config['lora'])
        self.student = get_peft_model(self.student, lora_config)
        self.generator = get_peft_model(self.generator, lora_config)

        # make optimizers
        self.student_opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config["student_lr"]
        )
        self.generator_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config["generator_lr"]
        )


        # Setup student and generator with fabric
        self.student, self.student_opt = self.fabric.setup(self.student, self.student_opt)
        self.generator, self.generator_opt = self.fabric.setup(self.generator, self.generator_opt)
        print("student is using ", self.student.dtype, "generator is using ", self.generator.dtype)
        # Initially freeze student and generator to save memory
        self.student.eval().requires_grad_(False)
        self.generator.eval().requires_grad_(False)
        self.teacher.eval().requires_grad_(False)


        self.global_step = 0


    def get_sigma(self, timestep):
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        return ((1 - alpha_t) / alpha_t) ** 0.5

    def merge_latents(self, video_latent, image_latent):
        # For classifer free guidance
        # latent_model_input = video_latent.repeat(1,1,2,1,1) # b, t, c, h, w
        # latent_image_input = image_latent.repeat(1,1,2,1,1)
        latent_model_input = self.scheduler.scale_model_input(video_latent, self.generator_one_step_time)
        return torch.cat([latent_model_input, image_latent], dim=2).to(self.fabric.device, dtype=self.pt_dtype) # across the channel dimension

    def prepare_model_inputs(self, batch):
        for tensor in batch:
            tensor =  tensor.to(self.fabric.device, dtype=self.pt_dtype)
        video_latent, image_latent, prompt_embeds = batch
        prompt_embeds = prompt_embeds
        timesteps = torch.randint(0, self.tmax, (video_latent.shape[0],)).to(self.fabric.device)
        latent_model_input = self.merge_latents(video_latent, image_latent)
        timestep = self.generator_one_step_time.expand(latent_model_input.shape[0]).to(self.fabric.device)
        ofs_emb = latent_model_input.new_full((1,), fill_value=2.0).to(self.fabric.device, dtype=self.pt_dtype)
        return video_latent, image_latent, latent_model_input, prompt_embeds, timesteps, timestep, ofs_emb

    def run_models(
        self,
        video_latent,
        image_latent,
        latent_model_input,
        prompt_embeds,
        timesteps,
        generator_timestep,
        ofs_emb,
        is_training_student=True,
    ):
        student_ctx = torch.inference_mode() if not is_training_student else nullcontext()
        generator_ctx = torch.inference_mode() if is_training_student else nullcontext()
        with generator_ctx:
            noise_pred = self.generator(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                ofs = ofs_emb,
                timestep=generator_timestep,
                return_dict=False,
            )[0]
        xg = self.student_scheduler.step(noise_pred, self.generator_one_step_time, video_latent, return_dict=False)[0]
        noise = torch.randn_like(video_latent).to(self.fabric.device, dtype=self.pt_dtype)
        xt = self.scheduler.add_noise(xg, noise, timesteps)

        # Teacher forward pass
        hd = self.merge_latents(xt, image_latent)
        with torch.inference_mode():
            teacher_score = self.teacher(
                hidden_states=hd,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                ofs = ofs_emb,
                return_dict=False,
            )[0]

        # Student forward pass
        with student_ctx:
            student_score = self.student(
                hidden_states=self.merge_latents(xg, image_latent),
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                ofs = ofs_emb,
                return_dict=False,
            )[0]
        sigma_t=self.sigmas[timesteps]
        loss_type = 'student' if is_training_student else 'generator'

        return teacher_score, student_score, xg, xt, sigma_t, loss_type

    def train_step(self, batch):
        timing_stats = {}
        total_start = start =  time.time()
        batch_size = batch[0].shape[0] // self.accumulation_steps
        accumulated_batches = [
            tuple(t.chunk(self.accumulation_steps)[i] for t in batch)
            for i in range(self.accumulation_steps)
        ]
        # Train student
        self.student.train().requires_grad_(True)
        self.student_opt.zero_grad(set_to_none=True)
        total_student_loss = 0
        for step, sub_batch in enumerate(accumulated_batches):
            with self.fabric.no_backward_sync(self.student, enabled=(step != self.accumulation_steps - 1)):
                student_loss = self.sid_loss(
                    *self.run_models(
                        *self.prepare_model_inputs(sub_batch),
                        is_training_student=True
                    )
                )
                total_student_loss += student_loss.detach()
                self.fabric.backward(student_loss/self.accumulation_steps)
                self.fabric.print(f"Student Loss: {student_loss:.4f}, time: {time.time() - start}")

        opt_time = time.time()
        self.student_opt.step()
        timing_stats["student_step_time"] = time.time() - opt_time
        self.student.eval().requires_grad_(False)  # Free up memory
        timing_stats["student_train_time"] = time.time() - start

        # Train generator
        start = time.time()
        self.generator.train().requires_grad_(True)
        self.generator_opt.zero_grad(set_to_none=True)
        total_generator_loss = 0
        for step, sub_batch in enumerate(accumulated_batches):
            with self.fabric.no_backward_sync(self.student, enabled=(step != self.accumulation_steps - 1)):
                generator_loss = self.sid_loss(
                    *self.run_models(
                        *self.prepare_model_inputs(sub_batch),
                        is_training_student=True
                    )
                )
                total_generator_loss += generator_loss.detach()
                self.fabric.backward(generator_loss)

        self.generator_opt.step()
        self.generator.eval().requires_grad_(False)  # Free up memory
        timing_stats["generator_train_time"] = time.time() - start
        timing_stats["total_time"] = time.time() - total_start
        self.global_step += 1
        return total_student_loss / self.accumulation_steps , total_generator_loss / self.accumulation_steps, timing_stats


    def train(self, train_loader):
        # Setup dataloader with fabric
        train_loader = self.fabric.setup_dataloaders(train_loader)

        for epoch in range(self.num_epochs):
            self.fabric.print(f"Starting epoch {epoch}")
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch_idx, batch in enumerate(train_loader):
                student_loss, generator_loss, timing_stats = self.train_step(batch)

                # Logging
                self.fabric.print(
                    f"Epoch {epoch} Step {batch_idx}: "
                    f"Student Loss: {student_loss:.4f}, "
                    f"Generator Loss: {generator_loss:.4f}"
                )
                self.fabric.print(timing_stats)
                if self.config.get("use_wandb", False):
                    wandb.log({
                        "step": self.global_step,
                        "epoch": epoch,
                        "student_loss": student_loss.item(),
                        "generator_loss": generator_loss.item(),
                    })

class RepeatedTensorDataset(Dataset):
    def __init__(self, video_latent, image_latent, encoded_prompt, num_copies):
        self.video_latent = video_latent
        self.image_latent = image_latent
        self.encoded_prompt = encoded_prompt
        self.num_copies = num_copies

    def __len__(self):
        return self.num_copies

    def __getitem__(self, idx):

        return self.video_latent, self.image_latent, self.encoded_prompt

def setup_dataloader(config):
    # Load configuration
    dataset_size = config.get("dataset_size", 1)
    # Load latent tensors and text prompt
    video_latent = torch.load("horse_riding_latents.pt", weights_only=True).squeeze(0)
    image_latent = torch.load("horse_riding_image_latents.pt", weights_only=True).squeeze(0)
    encoded_prompt = torch.load("horse_riding_text_prompt.pt", weights_only=True).squeeze(0)

    # while we only have one image for debug
    dataset = RepeatedTensorDataset(
            video_latent,
            image_latent,
            encoded_prompt,
            dataset_size
        )

    base_batch_size = config["batch_size"]
    accumulation_steps = config["gradient_accumulation_steps"]
    effective_batch_size = base_batch_size * accumulation_steps
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    return dataloader


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    fabric = Fabric(
        accelerator="cuda",
        devices=config.get("devices", 0),
        strategy="deepspeed_stage_2_offload" if config.get("ngpu", 1) > 1 else "auto",
        precision="bf16-mixed"
    )
    fabric.launch()
    trainer = VideoTrainer(fabric, config)
    train_loader = setup_dataloader(config)
    trainer.train(train_loader)

    # Cleanup wandb
    if trainer.config.get("use_wandb", False) and trainer.fabric.is_global_zero:
        wandb.finish()

    # Cleanup Fabric
    fabric.barrier()
    fabric.cleanup()

if __name__ == "__main__":
    main()
