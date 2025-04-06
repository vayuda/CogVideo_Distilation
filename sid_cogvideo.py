import yaml
import time
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import os
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
from diffusers.training_utils import compute_snr
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

# Custom dataset to hold our data
class VideoDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_size = config.get("dataset_size", 1)  # Number of iterations per epoch
        self.training_dtype = torch.float16 if config.get("use_fp16", True) else torch.float32

        # Load data
        self.load_data()

    def load_data(self):
        """Load the data and move it to the current device"""
        # Let PyTorch handle device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_latent = torch.load(
            "data/horse_2b_latent_image.pt",
            weights_only=True,
            map_location="cpu"
        ).to(device).to(self.training_dtype)

        self.positive_prompt = torch.load(
            "data/prompt_embed_2b.pt",
            weights_only=True,
            map_location="cpu"
        ).to(device).to(self.training_dtype)

        self.negative_prompt = torch.load(
            "data/negative_prompt_embed_2b.pt",
            weights_only=True,
            map_location="cpu"
        ).to(device).to(self.training_dtype)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return {
            "image_latent": self.image_latent,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "idx": idx  # Adding idx to track which sample we're on
        }

# Custom Trainer that handles LoRA adapters
class VideoTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.global_step = 0
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.student_gen_train_ratio = config.get("student_gen_train_ratio", 2)
        self.tmin = config.get("tmin", 20)
        self.tmax = config.get("tmax", 800)
        self.vlatent_shape = config.get("vlatent_shape", (1, 13, 16, 60, 90))

        # Set up scheduler
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(config["model"], subfolder="scheduler")
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.generator_timesteps = torch.tensor([int(config.get("generator_one_step_time", 999) * (1-i/self.config.get("student_timesteps",1))) for i in range(self.config.get("student_timesteps",1))], dtype=torch.int32)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Determine if we're in student or generator mode based on step
        is_generator_step = (self.global_step % self.student_gen_train_ratio == 0)

        # Get input data
        image_latent = inputs["image_latent"]
        positive_prompt = inputs["positive_prompt"]
        negative_prompt = inputs["negative_prompt"]

        # Use the current device (HF Trainer handles device placement)
        device = next(model.parameters()).device

        # Prepare model inputs
        batch_size = image_latent.shape[0]
        latent = torch.randn(batch_size, *self.vlatent_shape[1:]).to(device)
        noise = torch.randn_like(latent)

        t_index = torch.randint(0, len(self.generator_timesteps), (1,), device=device)
        generator_timestep = self.generator_timesteps[t_index].expand(batch_size).to(device)
        timesteps = torch.randint(self.tmin, self.tmax, (batch_size,), device=device)

        # Generator forward pass
        latent_model_input = torch.zeros_like(latent)
        latent_model_input = self.scheduler.add_noise(latent_model_input, noise, generator_timestep)

        # Set appropriate adapter
        model.set_adapter("generator")
        with torch.inference_mode():
            noise_pred = model(
                hidden_states=latent_model_input,
                encoder_hidden_states=positive_prompt,
                timestep=generator_timestep,
                return_dict=False,
            )[0]

        xg = self.scheduler.step(noise_pred, generator_timestep[0], latent, return_dict=False)[1]
        xt = self.scheduler.add_noise(xg.detach(), noise, timesteps)

        # Teacher forward pass (always in eval mode)
        with torch.inference_mode():
            model.disable_adapter()
            teacher_output = model(
                hidden_states=torch.cat([xt, xt], dim=0),
                encoder_hidden_states=torch.cat([negative_prompt, positive_prompt], dim=0),
                timestep=torch.cat([timesteps, timesteps], dim=0),
                return_dict=False,
            )[0]

            # Apply classifier-free guidance
            unc, cond = teacher_output.chunk(2)
            teacher_score = unc + self.guidance_scale * (cond - unc)

        if is_generator_step:
            # Generator update step
            model.set_adapter("generator")
            noise_pred = model(
                hidden_states=latent_model_input,
                encoder_hidden_states=positive_prompt,
                timestep=generator_timestep,
                return_dict=False,
            )[0]

            xg = self.scheduler.step(noise_pred, generator_timestep[0], latent, return_dict=False)[1]
            xt = self.scheduler.add_noise(xg, noise, timesteps)

            # Get student prediction (without gradients)
            with torch.inference_mode():
                model.set_adapter("student")
                student_score = model(
                    hidden_states=xt,
                    encoder_hidden_states=positive_prompt,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

            # Compute generator loss
            loss = self.generator_loss(teacher_score, student_score, xg)

            if self.is_local_process_zero() and self.config.get("use_wandb", False):
                wandb.log({"generator_loss": loss.item(), "step": self.global_step})
        else:
            # Student update step
            model.set_adapter("student")
            student_score = model(
                hidden_states=xt,
                encoder_hidden_states=positive_prompt,
                timestep=timesteps,
                return_dict=False,
            )[0]

            loss = self.student_loss(xg.detach(), noise, timesteps, student_score)

            if self.is_local_process_zero() and self.config.get("use_wandb", False):
                wandb.log({"student_loss": loss.item(), "step": self.global_step})

        self.global_step += 1
        return loss

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

    def generator_loss(self, teacher_score, student_score, xg):
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

    def generate_sample(self):
        """Generate a sample latent for visualization"""
        model = self.model
        model.set_adapter("generator")

        # Use the current device
        device = next(model.parameters()).device

        # Create latent tensors
        latent = torch.randn(1, *self.vlatent_shape[1:]).to(device)
        noise = torch.randn_like(latent)

        # Get a batch of prompts
        dataset = self.get_train_dataloader().dataset
        positive_prompt = dataset.positive_prompt

        with torch.inference_mode():
            for t in tqdm(self.generator_timesteps, desc="generating validation video"):
                latent_model_input = torch.zeros_like(latent)
                latent_model_input = self.scheduler.add_noise(
                    latent_model_input,
                    noise,
                    t * torch.ones(latent_model_input.shape[0], dtype=torch.long, device=device)
                )
                timestep = t * torch.ones(latent_model_input.shape[0]).to(device)

                noise_pred = model(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=positive_prompt,
                    timestep=timestep,
                    return_dict=False,
                )[0]

                out = self.scheduler.step(noise_pred, t, latent, return_dict=False)
                latent = out[0]
                image = out[1]

        return image

    def on_train_begin(self, *args, **kwargs):
        """Save initial generated sample"""
        if self.is_local_process_zero():
            generator_sample = self.generate_sample()
            torch.save(generator_sample, f"{self.args.output_dir}/gen_latents_step_initial.pt")

    def on_train_end(self, *args, **kwargs):
        """Save final generated sample and models"""
        if self.is_local_process_zero():
            generator_sample = self.generate_sample()
            torch.save(generator_sample, f"{self.args.output_dir}/gen_latents_step_final.pt")

            # Save final models
            self.model.set_adapter("generator")
            self.model.save_pretrained(f"{self.args.output_dir}/generator_final")

            self.model.set_adapter("student")
            self.model.save_pretrained(f"{self.args.output_dir}/student_final")


def main():
    # Enable high precision matrix multiplication
    torch.set_float32_matmul_precision('high')

    # Load configuration
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    # Create dataset
    dataset = VideoDataset(config)

    # Create model
    model = CogVideoXTransformer3DModel.from_pretrained(
        config["model"],
        subfolder="transformer",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16 if config.get("use_fp16", True) else torch.float32
    )

    # Disable gradients for non-LoRA params
    for param in model.parameters():
        param.requires_grad = False

    # Setup LoRA
    lora_config = LoraConfig(**config['lora'])
    model = get_peft_model(model, lora_config)
    model.add_adapter("generator", lora_config)
    model.add_adapter("student", lora_config)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.get("ckpt_save_path", "./output"),
        num_train_epochs=config.get("num_epochs", 10),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.get("student_lr", 1e-5),
        fp16=config.get("use_fp16", True),
        save_strategy="epoch",
        logging_dir=f"{config.get('ckpt_save_path', './output')}/logs",
        logging_steps=10,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        report_to="wandb" if config.get("use_wandb", False) else "none",
        # Let HF Trainer handle distributed training
    )

    # Initialize WandB if needed
    if config.get("use_wandb", False) and training_args.local_rank <= 0:
        wandb.init(
            entity=config.get("wandb_entity", "SiD_pawan"),
            project=config.get("wandb_project", "CogVideoX-SiD-Distillation"),
            name=config.get("wandb_run_name", "test"),
            group="multi-gpu",
            config=config
        )

    # Create trainer
    trainer = VideoTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Cleanup WandB
    if config.get("use_wandb", False) and training_args.local_rank <= 0:
        wandb.finish()


if __name__ == "__main__":
    main()
