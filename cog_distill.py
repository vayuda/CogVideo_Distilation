import copy
from peft.tuners.hra.layer import HRAConv2d
import torch
import yaml
import pytorch_lightning as pl
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import pdb
import gc

class SiD_Loss(torch.nn.Module):
    def __init__(self):
        super(SiD_Loss, self).__init__()
        self.sigma_data = 0.5
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
            student_weight = (sigma_t ** 2 + self.sigma_data ** 2) / (sigma_t * self.sigma_data) ** 2
            student_loss = student_weight * (student_score - xg) ** 2
            student_loss = student_loss.sum()
            return student_loss

        # generator loss
        else:
            with torch.no_grad():
                generator_weight = abs(xt - teacher_score).to(torch.float32).mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)
            generator_loss = (teacher_score - student_score) * ((teacher_score - xt) - self.alpha * (teacher_score - student_score)) / generator_weight
            generator_loss = generator_loss.sum()
            return generator_loss

def get_tensor_sizes():
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                total += obj.nelement() * obj.element_size()
                if obj.nelement() * obj.element_size() > 1e8:  # Print tensors over ~100MB
                    print(f"Large tensor: {obj.shape}, {obj.nelement() * obj.element_size()/1e9:.2f} GB")
        except:
            pass
    print(f"Total tensor memory: {total/1e9:.2f} GB")

def print_gpu_memory():
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Utility to find the closest timestep given a scheduler and sigma_init.
def get_one_step_t(scheduler, sigma_init):
    alphas_cumprod = scheduler.alphas_cumprod
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    distances = torch.abs(sigmas - sigma_init)
    return torch.argmin(distances)

def print_model_memory(model, name="model"):
    total_params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024/1024 #MB
    print(f"{name}: {param_size:.2f}MB, {total_params} parameters")

#image rotary embeddings
def prepare_rotary_positional_embeddings(config: dict, device: torch.device):
    grid_height = config.get("height", 480) // ( 8 * 2)
    grid_width = config.get("width", 720) // ( 8 * 2) # vae scale * transformer patch size reduction

    base_num_frames = config.get("num_frames", 48) // 2

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=64, # attention head dim from https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/transformer/config.json
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )

    return freqs_cos, freqs_sin



# Define the LightningModule.
class VideoLightningModule(pl.LightningModule):
    def __init__(self, config_path="config.yaml"):
        super().__init__()
        # Load configuration
        config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        self.config = config

        # Save hyperparameters if needed
        self.batch_size = config["batch_size"]
        self.generator_lr = config["generator_lr"]
        self.student_lr = config["student_lr"]
        self.teacher_timesteps = config["teacher_timesteps"]
        self.student_timesteps = config["student_timesteps"]
        self.sigma_init = config["sigma_init"]
        self.guidance_scale = config["guidance_scale"]
        self.height = config["height"]
        self.width = config["width"]
        self.tmax = config["tmax"]
        # Create student and generator as deep copies of teacher.
        self.teacher = CogVideoXTransformer3DModel.from_pretrained(config["model"],subfolder="transformer")
        self.student = copy.deepcopy(self.teacher)
        self.student.gradient_checkpointing = True
        self.generator = copy.deepcopy(self.student)
        self.generator.gradient_checkpointing = True

        lora_config = LoraConfig(
            r = config['lora']['r'],
            lora_alpha = config['lora']['lora_alpha'],
            target_modules= config['lora']['target_modules'],
        )
        for name, param in self.student.named_parameters():
            param.requires_grad = False
        for name, param in self.generator.named_parameters():
            param.requires_grad = False
        self.student = get_peft_model(self.student, lora_config)
        self.generator = get_peft_model(self.generator, lora_config)

        # Freeze teacher parameters since it is only used for computing targets.
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False



        # Set up schedulers.
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(config["model"],subfolder="scheduler")
        self.student_scheduler = copy.deepcopy(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps=self.teacher_timesteps)
        self.student_scheduler.set_timesteps(num_inference_steps=self.student_timesteps)
        # Compute one step time for the generator using the student scheduler.
        self.generator_one_step_time = get_one_step_t(self.student_scheduler, self.sigma_init)
        # print(f"one step time for generator: {self.generator_one_step_time}")
        # Define the loss.
        self.sid_loss = SiD_Loss()

        # We'll use manual optimization to update both student and generator.
        self.automatic_optimization = False
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
    def get_sigma(self, timestep):
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        return ((1 - alpha_t) / alpha_t) ** 0.5
    def prepare_latent_for_dit(self, latent_model_input, latent_image_input):

        # Concatenate video latents twice and scale using the scheduler.
        # latent_model_input = video_latent.repeat(1,1,1,1,1)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, self.generator_one_step_time)
        # Concatexnate image latent (repeated twice) along the channel dimension.
        # latent_image_input = image_latent.repeat(1,1,1,1,1)
        # print("latent_model_input shape", latent_model_input.shape, "latent_image_input shape", latent_image_input.shape)
        latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
        return latent_model_input

    def noise_pred_to_score(self, noise_pred, timestep, scheduler):
        # Compute sigma for the given timestep and convert noise prediction to a score.
        alpha_t = scheduler.alphas_cumprod[timestep]
        sigma_t = ((1 - alpha_t) / alpha_t) ** 0.5
        return -noise_pred / sigma_t.reshape(-1, 1, 1, 1, 1)

    def training_step(self, batch, batch_idx):
        # Unpack the batch (video_latent, image_latent, prompt_embeds).
        video_latent, image_latent, prompt_embeds = batch
        # move to device:
        video_latent = video_latent
        image_latent = image_latent
        prompt_embeds = prompt_embeds
        batch_size = video_latent.shape[0]
        # print(f"shapes: video_latent {video_latent.shape}, image_latent {image_latent.shape}, prompt_embeds {prompt_embeds.shape}")
        # Prepare latent inputs for the generator.
        latent_model_input = self.prepare_latent_for_dit(video_latent, image_latent)
        # image_rotary_embedding = prepare_rotary_positional_embeddings(
        #     self.config, self.device
        # )
        timestep = self.generator_one_step_time.expand(latent_model_input.shape[0]).to(self.device).to(torch.float16)
        # Generator forward pass.
        ofs_emb = latent_model_input.new_full((1,), fill_value=2.0)
        noise_pred = self.generator(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            # image_rotary_emb=image_rotary_embedding,
            ofs = ofs_emb,
            timestep=timestep,
            return_dict=False,
        )[0]
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=2)
        # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        # Use the student scheduler to perform a single denoising step.
        xg = self.student_scheduler.step(noise_pred, self.generator_one_step_time, video_latent, return_dict=False)[0]

        # Create the noisy target xt.
        timesteps = torch.randint(0, self.tmax, (batch_size,), device=self.device)
        noise = torch.randn_like(video_latent).to(self.device)
        xt = self.scheduler.add_noise(xg, noise, timesteps)

        # Teacher forward pass.
        self.teacher = self.teacher.to(self.device)
        hd = self.prepare_latent_for_dit(xt, image_latent)
        with torch.inference_mode():
            teacher_noise_prediction = self.teacher(
                hidden_states=hd,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                ofs = ofs_emb,
                # image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
        teacher_score = self.noise_pred_to_score(teacher_noise_prediction, timesteps, self.scheduler)
        self.student = self.student.to(self.device)
        # Student forward pass.
        student_noise_prediction = self.student(
            hidden_states=self.prepare_latent_for_dit(xg, image_latent),
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            ofs = ofs_emb,
            # image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        student_score = self.noise_pred_to_score(student_noise_prediction, timesteps, self.scheduler)

        # Compute the losses.
        sigma_t=self.get_sigma(timesteps).reshape(-1,1,1,1,1)
        student_loss = self.sid_loss(
            teacher_score, student_score, xg, xt,
            sigma_t=sigma_t, loss_type='student'
        )
        generator_loss = self.sid_loss(
            teacher_score, student_score, xg, xt,
            sigma_t=sigma_t, loss_type='generator'
        )

        # Manual optimization: zero grads, backpropagate each loss, and update optimizers.
        opt_gen, opt_student = self.optimizers()
        if (self.global_step + 1) % self.accumulation_steps == 0:
            opt_gen.zero_grad()
            opt_student.zero_grad()

        self.manual_backward(student_loss, retain_graph=True)
        self.manual_backward(generator_loss)

        # Only step optimizers after accumulating enough gradients
        if (self.global_step + 1) % self.accumulation_steps == 0:
            opt_gen.step()
            opt_student.step()
            # Log the losses.
            self.log("train/student_loss", student_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/generator_loss", generator_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return student_loss + generator_loss

    def configure_optimizers(self):
        optimizer_gen = torch.optim.AdamW(self.generator.parameters(), lr=self.generator_lr)
        optimizer_student = torch.optim.AdamW(self.student.parameters(), lr=self.student_lr)
        return [optimizer_gen, optimizer_student]


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

def setup_dataloader(config_path="config.yaml"):
    # Load configuration
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    batch_size = config.get("batch_size", 1)
    dataset_size = config.get("dataset_size", 1)
    # Load latent tensors and text prompt
    video_latent = torch.load("horse_riding_latents.pt", weights_only=True).squeeze(0)
    image_latent = torch.load("horse_riding_image_latents.pt", weights_only=True).squeeze(0)
    encoded_prompt = torch.load("horse_riding_text_prompt.pt", weights_only=True).squeeze(0)

    # Create dataset and dataloader
    dataset = RepeatedTensorDataset(
            video_latent,
            image_latent,
            encoded_prompt,
            dataset_size
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    # Load configuration
    # from diffusers import CogVideoXImageToVideoPipeline
    # model_path = "THUDM/CogVideoX1.5-5B-I2V"
    # pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')
    # pdb.set_trace()
    config_path = "config.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # Set up the dataloader
    train_loader = setup_dataloader(config_path)

    # Instantiate your Lightning module (which includes manual optimization)
    model = VideoLightningModule(config_path=config_path)

    # Configure the Trainer. You can adjust devices, ddp strategy, precision, etc.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.get("devices", 0),   # Use multiple GPUs if available
        strategy="deepspeed_stage_2_offload" if config.get("ngpu", 1) > 1 else "auto",
        max_steps=config["num_steps"],
        precision="16-mixed",  # Enables mixed precision training
    )

    # Start training
    trainer.fit(model, train_loader)
    torch.save(model.generator.state_dict(), "generator_model.pth")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
