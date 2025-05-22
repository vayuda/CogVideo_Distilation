import yaml
from tqdm import tqdm
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger
from diffsynth import FlowMatchScheduler
from diffusers import WanTransformer3DModel
from diffusers.training_utils import compute_snr
from peft import LoraConfig, inject_adapter_in_model, get_peft_model
from diffusers.utils.import_utils import is_xformers_available

# Create a random latent dataset
class LatentDataset(Dataset):
    def __init__(self, num_samples, latent_shape, positive_prompt, negative_prompt, device=None):
        """
        Initialize a LatentDataset object.

        Parameters
        ----------
        num_samples : int
            The number of latent samples to generate.
        latent_shape : tuple
            The shape of the latent tensor.
        positive_prompt : str
            The positive prompt for the generated samples.
        negative_prompt : str
            The negative prompt for the generated samples.
        device : torch.device, optional
            The device to generate the samples on. Defaults to None, which uses the default torch device.
        """
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
        self.generator_lr = float(config.get("generator_lr", 1e-6))
        self.fake_score_lr = float(config.get("student_lr", 1e-6))
        self.generator_timesteps = config.get("generator_timesteps", 1)
        self.guidance_scale = config.get("guidance_scale", 5.0)
        self.height = config.get("height", 480)
        self.width = config.get("width", 720)
        self.tmin = config.get("tmin", 20)
        self.tmax = config.get("tmax", 800)
        self.num_epochs = config.get("num_epochs", 10)
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.ckpt_directory = config.get("ckpt_directory", None)
        self.run_name = config.get("wandb_run_name", "test")
        self.device = fabric.device
        self.fabric = fabric

        # Set up schedulers
        self.scheduler = FlowMatchScheduler(shift=3.0, sigma_min=0.0, extra_one_step=True)  
        self.scheduler.set_timesteps(num_inference_steps=self.scheduler.num_train_timesteps, training=True)
        self.nt = self.config.get("generator_timesteps", 1)
        self.inference_scheduler = FlowMatchScheduler(shift=3.0, sigma_min=0.0, extra_one_step=True)

        # Initialize models, optimizers, loss
        self.setup_models()
        # load hardcoded target data
        self.load_data()
        self.global_step = 0

        #load checkpoint if exists
        checkpoint_path = config.get("checkpoint_path", None)
        if checkpoint_path is not None:
            self.fabric.print(f"Loading checkpoint from {checkpoint_path} ...")
            self.load_checkpoint(checkpoint_path)
        

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
        self.positive_prompt = self.positive_prompt.repeat(self.batch_size,1,1)
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
        if verbose:
            self.fabric.print(f"Mem usage at {place}: {torch.cuda.memory_allocated() / 1024**2}")

    def fake_score_loss(self, xg, noise, t, fake_score):
        xt = self.scheduler.add_noise(xg.detach(), noise, t.to(torch.float))
        target  = self.scheduler.training_target(xt, noise, t)   # ε - x_t
        weight  = self.scheduler.training_weight(t)              # scalar per sample
        loss    = ((fake_score - target) ** 2) * weight

        nan = torch.isnan(loss).flatten(start_dim=1).any(dim=1)
        if nan.any():
            self.fabric.print(f"Found {nan.sum()} nans in fake score loss")
            loss = loss[~nan]
        return loss.mean()

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
        nan = torch.isnan(generator_loss).flatten(start_dim=1).any(dim=1)
        if nan.any():
            self.fabric.print(f"Found {nan.sum()} nans in generator loss")
            generator_loss = generator_loss[~nan]
        return generator_loss.mean()

    def diffuse(self, model_name, train, latent, prompt, time_step, cfg=False):
        timestep = time_step.repeat(latent.shape[0])
        self.mem(f"before {model_name} fwd")
        if model_name == "generator":
            self.dit.set_adapter("generator")
        elif model_name == "fake":
            self.dit.set_adapter("fsm")
        else:
            self.dit.disable_adapter()
        with torch.set_grad_enabled(train):
            noise_pred = self.dit(
                hidden_states=latent,
                encoder_hidden_states=prompt,
                timestep=timestep,
                return_dict=False,
            )[0]
            
        if cfg:
            unc,cond = noise_pred.chunk(2)
            return unc + self.guidance_scale * (cond - unc)
        else:
            return noise_pred
        

    def run_generator(self,latent=None,train=False, use_cfg=True):
        self.dit.set_adapter("generator")
        if latent is None:
            latent = torch.randn((1,16,21,60,104)).to(self.device)
        prompt_embeds = self.cfg_prompt.repeat(latent.shape[0],1,1)
        self.inference_scheduler.set_timesteps(num_inference_steps=self.nt)
        for t in self.inference_scheduler.timesteps:
            latent_model_input = torch.cat([latent] * 2)
            timestep = t.expand(latent.shape[0]).to(self.device)
            with torch.set_grad_enabled(train):
                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    return_dict=False,
                )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latent = self.inference_scheduler.step(noise_pred, t, latent, return_dict=False)
            if not isinstance(latent, torch.Tensor):
                latent = latent[0]
        return latent
    
    
    def train_step(self, latent):
        for _ in range(self.accumulation_steps):
            noise = torch.randn_like(latent)
            # Generator forward pass
            xg = self.run_generator(latent)
            timesteps = torch.randint(self.tmin, self.tmax, (1,)).to(self.device).to(torch.long)
            xt = self.scheduler.add_noise(xg.detach(), noise, timesteps)

            # student forward pass and update
            fake_score = self.diffuse(
                "fake",
                True,
                torch.cat([xt, xt], dim=0),
                self.cfg_prompt,
                timesteps,
                cfg=True
            )
            fake_model_loss = self.fake_score_loss(xg.detach(), noise, timesteps, fake_score)
            with self.fabric.autocast():                   # handles AMP/FP16
               fake_model_loss /= self.accumulation_steps
            self.mem("before fake score model update")
            
            self.fabric.backward(fake_model_loss)   

        self.fsm_optimizer.step()
        self.fsm_optimizer.zero_grad(set_to_none=True)
        self.mem("after fake score model update")

        # update generator
        for _ in range(self.accumulation_steps):
            noise = torch.randn_like(latent)
            xg = self.run_generator(latent,train=True)
            timesteps = torch.randint(self.tmin, self.tmax, (1,)).to(self.device)
            xt = self.scheduler.add_noise(xg.detach(), noise, timesteps)
            self.mem("after generator fwd")
            # Teacher forward pass
            real_score = self.diffuse(
                "real",
                False,
                torch.cat([xt, xt], dim=0),
                self.cfg_prompt,
                timesteps,
                cfg=True
            )
            # student forward pass
            fake_score = self.diffuse(
                "fake",
                False,
                torch.cat([xt, xt], dim=0),
                self.cfg_prompt,
                timesteps,
                cfg=True
            )
            self.dit.set_adapter("generator")
            generator_loss = self.generator_loss(real_score, fake_score, xg)
            with self.fabric.autocast():                   # handles AMP/FP16
               generator_loss /= self.accumulation_steps
            
            self.fabric.backward(generator_loss)

        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return fake_model_loss.detach().item(), generator_loss.detach().item()

    def train(self):
        generator_samples = []
        if self.fabric.is_global_zero:
            generator_samples.append(self.run_generator())
            torch.save(torch.cat(generator_samples, dim=0),
                f"{self.ckpt_directory}/start_video.pt")
            print("generated start of training video")
        for epoch in range(self.num_epochs):
            loop = tqdm(self.dataloader) if self.fabric.is_global_zero else self.dataloader
            for latent in loop:
                student_loss, generator_loss = self.train_step(latent)
                # print(f"Epoch {epoch} step {self.global_step} student loss: {student_loss:.4f} generator loss: {generator_loss:.4f}")
                log_dict = {
                    "step": self.global_step,
                    "epoch": epoch,
                    "student_loss": student_loss,
                    "generator_loss": generator_loss
                }

                self.fabric.log_dict(log_dict)

            # On epoch end:
            if self.ckpt_directory is not None:
                self.save_checkpoint()
            
                if self.fabric.is_global_zero:
                    print(f"End of epoch {epoch} generating video ...")
                    with torch.inference_mode():
                        generator_samples.append(self.run_generator())
                    torch.save(torch.cat(generator_samples, dim=0), 
                        f"{self.ckpt_directory}/gen_latents_latest.pt")
            self.fabric.barrier()

    def save_checkpoint(self):
        checkpoint_dir = f"{self.ckpt_directory}/{self.run_name}.ckpt"
        state = {
            "dit": self.dit,
            "generator_optimizer": self.generator_optimizer,
            "fsm_optimizer": self.fsm_optimizer,
            "global_step": self.global_step
        }
        self.fabric.save(checkpoint_dir, state)

    def load_checkpoint(self, path):
        state = {
            "dit": self.dit,
            "generator_optimizer": self.generator_optimizer,
            "fsm_optimizer": self.fsm_optimizer,
            "global_step": self.global_step
        }
        self.fabric.load(path, state)
        self.global_step = state["global_step"]

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
        devices=config.get("n_devices", 1),
        strategy='ddp'
    )
    fabric.launch()

    trainer = VideoTrainer(config, fabric)
    trainer.train()

if __name__ == "__main__":
    main()
