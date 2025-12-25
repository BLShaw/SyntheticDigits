import torch
import torch.optim as optim
import torch.nn as nn
import os
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.syntheticdigits.config.schema import AppConfig
from src.syntheticdigits.models.networks import Generator, Discriminator, initialize_weights
import logging

logger = logging.getLogger(__name__)

class GANTrainer:
    def __init__(self, config: AppConfig, dataloader):
        self.config = config
        self.dataloader = dataloader
        self.device = self._get_device()
        
        # Initialize Models
        self.gen = Generator(
            config.model.noise_dim, 
            config.model.img_channels, 
            config.model.features_g,
            config.model.num_classes,
            config.model.embed_size
        ).to(self.device)
        
        self.disc = Discriminator(
            config.model.img_channels, 
            config.model.features_d,
            config.model.num_classes,
            config.model.img_size
        ).to(self.device)
        
        initialize_weights(self.gen)
        initialize_weights(self.disc)
        
        # Optimizers
        self.opt_gen = optim.Adam(
            self.gen.parameters(), 
            lr=config.training.lr, 
            betas=(config.training.beta1, config.training.beta2)
        )
        self.opt_disc = optim.Adam(
            self.disc.parameters(), 
            lr=config.training.lr, 
            betas=(config.training.beta1, config.training.beta2)
        )
        
        # Fixed noise and labels for visualization
        self.fixed_noise = torch.randn(100, config.model.noise_dim).to(self.device)
        self.fixed_labels = torch.arange(10).repeat_interleave(10).to(self.device)
        
        os.makedirs(os.path.join(config.output.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(config.output.output_dir, "checkpoints"), exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=os.path.join(config.output.output_dir, "logs"))

    def _get_device(self):
        if self.config.training.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.training.device)

    def load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.gen.load_state_dict(checkpoint["gen"])
        self.disc.load_state_dict(checkpoint["disc"])
        self.opt_gen.load_state_dict(checkpoint["opt_gen"])
        self.opt_disc.load_state_dict(checkpoint["opt_disc"])
        
        return checkpoint.get("epoch", 0) + 1

    def train(self, start_epoch=0):
        self.gen.train()
        self.disc.train()
        
        # Turn dataloader into an iterator to handle the critic loop manually
        data_iter = iter(self.dataloader)
        
        step = start_epoch * (len(self.dataloader) // self.config.training.critic_iterations)
        
        print(f"Starting training from epoch {start_epoch}...")
        
        for epoch in range(start_epoch, self.config.training.epochs):
            # Total batches / critic_iterations roughly
            num_gen_steps = len(self.dataloader) // self.config.training.critic_iterations
            loop = tqdm(range(num_gen_steps), leave=True)
            
            for _ in loop:
                # ---------------------
                # Train Discriminator
                # ---------------------
                for _ in range(self.config.training.critic_iterations):
                    try:
                        real, labels = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.dataloader)
                        real, labels = next(data_iter)
                        
                    real = real.to(self.device)
                    labels = labels.to(self.device)
                    cur_batch_size = real.shape[0]
                    
                    noise = torch.randn(cur_batch_size, self.config.model.noise_dim).to(self.device)
                    fake = self.gen(noise, labels)
                    
                    disc_real = self.disc(real, labels).reshape(-1)
                    disc_fake = self.disc(fake.detach(), labels).reshape(-1)
                    
                    gp = self.gradient_penalty(real, fake.detach(), labels)
                    
                    loss_disc = (
                        -(torch.mean(disc_real) - torch.mean(disc_fake)) + 
                        self.config.training.lambda_gp * gp
                    )
                    
                    self.opt_disc.zero_grad()
                    loss_disc.backward()
                    self.opt_disc.step()

                # -----------------
                # Train Generator
                # -----------------
                self.opt_gen.zero_grad()
                
                gen_noise = torch.randn(cur_batch_size, self.config.model.noise_dim).to(self.device)
                gen_fake = self.gen(gen_noise, labels)
                gen_output = self.disc(gen_fake, labels).reshape(-1)
                
                loss_gen = -torch.mean(gen_output)
                
                loss_gen.backward()
                self.opt_gen.step()
                
                # Update progress bar & TensorBoard
                if step % 10 == 0:
                    loop.set_description(f"Epoch [{epoch}/{self.config.training.epochs}]")
                    loop.set_postfix(loss_d=loss_disc.item(), loss_g=loss_gen.item())
                    
                    self.writer.add_scalar("Loss/Discriminator", loss_disc.item(), step)
                    self.writer.add_scalar("Loss/Generator", loss_gen.item(), step)
                
                step += 1

            # End of Epoch Actions
            self._on_epoch_end(epoch)
        
        self.writer.close()

    def gradient_penalty(self, real, fake, labels):
        batch_size, C, H, W = real.shape
        alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
        interpolated_images = real * alpha + fake * (1 - alpha)
        
        # Enable gradient calculation for interpolated images
        interpolated_images.requires_grad_(True)
        
        # Mixed scores
        mixed_scores = self.disc(interpolated_images, labels)
        
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def _on_epoch_end(self, epoch):
        # Save Images
        if epoch % self.config.output.save_image_interval == 0:
            with torch.no_grad():
                self.gen.eval()
                fake = self.gen(self.fixed_noise, self.fixed_labels)
                # Denormalize: [-1, 1] -> [0, 1]
                data = fake * 0.5 + 0.5
                
                # Log to TensorBoard
                grid = vutils.make_grid(data, nrow=10, normalize=False)
                self.writer.add_image("Generated Images", grid, epoch)
                
                # Save to disk
                path = os.path.join(self.config.output.output_dir, "images", f"epoch_{epoch}.png")
                vutils.save_image(data, path, normalize=False, nrow=10)
                self.gen.train()
                
        # Save Model
        if epoch % self.config.output.save_model_interval == 0 or epoch == self.config.training.epochs - 1:
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        checkpoint = {
            "gen": self.gen.state_dict(),
            "disc": self.disc.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
            "epoch": epoch,
            "config": self.config.model_dump()
        }
        path = os.path.join(self.config.output.output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")