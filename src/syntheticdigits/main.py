import argparse
import yaml
import torch
from src.syntheticdigits.config.schema import AppConfig
from src.syntheticdigits.data.datamodule import MNISTDataModule
from src.syntheticdigits.training.trainer import GANTrainer
from syntheticdigits.models.networks import Generator
import os
import torchvision.utils as vutils
from syntheticdigits.utils.logging_utils import setup_logging

def train(args):
    # Load config from YAML if provided, else use defaults
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Parse into Pydantic model (validates types)
        config = AppConfig(**config_dict)
    else:
        config = AppConfig()
        
    # Overrides
    if args.epochs:
        config.training.epochs = args.epochs
    if args.output_dir:
        config.output.output_dir = args.output_dir

    print(f"Configuration:\n{config.model_dump_json(indent=2)}")
    
    # Data
    datamodule = MNISTDataModule(config.data, config.training)
    dataloader = datamodule.get_dataloader()
    
    # Train
    trainer = GANTrainer(config, dataloader)
    
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
             start_epoch = trainer.load_checkpoint(args.resume)
             print(f"Resuming training from epoch {start_epoch}")
        else:
             print(f"Resume checkpoint not found: {args.resume}")
             return

    trainer.train(start_epoch=start_epoch)

def generate(args):
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu') # Load to CPU first
    
    # Try to recover config from checkpoint, else use default
    if "config" in checkpoint:
        config = AppConfig(**checkpoint["config"])
    else:
        config = AppConfig() # Default
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Generator
    gen = Generator(
        config.model.noise_dim, 
        config.model.img_channels, 
        config.model.features_g,
        config.model.num_classes,
        config.model.embed_size
    ).to(device)
    gen.load_state_dict(checkpoint["gen"])
    gen.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    
    # Generate
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        noise = torch.randn(args.num_images, config.model.noise_dim).to(device)
        
        if args.digit is not None:
            # Generate specific digit
            labels = torch.full((args.num_images,), args.digit, dtype=torch.long).to(device)
            filename = f"generated_digit_{args.digit}.png"
        else:
            # Generate random digits
            labels = torch.randint(0, config.model.num_classes, (args.num_images,)).to(device)
            filename = "generated_grid.png"
            
        fake = gen(noise, labels)
        fake = fake * 0.5 + 0.5 # Denormalize
        
        path = os.path.join(args.output_dir, filename)
        vutils.save_image(fake, path, nrow=8)
        print(f"Saved to {path}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="SyntheticDigits CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the GAN")
    train_parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs")
    train_parser.add_argument("--output_dir", type=str, help="Override output directory")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    
    # Generate Command
    gen_parser = subparsers.add_parser("generate", help="Generate images")
    gen_parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    gen_parser.add_argument("--num_images", type=int, default=64, help="Number of images to generate")
    gen_parser.add_argument("--output_dir", type=str, default="results/inference", help="Output directory")
    gen_parser.add_argument("--digit", type=int, help="Specific digit (0-9) to generate")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        parser.print_help()
