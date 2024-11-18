import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import pandas as pd
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters"""
    d: int  # Input dimension
    N: int  # Number of examples per task
    B: int  # Number of tasks
    B_val: int # Number of validation tasks
    R: float  # Signal-to-noise ratio
    max_steps: int  # Maximum training steps
    checkpoint_steps: List[int]  # Steps at which to save checkpoints
    learning_rate: float = 1e-2
    use_cuda: bool = True  # Flag for using CUDA
    use_wandb: bool = False # Disable wandb by default
    wandb_project: Optional[str] = "linear-transformer"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    
    def __post_init__(self):
        """Setup device based on CUDA availability"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")

class GaussianMixtureDataset(Dataset):
    """Dataset class for generating Gaussian mixture data"""
    def __init__(self, d: int, N: int, B: int, R: float, device: torch.device, is_validation: bool=False):
        self.d = d
        self.N = N 
        self.B = B
        self.R = R
        self.device = device
        self.is_validation = is_validation

        with torch.random.fork_rng():
            if is_validation:
                # use differents eed for validation data
                torch.manual_seed(42)
            
            # Generate all data at once and store in tensors
            self.context_x, self.context_y, self.target_x, self.target_y = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate all data at once and return fixed tensors.
        
        Returns:
            context_x: Shape (B, N, d)
            context_y: Shape (B, N)
            target_x: Shape (B, d)
            target_y: Shape (B)
        """
        # Generate mean vectors for all tasks - Shape: (B, d)
        mus = torch.randn(self.B, self.d, device=self.device)
        # Normalize and scale - Shape: (B, d)
        mus = mus / torch.norm(mus, dim=1, keepdim=True) * self.R
        
        # Generate all labels at once - Shape: (B, N+1)
        y_all = (torch.rand(self.B, self.N + 1, device=self.device) > 0.5).float()
        
        # Convert to {-1, 1} for signal generation - Shape: (B, N+1)
        y_signal = 2 * y_all - 1
        
        # Generate noise - Shape: (B, N+1, d)
        z = torch.randn(self.B, self.N + 1, self.d, device=self.device)
        
        # Broadcast for multiplication:
        # mus[:, None, :] shape: (B, 1, d)
        # y_signal[..., None] shape: (B, N+1, 1)
        # Result shape: (B, N+1, d)
        x = y_signal[..., None] * mus[:, None, :] + z
        
        # Split into context and target
        context_x = x[:, :self.N, :]  # (B, N, d)
        target_x = x[:, -1, :]        # (B, d)
        context_y = y_all[:, :self.N]  # (B, N)
        target_y = y_all[:, -1]       # (B)
        
        return context_x, context_y, target_x, target_y
    
    def __len__(self) -> int:
        return 1  # Only one batch for full-batch GD
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the entire dataset as one batch"""
        assert idx == 0, "Only one batch supported for full-batch GD"
        return self.context_x, self.context_y, self.target_x, self.target_y

class LinearTransformer(nn.Module):
    """One-layer linear transformer model"""
    def __init__(self, d: int):
        super().__init__()
        # Shape: (d, d)
        # Initialization shouldn't matter
        self.W = nn.Parameter(torch.randn(d, d) * 0.1)
        
    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, 
                target_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_x: Shape (batch_size, N, d)
            context_y: Shape (batch_size, N)
            target_x: Shape (batch_size, d)
        Returns:
            prediction: Shape (batch_size,)
        """
        N = context_x.shape[1]

        
        # Convert 0/1 labels to -1/+1 for the computation
        # Shape: (B, N) -> (B, N)
        context_y_signal = 2 * context_y - 1
        
        # 1. Add feature dimension to context_y_signal: (B, N) -> (B, N, 1)
        # 2. Multiply with context_x: (B, N, 1) * (B, N, d) -> (B, N, d)
        # 3. Sum over context dimension and normalize
        # Shape: (B, d)
        context_term = (1/N) * torch.sum(context_y_signal[..., None] * context_x, dim=1)
        
        # 1. Transform context: (B, d) @ (d, d) -> (B, d)
        # 2. Take inner product with target: (B, d) * (B, d) -> (B,)
        transformed = context_term @ self.W
        logits = (transformed * target_x).sum(dim=1)
        
        return logits

class Trainer:
    """Trainer class for linear transformer"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()
        
        # Initialize model and optimizer
        self.model = LinearTransformer(config.d).to(config.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=config.learning_rate,
            momentum=0.0)
        
        # Initialize datasets
        self.train_dataset = GaussianMixtureDataset(
            config.d, config.N, config.B, config.R, config.device, is_validation=False
        )
        self.val_dataset = GaussianMixtureDataset(
            config.d, config.N, config.B, config.R, config.device, is_validation=True
        )

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=None, 
            shuffle=True,
            pin_memory=True if config.device.type == "cuda" else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=None, 
            shuffle=False,
            pin_memory=True if config.device.type == "cuda" else False
        )

        # Initialize logging
        self.setup_wandb()
        self.metrics = {
            'step': [], 
            'train_loss': [], 
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'batch_time': [],
            'samples_per_second': []
        }
        
        print(f"Using device: {config.device}")
    
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                context_x, context_y, target_x, target_y = batch
                
                # Forward pass
                pred = self.model(context_x, context_y, target_x)
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, target_y.float())
                
                # Compute accuracy
                val_acc = ((pred > 0).float() == target_y).float().mean()
                
                return val_loss.item(), val_acc.item()
        
    def setup_directories(self):
        """Create directories for checkpoints and results"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        
    def setup_wandb(self):
        """Initialize W&B logging"""
        if not self.config.use_wandb or self.config.wandb_project is None:
            return 

        wandb.init(
            project=self.config.wandb_project,
            config={
                'd': self.config.d,
                'N': self.config.N,
                'B': self.config.B,
                'R': self.config.R,
                'max_steps': self.config.max_steps,
                'batch_size': self.config.B,
                'learning_rate': self.config.learning_rate,
                'device': str(self.config.device)
            }
        )
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_step_{step}.pt')
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, step: int):
        """Load model checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_step_{step}.pt')
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
        return checkpoint['step']
        
    def save_metrics(self):
        """Save metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        path = os.path.join(self.config.results_dir, 'metrics.csv')
        df.to_csv(path, index=False)
        
    def train(self):
        """Training loop with timing measurements"""
        self.model.train()
        step = 0
        total_start_time = time.time()
        num_samples = 0
        
        print(f"\nStarting training on {self.config.device}")
        print(f"Input dimension (d): {self.config.d}")
        print(f"Training tasks: {self.config.B}")
        print(f"Validation tasks: {self.config.B_val}")
        print(f"Batch size: {self.config.B}")
        print("-" * 50)
        
        while step < self.config.max_steps:
            for batch in self.train_loader:
                batch_start_time = time.time()
                # Shapes after batching:
                # context_x: (batch_size, N, d)
                # context_y: (batch_size, N)
                # target_x: (batch_size, d)
                # target_y: (batch_size,)
                context_x, context_y, target_x, target_y = batch
                
                # Forward pass - pred shape: (batch_size,)
                pred = self.model(context_x, context_y, target_x)
                train_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, target_y.float())
                
                # Backward pass
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                
                # Compute train accuracy
                train_acc = ((pred > 0).float() == target_y).float().mean()

                # Compute validation metrics
                val_loss, val_acc = self.evaluate()
                
                # Compute timing metrics
                batch_time = time.time() - batch_start_time
                num_samples += len(context_x)
                avg_samples_per_second = num_samples / (time.time() - total_start_time)
                
                # Log metrics
                self.metrics['step'].append(step)
                self.metrics['train_loss'].append(train_loss.item())
                self.metrics['train_acc'].append(train_acc.item())
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_acc'].append(val_acc)
                self.metrics['batch_time'].append(batch_time)
                self.metrics['samples_per_second'].append(avg_samples_per_second)
                
                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': train_loss.item(),
                        'train_acc': train_acc.item(),
                        'val_loss': val_loss.item(),
                        'val_acc': val_acc.item(),
                        'batch_time': batch_time,
                        'samples_per_second': avg_samples_per_second,
                        'step': step
                    })
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    print(f"Step {step}/{self.config.max_steps} | "
                          f"Train Loss: {train_loss.item():.4f} | "
                          f"Train Acc: {train_acc.item():.4f} | "
                          f"Val Loss: {train_loss.item():.4f} | "
                          f"Val Acc: {train_acc.item():.4f} | "
                          f"Batch time: {batch_time*1000:.2f}ms | "
                          f"Samples/sec: {avg_samples_per_second:.2f}")
                
                # Save checkpoint if needed
                if step in self.config.checkpoint_steps:
                    self.save_checkpoint(step)
                    print(f"Saved checkpoint at step {step}")
                    
                step += 1
                if step >= self.config.max_steps:
                    break
        
        # Print final timing statistics
        total_time = time.time() - total_start_time
        print("\nTraining completed!")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Final train accuracy: {self.metrics['train_acc'][-1]:.4f}")
        print(f"Final validation accuracy: {self.metrics['val_acc'][-1]:.4f}") 
        print(f"Average samples/second: {num_samples/total_time:.2f}")
        print(f"Average batch time: {np.mean(self.metrics['batch_time'])*1000:.2f}ms")
        
        # Save final metrics and checkpoint
        self.save_metrics()
        self.save_checkpoint(step)
        if self.config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    dimensions = [50, 500, 5000]
    for d in dimensions:
        B_values = [int(d**0.2), int(d**0.6), d, int(d**1.4)]
        R_values = [5 * d**0.25, 5 * np.sqrt(d)]
        for B in B_values:
            for R in R_values:
                print(f"\nStarting experiment with d={d}, B={B}, R={R}")
                
                config = ExperimentConfig(
                    d=d,
                    N=40,
                    B=B,
                    R=R,
                    max_steps=12800,
                    checkpoint_steps=[100, 400, 1600, 6400, 12800],
                    use_cuda=True  # Set to False to force CPU usage
                )
                
                trainer = Trainer(config)
                trainer.train()