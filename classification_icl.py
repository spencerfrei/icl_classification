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

class OldGaussianMixtureDataset(Dataset):
    """Dataset class for generating Gaussian mixture data"""
    def __init__(self, d: int, N: int, B: int, R: float, device: torch.device):
        self.d = d
        self.N = N 
        self.B = B
        self.R = R
        self.device = device
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.B):
            # Generate mean vector uniformly from sphere of radius R
            # Shape: (d,)
            mu = torch.randn(self.d, device=self.device)
            mu = mu / torch.norm(mu) * self.R
            
            # Generate N+1 examples (N for context, 1 for target)
            # Labels will be +/- 1
            # Shape: (N+1,)
            y_01 = (torch.rand(self.N + 1, device=self.device) > 0.5).float() 
            y = 2 * y_01 - 1
            
            # Generate Gaussian noise
            # Shape: (N+1, d)
            z = torch.randn(self.N + 1, self.d, device=self.device)
            
            # Broadcast y[:, None] from (N+1, 1) against mu (d,) -> (1, d) to get (N+1, d)
            # Then add noise z of shape (N+1, d)
            x = y[:, None] * mu[None, :] + z
            
            # Split into context and target:
            # x[:self.N] -> context inputs: (N, d)
            # y[:self.N] -> context labels: (N,)
            # x[self.N] -> target input: (d,)
            # y[self.N] -> target label: scalar
            data.append((x[:self.N], y_01[:self.N], x[self.N], y_01[self.N]))
        return data
    
    def __len__(self) -> int:
        return self.B
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]

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

def test_gaussian_mixture_dataset():
    """Test properties of the generated dataset"""
    print("\nTesting dataset generation...")
    
    # Setup small test configuration
    d, N, B = 10, 100, 100
    R = 5.0
    device = torch.device("cpu")
    
    dataset = GaussianMixtureDataset(d, N, B, R, device)
    
    # Get the full batch
    context_x, context_y, target_x, target_y = dataset[0]
    
    # Test shapes
    assert context_x.shape == (B, N, d), f"Expected context_x shape (B,N,d), got {context_x.shape}"
    assert context_y.shape == (B, N), f"Expected context_y shape (B,N), got {context_y.shape}"
    assert target_x.shape == (B, d), f"Expected target_x shape (B,d), got {target_x.shape}"
    assert target_y.shape == (B,), f"Expected target_y shape (B,), got {target_y.shape}"
    
    # Test label distribution (should be roughly balanced between 0s and 1s)
    all_labels = torch.cat([context_y.flatten(), target_y])
    label_mean = all_labels.mean().item()
    assert 0.35 < label_mean < 0.65, f"Labels should be roughly balanced, got mean {label_mean}"
    
    # Test signal strength (distance between class means) on the context set
    mean_diffs = []
    for i in range(min(10, B)):  # Test first 10 tasks
        task_x = context_x[i]  # (N, d)
        task_y = context_y[i]  # (N,)
        pos_mask = task_y == 1
        neg_mask = task_y == 0
        
        if pos_mask.any() and neg_mask.any():
            pos_mean = task_x[pos_mask].mean(dim=0)  # (d,)
            neg_mean = task_x[neg_mask].mean(dim=0)  # (d,)
            mean_diff = torch.norm(pos_mean - neg_mean).item()
            mean_diffs.append(mean_diff)
    
    if mean_diffs:
        avg_mean_diff = np.mean(mean_diffs)
        print(f"Average mean difference: {avg_mean_diff:.2f} (Target: {2*R})")
        
        # Due to noise, we expect the empirical mean difference to be roughly 2*R
        # but with some variance. Using wider bounds for the test.
        assert 1.5 * R < avg_mean_diff < 2.5 * R, \
            f"Average mean difference {avg_mean_diff} deviates too much from R={R}"
    
    # Test data loader behavior
    loader = DataLoader(dataset, batch_size=B, shuffle=False)
    num_batches = 0
    for batch in loader:
        num_batches += 1
        # Verify we get the same data each time
        assert torch.allclose(batch[0], context_x), "DataLoader should return same data each time"
    
    assert num_batches == 1, "Should only get one batch for full-batch GD"
    
    print("Dataset tests passed!")

def test_easy_learning():
    """Test that model can learn an easy task with high SNR"""
    print("\nTesting learning on easy task...")
    
    # Setup an easy learning problem
    d, N, B = 10, 80, 200
    B_val = 50  # Number of validation tasks
    R = 15.0  # High SNR
    device = torch.device("cpu")
    
    config = ExperimentConfig(
        d=d,
        N=N,
        B=B,
        B_val=B_val,
        R=R,
        max_steps=200,
        checkpoint_steps=[],
        learning_rate=0.01,
        use_cuda=False
    )
    
    trainer = Trainer(config)
    
    # Debug initial data
    test_batch = next(iter(trainer.train_loader))
    context_x, context_y, target_x, target_y = test_batch
    print("\nInitial data shapes and values:")
    print(f"context_x shape: {context_x.shape}, range: [{context_x.min():.2f}, {context_x.max():.2f}]")
    print(f"context_y shape: {context_y.shape}, values: {torch.unique(context_y).tolist()}")
    print(f"target_x shape: {target_x.shape}, range: [{target_x.min():.2f}, {target_x.max():.2f}]")
    print(f"target_y shape: {target_y.shape}, values: {torch.unique(target_y).tolist()}")
    
    trainer.train()
    
    # Check both train and validation accuracy
    final_train_acc = trainer.metrics['train_acc'][-1]
    final_val_acc = trainer.metrics['val_acc'][-1]
    
    print(f"\nAccuracy progression:")
    print(f"Train: Start={trainer.metrics['train_acc'][0]:.2f}, "
          f"Mid={trainer.metrics['train_acc'][len(trainer.metrics['train_acc'])//2]:.2f}, "
          f"Final={final_train_acc:.2f}")
    print(f"Val: Start={trainer.metrics['val_acc'][0]:.2f}, "
          f"Mid={trainer.metrics['val_acc'][len(trainer.metrics['val_acc'])//2]:.2f}, "
          f"Final={final_val_acc:.2f}")
    
    # Assert both train and validation accuracy meet threshold
    assert final_train_acc > 0.65, \
        f"Model should achieve >65% training accuracy, got {final_train_acc:.2%}"
    assert final_val_acc > 0.65, \
        f"Model should achieve >65% validation accuracy, got {final_val_acc:.2%}"
    
    # Check for overfitting
    assert abs(final_train_acc - final_val_acc) < 0.1, \
        f"Train-val accuracy gap too large: {abs(final_train_acc - final_val_acc):.2%}"
    
    print("Easy learning test passed!")


def old_test_easy_learning():
    """Test that model can learn an easy task with high SNR"""
    print("\nTesting learning on easy task...")


    # Setup an easy learning problem
    d, N, B = 10, 80, 200
    R = 15.0  # High SNR
    device = torch.device("cpu")
    
    config = ExperimentConfig(
        d=d,
        N=N,
        B=B,
        R=R,
        max_steps=200,
        checkpoint_steps=[],
        learning_rate=0.01,
        use_cuda=False
    )
    
    trainer = Trainer(config)
    
    # Debug initial data
    test_batch = next(iter(trainer.train_loader))
    context_x, context_y, target_x, target_y = test_batch
    print("\nInitial data shapes and values:")
    print(f"context_x shape: {context_x.shape}, range: [{context_x.min():.2f}, {context_x.max():.2f}]")
    print(f"context_y shape: {context_y.shape}, values: {torch.unique(context_y).tolist()}")
    print(f"target_x shape: {target_x.shape}, range: [{target_x.min():.2f}, {target_x.max():.2f}]")
    print(f"target_y shape: {target_y.shape}, values: {torch.unique(target_y).tolist()}")
    
    trainer.train()    
    
    # Check learning progression
    accs = trainer.metrics['val_acc']
    final_acc = accs[-1]
    print(f"Accuracy progression: Start={accs[0]:.2f}, Mid={accs[len(accs)//2]:.2f}, Final={final_acc:.2f}")
    
    # With high SNR and sufficient context, should achieve good accuracy
    assert final_acc > 0.65, \
        f"Model should achieve >65% accuracy on easy task, got {final_acc:.2%}"
    
    # Should show improvement during training
    assert final_acc > accs[0] + 0.1, \
        f"Model should improve significantly during training: {accs[0]:.2f} -> {final_acc:.2f}"
    
    print(f"Easy learning test passed! Final accuracy: {final_acc:.2%}")
        

def test_model_basic_properties():
    """Test basic properties of the model"""
    print("\nTesting model properties...")
    
    # Use same dimensions as in easy learning test
    d, N, B = 10, 80, 200  # Match easy learning test dimensions
    model = LinearTransformer(d)
    
    # Test weight initialization
    w_std = torch.std(model.W).item()
    print(f"Weight std: {w_std:.4f}")
    assert 0.05 < w_std < 0.2, \
        f"Weight initialization scale should be close to 0.1, got {w_std}"
    
    # Test shapes and basic forward pass
    context_x = torch.randn(B, N, d)
    context_y = (torch.rand(B, N) > 0.5).float()  # Binary 0/1 labels
    target_x = torch.randn(B, d)
    
    # Test forward pass
    with torch.no_grad():
        # Test intermediate values
        context_y_signal = 2 * context_y - 1
        assert torch.all((context_y_signal == 1) | (context_y_signal == -1)), \
            "Signal values should be exactly -1 or 1"
            
        output = model(context_x, context_y, target_x)
        assert output.shape == (B,), \
            f"Expected output shape ({B},), got {output.shape}"
        
        # Test output magnitude
        output_std = torch.std(output).item()
        print(f"Output std: {output_std:.4f}")
        assert output_std < 10.0, \
            f"Output seems to have unusually large magnitude: std={output_std}"
        
        # Test intermediate shapes
        N_test = context_x.shape[1]
        context_y_test = 2 * context_y - 1
        weighted = context_y_test[..., None] * context_x  # (B, N, d)
        assert weighted.shape == (B, N, d), \
            f"Weighted shape wrong, got {weighted.shape}"
        context_term = (1/N_test) * weighted.sum(dim=1)  # (B, d)
        assert context_term.shape == (B, d), \
            f"Context term shape wrong, got {context_term.shape}"
        transformed = context_term @ model.W  # (B, d)
        assert transformed.shape == (B, d), \
            f"Transformed shape wrong, got {transformed.shape}"
    
    print("Model property tests passed!")



def test_snr_sensitivity():
    """Test model's sensitivity to signal-to-noise ratio"""
    print("\nTesting SNR sensitivity...")
    
    d, N, B = 10, 40, 200  # Increased for stability
    device = torch.device("cpu")
    max_steps = 1000
    
    results = {}
    
    # Test with different SNRs
    for R in [2.0, 20.0]:  # Bigger gap between SNRs
        config = ExperimentConfig(
            d=d,
            N=N,
            B=B,
            B_val=500,
            R=R,
            max_steps=max_steps,
            checkpoint_steps=[],
            use_cuda=False,
            wandb_project=None
        )
        
        trainer = Trainer(config)
        trainer.train()
        
        results[R] = trainer.metrics['val_acc'][-1]
    
    print(f"Accuracies - Low SNR (R=2): {results[2.0]:.2%}, High SNR (R=20): {results[20.0]:.2%}")
    
    # Higher SNR should give substantially better accuracy
    assert (results[20.0] > 0.95 and results[2.0] > 0.95) or results[20.0] > results[2.0] + 0.05, \
        f"Higher SNR should give notably better accuracy. Got R=20: {results[20.0]:.2%}, R=2: {results[2.0]:.2%}"
    
    print("SNR sensitivity test passed!")

def test_context_size_effect():
    """Test effect of context size on learning"""
    print("\nTesting context size effect...")
    
    d, B = 20, 200
    R = d**0.25  # small SNR
    device = torch.device("cpu")
    max_steps = 1000
    
    results = {}
    
    # Test with different context sizes
    N_vals = [3, 80]
    for N in N_vals:  # Bigger gap in context sizes
        config = ExperimentConfig(
            d=d,
            N=N,
            B=B,
            B_val=500,
            R=R,
            max_steps=max_steps,
            checkpoint_steps=[],
            use_cuda=False,
            wandb_project=None
        )
        
        print(f'Training with: N={N}')
        trainer = Trainer(config)
        trainer.train()
        
        results[N] = trainer.metrics['val_acc'][-1]
    
    print(f"Accuracies - Small context (N={N_vals[0]}): {results[N_vals[0]]:.2%}, Large context (N={N_vals[1]}): {results[N_vals[1]]:.2%}")
    
    # Larger context should give better accuracy
    assert results[N_vals[1]] > results[N_vals[0]] + 0.05, \
        f"Larger context should give better accuracy. Got N={N_vals[1]}: {results[N_vals[1]]:.2%}, N={N_vals[0]}: {results[N_vals[0]]:.2%}"
    
    print("Context size test passed!")

def run_all_tests():
    """Run all unit tests"""
    print("Running all tests...")
    test_gaussian_mixture_dataset()
    test_model_basic_properties()
    test_easy_learning()
    test_snr_sensitivity()
    test_context_size_effect()
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Add this to the existing main block
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_all_tests()
    else:
        # Original training code...
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