import pytest
import torch
import numpy as np
from classification_icl import (
    ExperimentConfig,
    GaussianMixtureDataset,
    LinearTransformer,
    Trainer
)

# Fixtures for common test setup
@pytest.fixture
def device():
    """Force CPU device for testing"""
    return torch.device("cpu")

@pytest.fixture
def base_config():
    """Base configuration for tests"""
    return ExperimentConfig(
        d=10,
        N=80,
        B=200,
        B_val=50,
        R=5.0,
        max_steps=200,
        checkpoint_steps=[],
        learning_rate=0.01,
        use_cuda=False,
        use_wandb=False
    )

# Dataset Tests
class TestGaussianMixtureDataset:
    def test_dataset_shapes(self, base_config, device):
        """Test if dataset generates correct shapes"""
        dataset = GaussianMixtureDataset(
            base_config.d, 
            base_config.N, 
            base_config.B, 
            base_config.R, 
            device
        )
        
        context_x, context_y, target_x, target_y = dataset[0]
        
        assert context_x.shape == (base_config.B, base_config.N, base_config.d), "Wrong context_x shape"
        assert context_y.shape == (base_config.B, base_config.N), "Wrong context_y shape"
        assert target_x.shape == (base_config.B, base_config.d), "Wrong target_x shape"
        assert target_y.shape == (base_config.B,), "Wrong target_y shape"

    def test_label_distribution(self, base_config, device):
        """Test if labels are roughly balanced"""
        dataset = GaussianMixtureDataset(
            base_config.d, 
            base_config.N, 
            base_config.B, 
            base_config.R, 
            device
        )
        
        context_x, context_y, target_x, target_y = dataset[0]
        all_labels = torch.cat([context_y.flatten(), target_y])
        label_mean = all_labels.mean().item()
        
        assert 0.35 < label_mean < 0.65, f"Labels not balanced, got mean {label_mean}"

    def test_signal_strength(self, base_config, device):
        """Test if signal strength matches R parameter"""
        dataset = GaussianMixtureDataset(
            base_config.d, 
            base_config.N, 
            base_config.B, 
            base_config.R, 
            device
        )
        
        context_x, context_y, _, _ = dataset[0]
        
        # Test first few tasks
        mean_diffs = []
        for i in range(min(10, base_config.B)):
            task_x = context_x[i]
            task_y = context_y[i]
            pos_mask = task_y == 1
            neg_mask = task_y == 0
            
            if pos_mask.any() and neg_mask.any():
                pos_mean = task_x[pos_mask].mean(dim=0)
                neg_mean = task_x[neg_mask].mean(dim=0)
                mean_diff = torch.norm(pos_mean - neg_mean).item()
                mean_diffs.append(mean_diff)
        
        avg_mean_diff = np.mean(mean_diffs)
        assert 1.5 * base_config.R < avg_mean_diff < 2.5 * base_config.R, \
            f"Signal strength {avg_mean_diff} too far from target {2*base_config.R}"

    def test_validation_reproducibility(self, base_config, device):
        """Test if validation dataset is reproducible"""
        dataset1 = GaussianMixtureDataset(
            base_config.d, base_config.N, base_config.B, base_config.R, 
            device, is_validation=True
        )
        dataset2 = GaussianMixtureDataset(
            base_config.d, base_config.N, base_config.B, base_config.R, 
            device, is_validation=True
        )
        
        # Get data from both datasets
        data1 = dataset1[0]
        data2 = dataset2[0]
        
        # Check all tensors are equal
        for t1, t2 in zip(data1, data2):
            assert torch.allclose(t1, t2), "Validation datasets not reproducible"

# Model Tests
class TestLinearTransformer:
    def test_initialization(self, base_config):
        """Test model initialization"""
        model = LinearTransformer(base_config.d)
        w_std = torch.std(model.W).item()
        assert 0.05 < w_std < 0.2, f"Weight initialization scale wrong: {w_std}"

    def test_forward_shapes(self, base_config, device):
        """Test forward pass shapes"""
        model = LinearTransformer(base_config.d)
        B, N, d = base_config.B, base_config.N, base_config.d
        
        context_x = torch.randn(B, N, d, device=device)
        context_y = torch.randint(0, 2, (B, N), device=device).float()
        target_x = torch.randn(B, d, device=device)
        
        output = model(context_x, context_y, target_x)
        assert output.shape == (B,), f"Wrong output shape: {output.shape}"
        
        # Test output magnitude
        output_std = torch.std(output).item()
        assert output_std < 10.0, f"Output magnitude too large: std={output_std}"

# Training Tests
class TestTraining:
    def test_easy_learning(self, base_config):
        """Test model can learn an easy task"""
        # Increase SNR for easier learning
        config = base_config
        config.R = 15.0
        
        trainer = Trainer(config)
        trainer.train()
        
        final_train_acc = trainer.metrics['train_acc'][-1]
        final_val_acc = trainer.metrics['val_acc'][-1]
        
        assert final_train_acc > 0.65, f"Train accuracy too low: {final_train_acc:.2%}"
        assert final_val_acc > 0.65, f"Validation accuracy too low: {final_val_acc:.2%}"
        assert abs(final_train_acc - final_val_acc) < 0.1, \
            f"Train-val gap too large: {abs(final_train_acc - final_val_acc):.2%}"

    @pytest.mark.parametrize("R", [2.0, 20.0])
    def test_snr_sensitivity(self, base_config, R):
        """Test model's sensitivity to signal-to-noise ratio"""
        config = base_config
        config.R = R
        config.max_steps = 1000  # More steps for convergence
        
        trainer = Trainer(config)
        trainer.train()
        
        final_acc = trainer.metrics['val_acc'][-1]
        print(f"Final accuracy with R={R}: {final_acc:.2%}")
        assert final_acc > 0.5, f"Failed to learn with R={R}, acc={final_acc:.2%}"

    @pytest.mark.parametrize("N", [3, 80])
    def test_context_size_effect(self, base_config, N):
        """Test effect of context size on learning"""
        config = base_config
        config.N = N
        config.R = config.d**0.25  # Small SNR to make task challenging
        config.max_steps = 1000
        
        trainer = Trainer(config)
        trainer.train()
        
        final_acc = trainer.metrics['val_acc'][-1]
        print(f"Final accuracy with N={N}: {final_acc:.2%}")
        assert final_acc > 0.5, f"Failed to learn with N={N}, acc={final_acc:.2%}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])