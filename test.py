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
        B_val=200,
        R_train=5.0,
        R_val=5.0,
        max_steps=500,
        checkpoint_steps=[],
        learning_rate=0.01,
        use_cuda=False,
        use_wandb=False,
        save_checkpoints=False,
        save_results=False,
        experiment_name='test'
    )

# Dataset Tests
class TestGaussianMixtureDataset:
    def test_dataset_shapes(self, base_config, device):
        """Test if dataset generates correct shapes"""
        dataset = GaussianMixtureDataset(
            base_config.d, 
            base_config.N, 
            base_config.B, 
            base_config.R_train, 
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
            base_config.R_train, 
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
            base_config.R_train, 
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
        assert 1.5 * base_config.R_train < avg_mean_diff < 2.5 * base_config.R_train, \
            f"Signal strength {avg_mean_diff} too far from target {2*base_config.R_train}"

    def test_validation_reproducibility(self, base_config, device):
        """Test if validation dataset is reproducible"""
        dataset1 = GaussianMixtureDataset(
            base_config.d, base_config.N, base_config.B, base_config.R_train, 
            is_validation=True
        )
        dataset2 = GaussianMixtureDataset(
            base_config.d, base_config.N, base_config.B, base_config.R_train, 
            is_validation=True
        )
        
        # Get data from both datasets
        data1 = dataset1[0]
        data2 = dataset2[0]
        
        # Check all tensors are equal
        for t1, t2 in zip(data1, data2):
            assert torch.allclose(t1, t2), "Validation datasets not reproducible"
    
    @pytest.mark.parametrize("label_flip_p", [0.0, 0.1, 0.3])
    def test_label_flip_probability(self, base_config, device, label_flip_p):
        """Test if labels are flipped with correct probability"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        dataset = GaussianMixtureDataset(
            base_config.d, 
            base_config.N, 
            base_config.B, 
            base_config.R_train, 
            label_flip_p=label_flip_p
        )
        
        _, context_y, _, target_y = dataset[0]
        
        # Generate clean dataset for comparison
        clean_dataset = GaussianMixtureDataset(
            base_config.d, 
            base_config.N, 
            base_config.B, 
            base_config.R_train, 
            label_flip_p=0.0
        )
        
        _, clean_y, _, clean_target_y = clean_dataset[0]
        
        # Calculate empirical flip rate
        context_flips = (context_y != clean_y).float().mean().item()
        target_flips = (target_y != clean_target_y).float().mean().item()
        
        # Allow for some statistical variation
        tolerance = 0.05 if label_flip_p > 0 else 0.001
        
        print(f"\nLabel flip test (p={label_flip_p}):")
        print(f"Context flip rate: {context_flips:.3f}")
        print(f"Target flip rate: {target_flips:.3f}")
        
        assert abs(context_flips - label_flip_p) < tolerance, \
            f"Context flip rate {context_flips:.3f} too far from target {label_flip_p}"
        assert abs(target_flips - label_flip_p) < tolerance, \
            f"Target flip rate {target_flips:.3f} too far from target {label_flip_p}"

def test_identity_memorization():
    """Test that W=I gives perfect memorization in high-d low-N low-R regime"""
    # Setup parameters for high-d, low-N regime
    d = 1000  # High dimension
    N = 3     # Small number of examples
    B = 1000  # Large batch size
    R_val = 1.0   # Low signal-to-noise ratio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = GaussianMixtureDataset(d, N, B, R_val)
    context_x, context_y, _, _ = [t.to(device) for t in dataset[0]]
    
    # Create model and set W to identity
    model = LinearTransformer(d).to(device)
    with torch.no_grad():
        model.W.data = torch.eye(d, device=device)
    
    # Get predictions
    preds = model.compute_in_context_preds(context_x, context_y)
    accuracy = (preds == context_y).float().mean().item()
    
    print("\nIdentity Matrix Memorization Test:")
    print(f"Accuracy: {accuracy:.2%}")
    
    assert accuracy > 0.95, f"Identity matrix should give near-perfect memorization, got {accuracy:.2%}"


# Model Tests
class TestLinearTransformer:
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
    @pytest.mark.parametrize("d,R", [
        (5, 10),
        (15, 12)
    ])
    def test_easy_learning(self, base_config, d, R):
        """Test model can learn an easy task"""
        # Increase SNR for easier learning
        config = base_config
        config.d = d 
        config.R_train = R 
        config.R_val = R
        
        trainer = Trainer(config)
        trainer.train()
        
        final_train_acc = trainer.metrics['train_acc'][-1]
        final_val_acc = trainer.metrics['val_acc'][-1]

        print(f"\nResults for d={d}, R={R}:")
        print(f"Train accuracy: {final_train_acc:.2%}")
        print(f"Val accuracy: {final_val_acc:.2%}")
        
        assert final_val_acc > 0.65, \
            f"Validation accuracy too low for d={d}, R={R}: {final_val_acc:.2%}"


    def test_snr_sensitivity(self, base_config):
        """Test that higher SNR leads to better performance"""
        results = {}
        
        # Test with different SNRs
        for R in [2.0, 20.0]:
            config = base_config
            config.R_train = R
            config.R_val = R
            config.d = 40
            config.B = 40
            config.max_steps = 1000
            
            trainer = Trainer(config)
            trainer.train()
            
            results[R] = trainer.metrics['val_acc'][-1]
        
        print(f"Accuracies - Low SNR (R=2): {results[2.0]:.2%}, High SNR (R=20): {results[20.0]:.2%}")
        
        # Check that higher SNR gives better accuracy
        assert results[20.0] > results[2.0] + 0.02, \
            f"Higher SNR should give notably better accuracy. Got R=20: {results[20.0]:.2%}, R=2: {results[2.0]:.2%}"

    def test_context_size_effect(self, base_config):
        """Test that larger context size leads to better performance"""
        results = {}
        
        # Test with different context sizes
        for N in [3, 80]:
            config = base_config
            config.N = N
            config.R_train = config.d**0.25  # Small SNR to make task challenging
            config.R_val = config.R_train
            config.max_steps = 1000
            
            trainer = Trainer(config)
            trainer.train()
            
            results[N] = trainer.metrics['val_acc'][-1]
        
        print(f"Accuracies - Small context (N=3): {results[3]:.2%}, Large context (N=80): {results[80]:.2%}")
        
        # Check that larger context gives better accuracy
        assert results[80] > results[3] + 0.05, \
            f"Larger context should give better accuracy. Got N=80: {results[80]:.2%}, N=3: {results[3]:.2%}"

    @pytest.mark.parametrize("label_flip_p", [0.0, 0.4])
    def test_in_context_accuracy(self, base_config, label_flip_p):
        """Test in-context accuracy computation"""
        # Low SNR, high-D, low-N should have high in-context acc, regardless of label flipping
        d = 1000
        config = base_config
        config.R_train = 10 * d**0.5
        config.R_val = d**0.3
        config.d = d
        config.B = d
        config.B_val = 200
        config.N = 5
        config.label_flip_p = label_flip_p
        
        trainer = Trainer(config)
        trainer.train()
        
        final_in_context_acc = trainer.metrics['in_context_acc'][-1]
        
        print(f"label_flip_p={label_flip_p}, In-context train acc: {final_in_context_acc:.2%}")
        
        assert final_in_context_acc > 0.95, \
            f"In-context accuracy too low: {final_in_context_acc:.2%}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])