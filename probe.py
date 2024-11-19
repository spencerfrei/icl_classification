import torch
import numpy as np
from classification_icl import LinearTransformer, Trainer, ExperimentConfig

def analyze_matrix(W, name=""):
    """Analyze key properties of a matrix"""
    # Convert to numpy for easier analysis
    if torch.is_tensor(W):
        W = W.detach().cpu().numpy()
    
    # Compute SVD
    U, s, Vh = np.linalg.svd(W)
    
    # Compute properties
    stable_rank = (np.linalg.norm(s)**2) / (s[0]**2)  # ||W||_F^2 / ||W||_2^2
    trace = np.trace(W @ W.T)
    frob_norm = np.linalg.norm(s)  # Frobenius norm
    trace_frob_ratio = np.abs(trace) / (frob_norm**2)
    
    print(f"\n=== Matrix Analysis for {name} ===")
    print(f"Shape: {W.shape}")
    print(f"Stable Rank: {stable_rank:.3f}")
    print(f"Trace/Frob^2 Ratio: {trace_frob_ratio:.3f}")
    print(f"Top 5 singular values: {s[:5]}")
    print(f"Bottom 5 singular values: {s[-5:]}")
    print(f"Frobenius norm: {frob_norm:.3f}")
    print(f"Trace: {trace:.3f}")
    
    return {
        "stable_rank": stable_rank,
        "trace_frob_ratio": trace_frob_ratio,
        "singular_values": s,
        "frob_norm": frob_norm,
        "trace": trace
    }

def compare_to_identity():
    """Train a model and compare its W matrix to identity"""
    # Setup in high-d low-N regime where memorization should occur
    d = 1000
    config = ExperimentConfig(
        d=d,               # High dimension
        N=5,              # Small number of examples
        B=d,           # Large batch size
        B_val=100,       # Validation batch size
        R_train=10*d**0.6,            
        R_val= 10,
        max_steps=300,   # Training steps
        checkpoint_steps=[],
        learning_rate=0.0001,
        use_cuda=False,
        use_wandb=False,
        label_flip_p=0.4
    )
    
    # Train model
    trainer = Trainer(config)
    trainer.train()
    
    # Get trained W matrix
    W_trained = trainer.model.W.data
    
    # Create identity matrix with same scale as initialization
    W_init_scale = 0.1  # From your initialization in LinearTransformer
    W_identity = W_init_scale * torch.eye(d, device=W_trained.device)
    
    # Analyze both matrices
    trained_stats = analyze_matrix(W_trained, "Trained W")
    identity_stats = analyze_matrix(W_identity, "Scaled Identity")
    
    # Print final accuracies
    print("\n=== Final Accuracies ===")
    print(f"Training accuracy: {trainer.metrics['train_acc'][-1]:.3f}")
    print(f"In-context accuracy: {trainer.metrics['in_context_acc'][-1]:.3f}")
    
    return trained_stats, identity_stats

if __name__ == "__main__":
    trained_stats, identity_stats = compare_to_identity()