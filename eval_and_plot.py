from classification_icl import ExperimentConfig, LinearTransformer, GaussianMixtureDataset
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import List, Dict, Tuple, Optional

class CheckpointEvaluator:
    """Evaluator class for analyzing trained model checkpoints"""
    
    def __init__(self, checkpoint_dir: str, label_flips: Optional[List[float]] = [0.0, 0.2]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.label_flips = label_flips
        
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[LinearTransformer, ExperimentConfig]:
        """Load model and config from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = ExperimentConfig(**asdict(checkpoint['config']))
        model = LinearTransformer(config.d)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, config

    def evaluate_risk_curves(
        self,
        model: LinearTransformer,
        d: int,
        max_seq_length: int,
        R: Optional[float] = None,
        num_samples: int = 2500,
        label_flip_ps: Optional[List[float]] = None,
        device: str = 'cpu'
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Evaluate model's performance curves including both means and standard errors.
        """
        if label_flip_ps is None:
            label_flip_ps = self.label_flips
            
        model = model.to(device)
        results = {}
        if R is None:
            R = d ** 0.3
        
        for label_flip_p in label_flip_ps:
            print(f"\nEvaluating curves for d={d}, label_flip_p={label_flip_p}")
            
            dataset = GaussianMixtureDataset(
                d=d,
                N=max_seq_length,
                B=num_samples,
                R=R,
                is_validation=True,
                label_flip_p=label_flip_p
            )
            
            context_x, context_y, _, _ = [t.to(device) for t in dataset[0]]
            
            memorization_accuracies = np.zeros((max_seq_length-1, num_samples))
            test_accuracies = np.zeros((max_seq_length-1, num_samples))
            
            with torch.no_grad():
                for k in range(1, max_seq_length):
                    curr_context_x = context_x[:, :k]
                    curr_context_y = context_y[:, :k]
                    
                    # Memorization accuracy (per example)
                    mem_preds = model.compute_in_context_preds(curr_context_x, curr_context_y)
                    mem_correct = (mem_preds == curr_context_y).float()
                    memorization_accuracies[k-1] = mem_correct.mean(dim=1).cpu().numpy()
                    
                    # Test accuracy (per example)
                    next_x = context_x[:, k:k+1, :]
                    next_y = context_y[:, k:k+1]
                    
                    pred_logits = model(curr_context_x, curr_context_y, next_x.squeeze(1))
                    test_preds = (pred_logits > 0).float()
                    test_correct = (test_preds == next_y.squeeze(1)).float()
                    test_accuracies[k-1] = test_correct.cpu().numpy()
                    
                    if k % 20 == 0:
                        print(f"Position {k}: memorization = {memorization_accuracies[k-1].mean():.3f}, test = {test_accuracies[k-1].mean():.3f}")
            
            results[label_flip_p] = {
                'memorization': {
                    'mean': memorization_accuracies.mean(axis=1),
                    'stderr': memorization_accuracies.std(axis=1) / np.sqrt(num_samples-1)
                },
                'test': {
                    'mean': test_accuracies.mean(axis=1),
                    'stderr': test_accuracies.std(axis=1) / np.sqrt(num_samples-1)
                }
            }
            
        return results

    def plot_dimension_curves(self, results_by_d: Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]],  R_d_to_power: float = 0.3,
                            sequence_length: int = 40, save_path: Optional[str] = None,
                            label_flip_ps: Optional[List[float]] = None, force_y_range = False):
        """Plot separate accuracy vs dimension curves for each label flip probability"""
        if label_flip_ps is None:
            label_flip_ps = self.label_flips

        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        dimensions = sorted(results_by_d.keys())
        seq_idx = sequence_length - 2
        
        mem_color = 'red'
        test_color = 'blue'
        opt_color = 'green'
        
        # Create separate plot for each label flip probability
        for label_flip_p in label_flip_ps:
            plt.figure(figsize=(8, 6))
            
            mem_means = []
            mem_errs = []
            test_means = []
            test_errs = []
            
            for d in dimensions:
                curves = results_by_d[d][label_flip_p]
                mem_means.append(curves['memorization']['mean'][seq_idx])
                mem_errs.append(curves['memorization']['stderr'][seq_idx])
                test_means.append(curves['test']['mean'][seq_idx])
                test_errs.append(curves['test']['stderr'][seq_idx])
            
            mem_means = np.array(mem_means)
            mem_errs = np.array(mem_errs)
            test_means = np.array(test_means)
            test_errs = np.array(test_errs)
            plt.errorbar(dimensions, mem_means, yerr=1.96*mem_errs, 
                        color=mem_color, linestyle='--', linewidth=2,
                        label='In-context train', capsize=3)
            plt.errorbar(dimensions, test_means, yerr=1.96*test_errs,
                        color=test_color, linestyle='-', linewidth=2,
                        label='Test', capsize=3)
            
            optimal_acc = 1.0 - label_flip_p
            plt.plot(dimensions, [optimal_acc] * len(dimensions), 
                    color=opt_color, linestyle='-.', linewidth=2, 
                    label=f'Optimal Test ({optimal_acc:.2f})')
            
            base_font = 18
            if force_y_range:
                plt.ylim(0.49, 1.01)
            plt.xlabel('Input Dimension (d)', fontsize = base_font+1)
            plt.ylabel('Accuracy', fontsize = base_font+1)

            plt.title(f'Performance vs Dimension ($\\tilde R=d^{{{R_d_to_power}}}$)\n(Seq. Length = {sequence_length}, Label Flip = {label_flip_p})', fontsize=base_font+2)
            plt.grid(True, alpha=0.4, color='gray', linewidth=0.5)
            plt.xticks(fontsize=base_font)
            plt.yticks(fontsize=base_font)
            plt.xscale('log')

            # Check if bottom right area is crowded by looking at the final values
            final_mem = mem_means[-1]  # Last memorization value
            final_test = test_means[-1]  # Last test value
            legend_threshold = 0.75  # Adjust this value to change sensitivity

            if final_mem > legend_threshold and final_test > legend_threshold:
                # If both lines are above threshold in bottom right, place legend there
                plt.legend(loc='lower right', fontsize=base_font)
            else:
                # Otherwise use the default center right position
                plt.legend(loc='center right', fontsize=base_font)
            
            if save_path:
                base_path = Path(save_path)
                # Ensure the parent directory exists
                base_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Add .png extension if no extension is provided
                if not base_path.suffix:
                    base_path = base_path.with_suffix('.png')
                    
                flip_specific_path = base_path.parent / f"{base_path.stem}_N{sequence_length}_R{R_d_to_power}_p{label_flip_p}{base_path.suffix}"
                plt.savefig(flip_specific_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot for label_flip_p={label_flip_p} to {flip_specific_path}")
            else:
                plt.show()
                
            plt.close()
    
    def evaluate_batch_sizes(self, 
                           dimension: int = 1000,
                           max_seq_length: int = 40,
                           R_d_to_power: Optional[float] = 0.3,
                           num_samples: int = 2500) -> Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]]:
        """
        Evaluate checkpoints for different batch sizes with fixed dimension.
        
        Args:
            dimension: Fixed dimension to analyze (default 1000)
            max_seq_length: Maximum sequence length to evaluate
            R: Optional radius parameter
            num_samples: Number of samples for evaluation
        """
        results_by_batch = {}

        if R_d_to_power is None:
            R = dimension**0.3
        else: 
            R = dimension**R_d_to_power
        
        # Find all checkpoints matching the dimension
        matches = list(self.checkpoint_dir.glob(f"checkpoint_d{dimension}*.pt"))
        
        for checkpoint_path in matches:
            # Extract batch size from filename
            # Assuming filename format contains "B{batch_size}"
            batch_str = str(checkpoint_path)
            batch_size = int(batch_str[batch_str.find('B')+1:].split('_')[0])
            print(f"\nEvaluating batch size {batch_size}")
            
            model, config = self.load_checkpoint(str(checkpoint_path))
            curves = self.evaluate_risk_curves(
                model=model,
                d=dimension,
                R=R,
                max_seq_length=max_seq_length + 1,
                num_samples=num_samples,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            results_by_batch[batch_size] = curves
            
        return results_by_batch

    def plot_batch_size_curves(self,
                             results_by_batch: Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]],
                             sequence_length: int = 40,
                             save_path: Optional[str] = None,
                             R_d_to_power: Optional[float] = 0.3,
                             label_flip_ps: Optional[List[float]] = None):
        """
        Plot accuracy vs batch size curves.
        """
        if label_flip_ps is None:
            label_flip_ps = self.label_flips

        if R_d_to_power is None:
            R_d_to_power = 0.3
            
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'
        
        batch_sizes = sorted(results_by_batch.keys())
        seq_idx = sequence_length - 2
        
        # Create separate plot for each label flip probability
        for label_flip_p in label_flip_ps:
            plt.figure(figsize=(8, 6))
            
            mem_means = []
            mem_errs = []
            test_means = []
            test_errs = []
            
            for batch_size in batch_sizes:
                curves = results_by_batch[batch_size][label_flip_p]
                mem_means.append(curves['memorization']['mean'][seq_idx])
                mem_errs.append(curves['memorization']['stderr'][seq_idx])
                test_means.append(curves['test']['mean'][seq_idx])
                test_errs.append(curves['test']['stderr'][seq_idx])
            
            mem_means = np.array(mem_means)
            mem_errs = np.array(mem_errs)
            test_means = np.array(test_means)
            test_errs = np.array(test_errs)
            
            plt.errorbar(batch_sizes, mem_means, yerr=1.96*mem_errs,
                        color='red', linestyle='--', linewidth=2,
                        label='In-context train', capsize=3)
            plt.errorbar(batch_sizes, test_means, yerr=1.96*test_errs,
                        color='blue', linestyle='-', linewidth=2,
                        label='Test', capsize=3)
            
            optimal_acc = 1.0 - label_flip_p
            plt.plot(batch_sizes, [optimal_acc] * len(batch_sizes),
                    color='green', linestyle='-.', linewidth=2,
                    label=f'Optimal Test ({optimal_acc:.2f})')
            
            base_font = 18
            plt.ylim(0.49, 1.01)
            plt.xlabel('Tasks (B)', fontsize=base_font+1)
            plt.ylabel('Accuracy', fontsize=base_font+1)
            plt.title(f'Performance vs Number of Tasks ($\\tilde R=d^{{{R_d_to_power}}}$)\n(d=1000, Seq. Length = {sequence_length}, Label Flip = {label_flip_p})', fontsize=base_font+2)
            plt.grid(True, alpha=0.4, color='gray', linewidth=0.5)
            plt.xticks(fontsize=base_font)
            plt.yticks(fontsize=base_font)
            plt.xscale('log')
            plt.legend(fontsize=base_font)
            
            if save_path:
                base_path = Path(save_path)
                base_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not base_path.suffix:
                    base_path = base_path.with_suffix('.png')
                    
                flip_specific_path = base_path.parent / f"{base_path.stem}_N{sequence_length}_R{R_d_to_power}_p{label_flip_p}{base_path.suffix}"
                plt.savefig(flip_specific_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot for label_flip_p={label_flip_p} to {flip_specific_path}")
            else:
                plt.show()
                
            plt.close()

    def evaluate_checkpoint(self, checkpoint_file: str, max_seq_length: int, R: Optional[float]=None,
                          label_flip_ps: Optional[List[float]]=None) -> Dict[int, Dict[float, Dict[str, np.ndarray]]]:
        """Evaluate a single checkpoint with specified maximum sequence length."""
        if label_flip_ps is None:
            label_flip_ps = self.label_flips
            
        print(f"\nEvaluating {checkpoint_file}")
        model, config = self.load_checkpoint(checkpoint_file)
        results = self.evaluate_risk_curves(
            model=model,
            d=config.d,
            R=R,
            max_seq_length=max_seq_length + 1,  # Add 1 to get desired sequence length
            num_samples=2500,
            label_flip_ps=label_flip_ps,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        return {config.d: results}

def main():
    evaluator = CheckpointEvaluator('checkpoints/', label_flips = [0.2])
    max_seq_length = 20

    # first run batch size results

    R_d_to_powers = [0.1, 0.3, 0.6]

    for R_d_to_power in R_d_to_powers:
        batch_results = evaluator.evaluate_batch_sizes(
            dimension=1000,
            max_seq_length=max_seq_length,
            num_samples=2500,
            R_d_to_power=R_d_to_power,
        )
        evaluator.plot_batch_size_curves(
            batch_results,
            sequence_length=max_seq_length,
            save_path='plots/batch_size_curves.png',
            R_d_to_power = R_d_to_power,
        )

    # then run high-dimensionality results

    all_results = {}
    dimensions = [10, 50, 100, 200, 400, 600, 800, 1000, 1250, 1500, 2000]

    R_d_to_powers = [0.1, 0.3, 0.6]

    
    for R_d_to_power in R_d_to_powers:
        for d in dimensions:
            matches = list(Path('checkpoints/').glob(f"checkpoint_d{d}*.pt"))
            if matches:
                for checkpoint_path in matches: 
                    batch_str = str(checkpoint_path)
                    batch_size = int(batch_str[batch_str.find('B')+1:].split('_')[0])
                    R = d**R_d_to_power
                    if batch_size == d: # only considering those where B=d
                        results = evaluator.evaluate_checkpoint(checkpoint_path, max_seq_length, R)
                        all_results.update(results)
        
        evaluator.plot_dimension_curves(all_results, sequence_length=max_seq_length, save_path="plots/dimension_curves.png", R_d_to_power=R_d_to_power, force_y_range = True)


if __name__ == "__main__":
    main()