# ragst/evaluation/metrics.py

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compute_correlation_metrics(predictions: np.ndarray, 
                               targets: np.ndarray) -> Dict[str, float]:
    """
    Compute correlation-based metrics between predictions and targets.
    
    Args:
        predictions: Predicted gene expressions [n_samples, n_genes]
        targets: True gene expressions [n_samples, n_genes]
        
    Returns:
        Dictionary containing correlation metrics
    """
    n_samples, n_genes = predictions.shape
    
    # Sample-wise correlations (across genes for each sample)
    sample_correlations = []
    sample_spearman = []
    
    for i in range(n_samples):
        # Pearson correlation
        pearson_corr, _ = stats.pearsonr(predictions[i], targets[i])
        sample_correlations.append(pearson_corr if not np.isnan(pearson_corr) else 0.0)
        
        # Spearman correlation
        spearman_corr, _ = stats.spearmanr(predictions[i], targets[i])
        sample_spearman.append(spearman_corr if not np.isnan(spearman_corr) else 0.0)
    
    # Gene-wise correlations (across samples for each gene)
    gene_correlations = []
    gene_spearman = []
    
    for j in range(n_genes):
        # Pearson correlation
        pearson_corr, _ = stats.pearsonr(predictions[:, j], targets[:, j])
        gene_correlations.append(pearson_corr if not np.isnan(pearson_corr) else 0.0)
        
        # Spearman correlation
        spearman_corr, _ = stats.spearmanr(predictions[:, j], targets[:, j])
        gene_spearman.append(spearman_corr if not np.isnan(spearman_corr) else 0.0)
    
    return {
        'pearson_mean': np.mean(sample_correlations),
        'pearson_std': np.std(sample_correlations),
        'pearson_median': np.median(sample_correlations),
        'spearman_mean': np.mean(sample_spearman),
        'spearman_std': np.std(sample_spearman),
        'spearman_median': np.median(sample_spearman),
        'gene_pearson_mean': np.mean(gene_correlations),
        'gene_pearson_std': np.std(gene_correlations),
        'gene_spearman_mean': np.mean(gene_spearman),
        'gene_spearman_std': np.std(gene_spearman)
    }


def compute_mse_metrics(predictions: np.ndarray, 
                       targets: np.ndarray) -> Dict[str, float]:
    """
    Compute MSE-based metrics.
    
    Args:
        predictions: Predicted gene expressions
        targets: True gene expressions
        
    Returns:
        Dictionary containing MSE metrics
    """
    # Overall metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    # Sample-wise metrics
    sample_mse = np.mean((predictions - targets) ** 2, axis=1)
    sample_mae = np.mean(np.abs(predictions - targets), axis=1)
    
    # Gene-wise metrics
    gene_mse = np.mean((predictions - targets) ** 2, axis=0)
    gene_mae = np.mean(np.abs(predictions - targets), axis=0)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'sample_mse_mean': np.mean(sample_mse),
        'sample_mse_std': np.std(sample_mse),
        'sample_mae_mean': np.mean(sample_mae),
        'sample_mae_std': np.std(sample_mae),
        'gene_mse_mean': np.mean(gene_mse),
        'gene_mse_std': np.std(gene_mse),
        'gene_mae_mean': np.mean(gene_mae),
        'gene_mae_std': np.std(gene_mae)
    }


def compute_biological_metrics(predictions: np.ndarray,
                              targets: np.ndarray,
                              gene_names: List[str]) -> Dict[str, float]:
    """
    Compute biological interpretation metrics.
    
    Args:
        predictions: Predicted gene expressions
        targets: True gene expressions  
        gene_names: List of gene names
        
    Returns:
        Dictionary containing biological metrics
    """
    # Top gene preservation
    top_k_values = [10, 50, 100]
    top_gene_metrics = {}
    
    for k in top_k_values:
        if k <= len(gene_names):
            # Find top-k genes for each sample
            pred_top_genes = np.argsort(predictions, axis=1)[:, -k:]
            true_top_genes = np.argsort(targets, axis=1)[:, -k:]
            
            # Compute overlap
            overlaps = []
            for i in range(predictions.shape[0]):
                overlap = len(np.intersect1d(pred_top_genes[i], true_top_genes[i]))
                overlaps.append(overlap / k)
            
            top_gene_metrics[f'top_{k}_overlap'] = np.mean(overlaps)
    
    # Expression range preservation
    pred_ranges = np.max(predictions, axis=1) - np.min(predictions, axis=1)
    true_ranges = np.max(targets, axis=1) - np.min(targets, axis=1)
    range_correlation, _ = stats.pearsonr(pred_ranges, true_ranges)
    
    # Variance preservation
    pred_vars = np.var(predictions, axis=1)
    true_vars = np.var(targets, axis=1)
    var_correlation, _ = stats.pearsonr(pred_vars, true_vars)
    
    return {
        **top_gene_metrics,
        'range_correlation': range_correlation if not np.isnan(range_correlation) else 0.0,
        'variance_correlation': var_correlation if not np.isnan(var_correlation) else 0.0
    }


class EvaluationPipeline:
    """Comprehensive evaluation pipeline for RAG-ST models."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = 'cuda',
                 gene_names: Optional[List[str]] = None):
        """
        Initialize evaluation pipeline.
        
        Args:
            model: Trained RAG-ST model
            device: Computation device
            gene_names: List of gene names for biological interpretation
        """
        self.model = model
        self.device = device
        self.gene_names = gene_names or []
    
    def evaluate_dataset(self, 
                        data_loader: torch.utils.data.DataLoader,
                        save_predictions: bool = True,
                        output_dir: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            save_predictions: Whether to save predictions for analysis
            output_dir: Output directory for saving results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_sample_ids = []
        all_images = []
        
        print("Generating predictions...")
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                expressions = batch['expression']
                sample_ids = batch.get('sample_id', [])
                
                # Forward pass
                predictions = self.model(images)
                
                # Collect results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(expressions.numpy())
                all_sample_ids.extend(sample_ids)
                
                # Save first few images for visualization
                if len(all_images) < 10:
                    all_images.extend(images.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        print(f"Evaluating {predictions.shape[0]} samples with {predictions.shape[1]} genes...")
        
        # Compute metrics
        correlation_metrics = compute_correlation_metrics(predictions, targets)
        mse_metrics = compute_mse_metrics(predictions, targets)
        
        if self.gene_names:
            bio_metrics = compute_biological_metrics(predictions, targets, self.gene_names)
        else:
            bio_metrics = {}
        
        # Combine all metrics
        all_metrics = {
            **correlation_metrics,
            **mse_metrics,
            **bio_metrics
        }
        
        # Save predictions and visualizations
        if save_predictions and output_dir:
            self._save_evaluation_results(
                predictions, targets, all_sample_ids, 
                all_metrics, output_dir, all_images[:10]
            )
        
        return all_metrics
    
    def _save_evaluation_results(self,
                                predictions: np.ndarray,
                                targets: np.ndarray,
                                sample_ids: List[str],
                                metrics: Dict[str, float],
                                output_dir: str,
                                sample_images: List[np.ndarray]):
        """Save evaluation results and create visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw predictions
        np.save(output_path / 'predictions.npy', predictions)
        np.save(output_path / 'targets.npy', targets)
        
        with open(output_path / 'sample_ids.txt', 'w') as f:
            for sid in sample_ids:
                f.write(f"{sid}\n")
        
        # Save metrics
        import json
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        self._create_evaluation_plots(predictions, targets, metrics, output_path)
        
        # Create sample-specific visualizations
        self._create_sample_plots(predictions, targets, sample_images, output_path)
    
    def _create_evaluation_plots(self,
                                predictions: np.ndarray,
                                targets: np.ndarray,
                                metrics: Dict[str, float],
                                output_dir: Path):
        """Create comprehensive evaluation plots."""
        
        # 1. Overall correlation scatter plot
        plt.figure(figsize=(10, 8))
        
        # Sample random points for visualization if dataset is large
        n_points = min(5000, predictions.size)
        indices = np.random.choice(predictions.size, n_points, replace=False)
        
        pred_flat = predictions.flatten()[indices]
        target_flat = targets.flatten()[indices]
        
        plt.scatter(target_flat, pred_flat, alpha=0.5, s=1)
        plt.plot([target_flat.min(), target_flat.max()], 
                [target_flat.min(), target_flat.max()], 'r--', lw=2)
        plt.xlabel('True Expression')
        plt.ylabel('Predicted Expression')
        plt.title(f'Gene Expression Prediction\nPearson r = {metrics.get("pearson_mean", 0):.3f}')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-sample correlation distribution
        n_samples = predictions.shape[0]
        sample_correlations = []
        
        for i in range(n_samples):
            corr, _ = stats.pearsonr(predictions[i], targets[i])
            sample_correlations.append(corr if not np.isnan(corr) else 0.0)
        
        plt.figure(figsize=(10, 6))
        plt.hist(sample_correlations, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(sample_correlations), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(sample_correlations):.3f}')
        plt.xlabel('Sample-wise Pearson Correlation')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sample-wise Correlations')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Gene-wise performance heatmap
        if len(self.gene_names) > 0 and len(self.gene_names) <= 100:
            gene_corrs = []
            for j in range(predictions.shape[1]):
                corr, _ = stats.pearsonr(predictions[:, j], targets[:, j])
                gene_corrs.append(corr if not np.isnan(corr) else 0.0)
            
            plt.figure(figsize=(15, 8))
            gene_names_display = self.gene_names[:len(gene_corrs)]
            
            # Create heatmap data
            heatmap_data = np.array(gene_corrs).reshape(1, -1)
            
            sns.heatmap(heatmap_data, 
                       xticklabels=gene_names_display,
                       yticklabels=['Correlation'],
                       cmap='RdYlBu_r',
                       center=0,
                       cbar_kws={'label': 'Pearson Correlation'})
            plt.title('Gene-wise Prediction Performance')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(output_dir / 'gene_performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Error analysis
        errors = predictions - targets
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error distribution
        axes[0, 0].hist(errors.flatten(), bins=100, alpha=0.7)
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        
        # Error vs true expression
        axes[0, 1].scatter(targets.flatten()[::100], errors.flatten()[::100], alpha=0.5, s=1)
        axes[0, 1].set_xlabel('True Expression')
        axes[0, 1].set_ylabel('Prediction Error')
        axes[0, 1].set_title('Error vs True Expression')
        axes[0, 1].axhline(0, color='red', linestyle='--')
        
        # Sample-wise MSE
        sample_mse = np.mean(errors**2, axis=1)
        axes[1, 0].hist(sample_mse, bins=50, alpha=0.7)
        axes[1, 0].set_xlabel('Sample MSE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sample-wise MSE Distribution')
        
        # Gene-wise MSE
        gene_mse = np.mean(errors**2, axis=0)
        axes[1, 1].hist(gene_mse, bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('Gene MSE')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Gene-wise MSE Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sample_plots(self,
                            predictions: np.ndarray,
                            targets: np.ndarray,
                            sample_images: List[np.ndarray],
                            output_dir: Path):
        """Create sample-specific visualization plots."""
        
        n_samples_to_show = min(6, len(sample_images), predictions.shape[0])
        
        fig, axes = plt.subplots(3, n_samples_to_show, figsize=(4*n_samples_to_show, 12))
        if n_samples_to_show == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_samples_to_show):
            # Show histology image
            if len(sample_images) > i:
                img = sample_images[i].transpose(1, 2, 0)  # CHW to HWC
                # Denormalize image
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Sample {i+1}')
                axes[0, i].axis('off')
            
            # Show gene expression comparison
            genes_to_show = min(50, predictions.shape[1])
            x_pos = np.arange(genes_to_show)
            
            axes[1, i].bar(x_pos - 0.2, targets[i, :genes_to_show], 0.4, 
                          label='True', alpha=0.7)
            axes[1, i].bar(x_pos + 0.2, predictions[i, :genes_to_show], 0.4, 
                          label='Predicted', alpha=0.7)
            axes[1, i].set_xlabel('Gene Index')
            axes[1, i].set_ylabel('Expression Level')
            axes[1, i].set_title(f'Expression Comparison')
            if i == 0:
                axes[1, i].legend()
            
            # Show scatter plot for this sample
            axes[2, i].scatter(targets[i], predictions[i], alpha=0.6, s=10)
            
            # Fit line
            min_val = min(targets[i].min(), predictions[i].min())
            max_val = max(targets[i].max(), predictions[i].max())
            axes[2, i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # Calculate correlation for this sample
            corr, _ = stats.pearsonr(targets[i], predictions[i])
            axes[2, i].set_xlabel('True Expression')
            axes[2, i].set_ylabel('Predicted Expression')
            axes[2, i].set_title(f'r = {corr:.3f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def benchmark_models(models_dict: Dict[str, torch.nn.Module],
                    data_loader: torch.utils.data.DataLoader,
                    device: str = 'cuda',
                    output_dir: str = 'benchmark_results') -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple models on the same dataset.
    
    Args:
        models_dict: Dictionary mapping model names to model instances
        data_loader: DataLoader for evaluation
        device: Computation device
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        
        evaluator = EvaluationPipeline(model, device)
        model_results = evaluator.evaluate_dataset(
            data_loader,
            save_predictions=True,
            output_dir=str(output_path / model_name)
        )
        
        results[model_name] = model_results
        
        print(f"{model_name} Results:")
        print(f"  Pearson Correlation: {model_results.get('pearson_mean', 0):.4f}")
        print(f"  MSE: {model_results.get('mse', 0):.4f}")
        print(f"  RMSE: {model_results.get('rmse', 0):.4f}")
    
    # Create comparison plots
    _create_benchmark_comparison(results, output_path)
    
    return results


def _create_benchmark_comparison(results: Dict[str, Dict[str, float]], 
                                output_dir: Path):
    """Create comparison plots for benchmarked models."""
    
    # Extract key metrics for comparison
    metrics_to_compare = [
        'pearson_mean', 'spearman_mean', 'mse', 'rmse',
        'gene_pearson_mean', 'top_10_overlap'
    ]
    
    model_names = list(results.keys())
    
    # Create comparison bar plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_compare):
        if i < len(axes):
            values = [results[model].get(metric, 0) for model in model_names]
            
            bars = axes[i].bar(model_names, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            
            # Color bars based on performance (higher is better for correlations, lower for errors)
            if 'correlation' in metric or 'overlap' in metric:
                colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(values)))
            else:  # MSE, RMSE - lower is better
                colors = plt.cm.Reds_r(np.linspace(0.4, 0.8, len(values)))
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for i in range(len(metrics_to_compare), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    import pandas as pd
    
    df_data = {}
    for metric in metrics_to_compare:
        df_data[metric] = [results[model].get(metric, 0) for model in model_names]
    
    df = pd.DataFrame(df_data, index=model_names)
    df.to_csv(output_dir / 'benchmark_results.csv')
    
    print(f"\nBenchmark results saved to {output_dir}")
    print("\nSummary:")
    print(df.round(4))