# predict.py

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from ragst.models.ragst_model import RAGSTModel
from ragst.data.hest_dataset import create_data_loaders
from ragst.evaluation.metrics import EvaluationPipeline


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> RAGSTModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get('config', {})
    
    # Create model
    vision_config = model_config['model']['vision_encoder']
    classifier_config = model_config['model']['cell_type_classifier']
    generator_config = model_config['model'].get('generator', {})
    
    model = RAGSTModel(
        vision_config=vision_config,
        classifier_config=classifier_config,
        generator_config=generator_config
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_single_image(model: RAGSTModel, 
                        image_path: str, 
                        device: str = 'cuda',
                        image_size: int = 224) -> dict:
    """Predict gene expression from a single histology image."""
    from torchvision import transforms
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor, return_intermediate=True)
    
    # Extract results
    results = {
        'gene_expression': outputs['gene_expression'].cpu().numpy()[0],
        'cell_type_probs': outputs['cell_type_probs'].cpu().numpy()[0],
        'image_embeddings': outputs['image_embeddings'].cpu().numpy()[0]
    }
    
    return results


def visualize_prediction(image_path: str,
                        results: dict,
                        gene_names: list = None,
                        output_path: str = None,
                        top_genes: int = 20):
    """Visualize prediction results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Show original image
    image = Image.open(image_path)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Histology Image')
    axes[0, 0].axis('off')
    
    # Show cell type probabilities
    cell_type_probs = results['cell_type_probs']
    top_cell_types = np.argsort(cell_type_probs)[-10:][::-1]  # Top 10 cell types
    
    axes[0, 1].bar(range(len(top_cell_types)), cell_type_probs[top_cell_types])
    axes[0, 1].set_title('Top Cell Type Probabilities')
    axes[0, 1].set_xlabel('Cell Type Index')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_xticks(range(len(top_cell_types)))
    axes[0, 1].set_xticklabels([f'Type_{i}' for i in top_cell_types], rotation=45)
    
    # Show top expressed genes
    gene_expression = results['gene_expression']
    top_gene_indices = np.argsort(gene_expression)[-top_genes:][::-1]
    
    if gene_names and len(gene_names) > max(top_gene_indices):
        gene_labels = [gene_names[i] for i in top_gene_indices]
    else:
        gene_labels = [f'Gene_{i}' for i in top_gene_indices]
    
    axes[1, 0].barh(range(top_genes), gene_expression[top_gene_indices])
    axes[1, 0].set_title(f'Top {top_genes} Expressed Genes')
    axes[1, 0].set_xlabel('Expression Level')
    axes[1, 0].set_yticks(range(top_genes))
    axes[1, 0].set_yticklabels(gene_labels)
    
    # Show expression distribution
    axes[1, 1].hist(gene_expression, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Gene Expression Distribution')
    axes[1, 1].set_xlabel('Expression Level')
    axes[1, 1].set_ylabel('Number of Genes')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(model_path: str,
                  data_config: dict,
                  output_dir: str,
                  device: str = 'cuda'):
    """Evaluate model on test dataset."""
    
    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(model_path, device)
    model.to(device)
    
    # Create data loaders
    print("Loading test data...")
    data_loaders = create_data_loaders(data_config)
    
    # Create evaluator
    evaluator = EvaluationPipeline(model, device)
    
    # Evaluate on test set
    print("Running evaluation...")
    test_metrics = evaluator.evaluate_dataset(
        data_loaders['test'],
        save_predictions=True,
        output_dir=output_dir
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"  Pearson Correlation: {test_metrics.get('pearson_mean', 0):.4f} ± {test_metrics.get('pearson_std', 0):.4f}")
    print(f"  Spearman Correlation: {test_metrics.get('spearman_mean', 0):.4f} ± {test_metrics.get('spearman_std', 0):.4f}")
    print(f"  MSE: {test_metrics.get('mse', 0):.6f}")
    print(f"  RMSE: {test_metrics.get('rmse', 0):.4f}")
    print(f"  MAE: {test_metrics.get('mae', 0):.4f}")
    
    if 'top_10_overlap' in test_metrics:
        print(f"  Top-10 Gene Overlap: {test_metrics['top_10_overlap']:.4f}")
    
    return test_metrics


def batch_predict(model_path: str,
                 image_dir: str,
                 output_dir: str,
                 device: str = 'cuda',
                 image_size: int = 224):
    """Run batch prediction on a directory of images."""
    
    # Load model
    model = load_model_from_checkpoint(model_path, device)
    model.to(device)
    
    # Find all images
    image_dir = Path(image_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_predictions = {}
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        try:
            # Predict
            results = predict_single_image(model, str(image_file), device, image_size)
            
            # Save results
            sample_id = image_file.stem
            all_predictions[sample_id] = results
            
            # Save individual visualization
            viz_path = output_dir / f"{sample_id}_prediction.png"
            visualize_prediction(str(image_file), results, output_path=str(viz_path))
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    # Save all predictions
    predictions_file = output_dir / 'batch_predictions.npz'
    
    # Combine all predictions into arrays
    gene_expressions = np.stack([pred['gene_expression'] for pred in all_predictions.values()])
    cell_type_probs = np.stack([pred['cell_type_probs'] for pred in all_predictions.values()])
    sample_ids = list(all_predictions.keys())
    
    np.savez(predictions_file,
             gene_expressions=gene_expressions,
             cell_type_probs=cell_type_probs,
             sample_ids=sample_ids)
    
    print(f"Batch predictions saved to {predictions_file}")
    
    return all_predictions


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='RAG-ST Inference')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'evaluate'], 
                       required=True, help='Inference mode')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to data configuration file (for evaluate mode)')
    parser.add_argument('--input', type=str,
                       help='Input image path (single mode) or directory (batch mode)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--genes', type=str,
                       help='Path to gene names file (one per line)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load gene names if provided
    gene_names = None
    if args.genes:
        with open(args.genes, 'r') as f:
            gene_names = [line.strip() for line in f]
        print(f"Loaded {len(gene_names)} gene names")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'single':
        print(f"Running single image prediction on {args.input}")
        
        # Load model
        model = load_model_from_checkpoint(args.model, args.device)
        model.to(args.device)
        
        # Predict
        results = predict_single_image(model, args.input, args.device, args.image_size)
        
        # Visualize
        viz_path = output_dir / 'prediction_visualization.png'
        visualize_prediction(args.input, results, gene_names, str(viz_path))
        
        # Save numerical results
        np.save(output_dir / 'gene_expression.npy', results['gene_expression'])
        np.save(output_dir / 'cell_type_probs.npy', results['cell_type_probs'])
        
        print(f"Results saved to {output_dir}")
        
    elif args.mode == 'batch':
        print(f"Running batch prediction on {args.input}")
        
        predictions = batch_predict(
            args.model, args.input, str(output_dir), 
            args.device, args.image_size
        )
        
        print(f"Processed {len(predictions)} images")
        
    elif args.mode == 'evaluate':
        if not args.config:
            raise ValueError("Config file required for evaluation mode")
        
        print(f"Running model evaluation")
        
        # Load data config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        metrics = evaluate_model(
            args.model, config['data'], str(output_dir), args.device
        )
        
        # Save metrics
        with open(output_dir / 'evaluation_metrics.yaml', 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)


if __name__ == '__main__':
    main()