# train.py

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from ragst.models.ragst_model import RAGSTModel
from ragst.training.trainer import RAGSTTrainer
from ragst.data.hest_dataset import CellxGeneDataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> RAGSTModel:
    """Create RAG-ST model from configuration."""
    
    # Extract model configurations
    vision_config = config['model']['vision_encoder']
    classifier_config = config['model']['cell_type_classifier']
    generator_config = config['model'].get('generator', {})
    
    # Setup retrieval data if needed
    retrieval_data = None
    if config.get('use_retrieval', True) and 'cellxgene' in config:
        print("Loading CellxGene reference data for retrieval...")
        cellxgene_dataset = CellxGeneDataset(**config['cellxgene'])
        
        embeddings, expressions, cell_labels = cellxgene_dataset.get_embeddings_and_expressions()
        
        retrieval_data = {
            'scrna_embeddings': embeddings,
            'scrna_expressions': expressions,
            'cell_type_labels': cell_labels,
            'embedding_dim': embeddings.shape[1],
            'top_k': config.get('retrieval_top_k', 10)
        }
        
        print(f"Loaded {len(expressions)} scRNA-seq profiles for retrieval")
    
    # Create model
    model = RAGSTModel(
        vision_config=vision_config,
        classifier_config=classifier_config,
        generator_config=generator_config,
        retrieval_data=retrieval_data
    )
    
    return model


def load_pretrained_stage1(model: RAGSTModel, checkpoint_path: str):
    """Load pretrained stage 1 weights."""
    if Path(checkpoint_path).exists():
        print(f"Loading pretrained stage 1 model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load only vision encoder and classifier weights
        model_state = checkpoint.get('model_state_dict', checkpoint)
        
        # Filter stage 1 parameters
        stage1_state = {}
        for key, value in model_state.items():
            if key.startswith('vision_encoder.') or key.startswith('cell_type_classifier.'):
                stage1_state[key] = value
        
        # Load weights (ignore missing generator weights)
        model.load_state_dict(stage1_state, strict=False)
        print("Successfully loaded pretrained stage 1 weights")
    else:
        print(f"Warning: Pretrained model not found at {checkpoint_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RAG-ST model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    config['device'] = args.device
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Training configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Random seed: {args.seed}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
        config['device'] = 'cpu'
    
    # Create model
    print("Creating RAG-ST model...")
    model = create_model(config)
    
    # Load pretrained stage 1 if specified
    if config.get('pretrained_stage1'):
        load_pretrained_stage1(model, config['pretrained_stage1'])
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = RAGSTTrainer(config, model, args.device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.training_stage = checkpoint.get('training_stage', 1)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'stage1_optimizer_state_dict' in checkpoint:
            trainer.stage1_optimizer.load_state_dict(checkpoint['stage1_optimizer_state_dict'])
        if 'stage2_optimizer_state_dict' in checkpoint:
            trainer.stage2_optimizer.load_state_dict(checkpoint['stage2_optimizer_state_dict'])
        
        print(f"Resumed from epoch {trainer.epoch}, stage {trainer.training_stage}")
    
    # Start training
    try:
        trainer.train()
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Save current state
        checkpoint_path = output_dir / 'interrupted_checkpoint.pth'
        trainer.save_checkpoint({}, is_best=False)
        print(f"Saved current state to {checkpoint_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save current state for debugging
        try:
            checkpoint_path = output_dir / 'error_checkpoint.pth'
            trainer.save_checkpoint({}, is_best=False)
            print(f"Saved state for debugging to {checkpoint_path}")
        except:
            print("Could not save checkpoint")


if __name__ == '__main__':
    main()