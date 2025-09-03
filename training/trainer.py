# ragst/training/trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import wandb
from pathlib import Path
import json

from ..models.ragst_model import RAGSTModel
from ..data.hest_dataset import create_data_loaders, CellxGeneDataset
from ..evaluation.metrics import compute_correlation_metrics, compute_mse_metrics

class RAGSTTrainer:
    """Trainer class for RAG-ST model with two-stage training."""
    
    def __init__(self, 
                 config: Dict,
                 model: RAGSTModel,
                 device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            model: RAG-ST model instance
            device: Training device
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Setup data loaders
        self.data_loaders = create_data_loaders(config['data'])
        
        # Setup CellxGene reference data
        if config.get('use_retrieval', True):
            self._setup_retrieval_data()
        
        # Setup optimizers and schedulers
        self._setup_optimizers()
        
        # Setup loss functions
        self._setup_loss_functions()
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_stage = 1  # Start with stage 1
        
    def _setup_retrieval_data(self):
        """Setup retrieval module with CellxGene data."""
        cellxgene_config = self.config.get('cellxgene', {})
        cellxgene_dataset = CellxGeneDataset(**cellxgene_config)
        
        # Get embeddings and expressions for retrieval
        embeddings, expressions, cell_labels = cellxgene_dataset.get_embeddings_and_expressions()
        
        # Initialize retrieval module
        from ..models.ragst_model import RetrievalModule
        self.model.retrieval_module = RetrievalModule(
            scrna_embeddings=embeddings,
            scrna_expressions=expressions,
            cell_type_labels=cell_labels,
            embedding_dim=embeddings.shape[1],
            top_k=self.config.get('retrieval_top_k', 10)
        )
        
    def _setup_optimizers(self):
        """Setup optimizers for different training stages."""
        stage1_params = list(self.model.vision_encoder.parameters()) + \
                       list(self.model.cell_type_classifier.parameters())
        
        stage2_params = list(self.model.generator.parameters())
        if hasattr(self.model, 'retrieval_module'):
            # Retrieval module doesn't have trainable parameters
            pass
        
        self.stage1_optimizer = optim.AdamW(
            stage1_params,
            lr=self.config['training']['stage1_lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.stage2_optimizer = optim.AdamW(
            stage2_params,
            lr=self.config['training']['stage2_lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate schedulers
        self.stage1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.stage1_optimizer,
            T_max=self.config['training']['stage1_epochs']
        )
        
        self.stage2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.stage2_optimizer,
            T_max=self.config['training']['stage2_epochs']
        )
    
    def _setup_loss_functions(self):
        """Setup loss functions for training."""
        # Cell type classification loss
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Gene expression regression loss
        self.regression_loss = nn.MSELoss()
        
        # Optional: Correlation-based loss
        self.use_correlation_loss = self.config['training'].get('use_correlation_loss', False)
        
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard logging
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Weights & Biases logging
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'rag-st'),
                config=self.config,
                name=self.config.get('experiment_name', 'rag-st-experiment')
            )
        
    def correlation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute correlation-based loss."""
        # Center the tensors
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        
        # Compute correlations
        numerator = (pred_centered * target_centered).sum(dim=1)
        pred_norm = torch.sqrt((pred_centered ** 2).sum(dim=1))
        target_norm = torch.sqrt((target_centered ** 2).sum(dim=1))
        
        correlations = numerator / (pred_norm * target_norm + 1e-8)
        
        # Return negative correlation as loss
        return -correlations.mean()
    
    def train_stage1(self) -> Dict[str, float]:
        """Train stage 1: Image to cell type mapping."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.data_loaders['train'], desc=f"Stage 1 - Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            cell_types = batch['cell_type'].to(self.device)
            
            self.stage1_optimizer.zero_grad()
            
            # Forward pass through vision encoder and classifier
            image_embeddings = self.model.vision_encoder(images)
            cell_type_logits = self.model.cell_type_classifier(image_embeddings)
            
            # Compute classification loss
            loss = self.classification_loss(cell_type_logits, cell_types)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.vision_encoder.parameters()) + 
                list(self.model.cell_type_classifier.parameters()),
                max_norm=1.0
            )
            self.stage1_optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = cell_type_logits.argmax(dim=1)
            correct_predictions += (predictions == cell_types).sum().item()
            total_samples += cell_types.size(0)
            
            # Update progress bar
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
            
            # Log batch metrics
            if batch_idx % self.config['training'].get('log_interval', 100) == 0:
                self.writer.add_scalar('Stage1/BatchLoss', loss.item(), 
                                     self.epoch * len(self.data_loaders['train']) + batch_idx)
        
        avg_loss = total_loss / len(self.data_loaders['train'])
        accuracy = correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train_stage2(self) -> Dict[str, float]:
        """Train stage 2: Retrieval-augmented generation."""
        self.model.train()
        
        # Freeze stage 1 parameters
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.model.cell_type_classifier.parameters():
            param.requires_grad = False
        
        total_loss = 0.0
        total_mse = 0.0
        total_correlation = 0.0
        
        progress_bar = tqdm(self.data_loaders['train'], desc=f"Stage 2 - Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            true_expressions = batch['expression'].to(self.device)
            
            self.stage2_optimizer.zero_grad()
            
            # Forward pass through full model
            outputs = self.model(images, return_intermediate=True)
            predicted_expressions = outputs['gene_expression']
            
            # Compute regression loss
            mse_loss = self.regression_loss(predicted_expressions, true_expressions)
            loss = mse_loss
            
            # Add correlation loss if enabled
            if self.use_correlation_loss:
                corr_loss = self.correlation_loss(predicted_expressions, true_expressions)
                loss = loss + 0.1 * corr_loss  # Weight correlation loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(),
                max_norm=1.0
            )
            self.stage2_optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mse += mse_loss.item()
            
            # Compute correlation
            with torch.no_grad():
                correlation = compute_correlation_metrics(
                    predicted_expressions.cpu().numpy(),
                    true_expressions.cpu().numpy()
                )
                total_correlation += correlation['pearson_mean']
            
            # Update progress bar
            avg_corr = total_correlation / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'Corr': f'{avg_corr:.4f}'
            })
            
            # Log batch metrics
            if batch_idx % self.config['training'].get('log_interval', 100) == 0:
                step = self.epoch * len(self.data_loaders['train']) + batch_idx
                self.writer.add_scalar('Stage2/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Stage2/BatchMSE', mse_loss.item(), step)
        
        avg_loss = total_loss / len(self.data_loaders['train'])
        avg_mse = total_mse / len(self.data_loaders['train'])
        avg_correlation = total_correlation / len(self.data_loaders['train'])
        
        return {
            'loss': avg_loss,
            'mse': avg_mse,
            'correlation': avg_correlation
        }
    
    def validate(self, stage: int) -> Dict[str, float]:
        """Validate model performance."""
        self.model.eval()
        
        if stage == 1:
            return self._validate_stage1()
        else:
            return self._validate_stage2()
    
    def _validate_stage1(self) -> Dict[str, float]:
        """Validate stage 1 performance."""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                images = batch['image'].to(self.device)
                cell_types = batch['cell_type'].to(self.device)
                
                image_embeddings = self.model.vision_encoder(images)
                cell_type_logits = self.model.cell_type_classifier(image_embeddings)
                
                loss = self.classification_loss(cell_type_logits, cell_types)
                total_loss += loss.item()
                
                predictions = cell_type_logits.argmax(dim=1)
                correct_predictions += (predictions == cell_types).sum().item()
                total_samples += cell_types.size(0)
        
        avg_loss = total_loss / len(self.data_loaders['val'])
        accuracy = correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _validate_stage2(self) -> Dict[str, float]:
        """Validate stage 2 performance."""
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                images = batch['image'].to(self.device)
                true_expressions = batch['expression'].to(self.device)
                
                predicted_expressions = self.model(images)
                
                loss = self.regression_loss(predicted_expressions, true_expressions)
                total_loss += loss.item()
                
                all_predictions.append(predicted_expressions.cpu().numpy())
                all_targets.append(true_expressions.cpu().numpy())
        
        # Compute comprehensive metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        correlation_metrics = compute_correlation_metrics(predictions, targets)
        mse_metrics = compute_mse_metrics(predictions, targets)
        
        results = {
            'loss': total_loss / len(self.data_loaders['val']),
            **correlation_metrics,
            **mse_metrics
        }
        
        return results
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'training_stage': self.training_stage,
            'model_state_dict': self.model.state_dict(),
            'stage1_optimizer_state_dict': self.stage1_optimizer.state_dict(),
            'stage2_optimizer_state_dict': self.stage2_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # Keep only recent checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = list(self.output_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 5:  # Keep only 5 most recent
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
    
    def train(self):
        """Main training loop."""
        print("Starting RAG-ST training...")
        
        # Stage 1: Train cell type classifier
        print(f"\n=== Stage 1: Cell Type Classification ===")
        self.training_stage = 1
        
        for epoch in range(self.config['training']['stage1_epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_stage1()
            
            # Validate
            val_metrics = self.validate(stage=1)
            
            # Learning rate scheduling
            self.stage1_scheduler.step()
            
            # Logging
            self.writer.add_scalar('Stage1/Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Stage1/Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Stage1/Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Stage1/Val/Accuracy', val_metrics['accuracy'], epoch)
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'stage1_train_loss': train_metrics['loss'],
                    'stage1_train_acc': train_metrics['accuracy'],
                    'stage1_val_loss': val_metrics['loss'],
                    'stage1_val_acc': val_metrics['accuracy'],
                    'epoch': epoch
                })
            
            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(val_metrics)
        
        # Stage 2: Train retrieval-augmented generator
        print(f"\n=== Stage 2: Retrieval-Augmented Generation ===")
        self.training_stage = 2
        
        for epoch in range(self.config['training']['stage2_epochs']):
            self.epoch = epoch + self.config['training']['stage1_epochs']
            
            # Train
            train_metrics = self.train_stage2()
            
            # Validate
            val_metrics = self.validate(stage=2)
            
            # Learning rate scheduling
            self.stage2_scheduler.step()
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Logging
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Stage2/Train/{key.capitalize()}', value, self.epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Stage2/Val/{key.capitalize()}', value, self.epoch)
            
            if self.config.get('use_wandb', False):
                log_dict = {f'stage2_train_{k}': v for k, v in train_metrics.items()}
                log_dict.update({f'stage2_val_{k}': v for k, v in val_metrics.items()})
                log_dict['epoch'] = self.epoch
                wandb.log(log_dict)
            
            print(f"Epoch {self.epoch}: "
                  f"Train Loss={train_metrics['loss']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Corr={val_metrics.get('pearson_mean', 0):.4f}")
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best=is_best)
        
        print("Training completed!")
        
        # Close logging
        self.writer.close()
        if self.config.get('use_wandb', False):
            wandb.finish()