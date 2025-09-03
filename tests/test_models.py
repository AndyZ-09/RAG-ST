# tests/test_models.py

import pytest
import torch
import numpy as np
from ragst.models.ragst_model import RAGSTModel, VisionEncoder, CellTypeClassifier, GeneratorModule
from ragst.models.baselines import create_baseline_model


class TestVisionEncoder:
    """Test cases for vision encoder component."""
    
    def test_vision_encoder_forward(self):
        """Test forward pass through vision encoder."""
        encoder = VisionEncoder(
            backbone="vit-base",
            output_dim=768
        )
        
        # Create dummy input
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        embeddings = encoder(images)
        
        # Check output shape
        assert embeddings.shape == (batch_size, 768)
        assert not torch.isnan(embeddings).any()
    
    def test_vision_encoder_different_dims(self):
        """Test vision encoder with different output dimensions."""
        for output_dim in [256, 512, 1024]:
            encoder = VisionEncoder(output_dim=output_dim)
            images = torch.randn(1, 3, 224, 224)
            embeddings = encoder(images)
            assert embeddings.shape == (1, output_dim)


class TestCellTypeClassifier:
    """Test cases for cell type classifier."""
    
    def test_classifier_forward(self):
        """Test forward pass through classifier."""
        classifier = CellTypeClassifier(
            input_dim=768,
            num_cell_types=50
        )
        
        embeddings = torch.randn(3, 768)
        logits = classifier(embeddings)
        
        assert logits.shape == (3, 50)
        
        # Check probabilities sum to 1
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3))
    
    def test_classifier_different_configs(self):
        """Test classifier with different configurations."""
        configs = [
            {'input_dim': 512, 'num_cell_types': 20, 'hidden_dims': [256]},
            {'input_dim': 1024, 'num_cell_types': 100, 'hidden_dims': [512, 256, 128]}
        ]
        
        for config in configs:
            classifier = CellTypeClassifier(**config)
            embeddings = torch.randn(2, config['input_dim'])
            logits = classifier(embeddings)
            assert logits.shape == (2, config['num_cell_types'])


class TestGeneratorModule:
    """Test cases for generator module."""
    
    def test_generator_forward(self):
        """Test forward pass through generator."""
        generator = GeneratorModule(
            image_dim=768,
            scrna_dim=2000,
            output_dim=2000
        )
        
        batch_size = 2
        top_k = 5
        
        image_embeddings = torch.randn(batch_size, 768)
        retrieved_expressions = torch.randn(batch_size, top_k, 2000)
        retrieval_weights = torch.ones(batch_size, top_k) / top_k
        
        output = generator(image_embeddings, retrieved_expressions, retrieval_weights)
        
        assert output.shape == (batch_size, 2000)
        assert not torch.isnan(output).any()


class TestRAGSTModel:
    """Test cases for complete RAG-ST model."""
    
    def test_ragst_model_creation(self):
        """Test RAG-ST model creation."""
        vision_config = {
            'backbone': 'vit-base',
            'output_dim': 768
        }
        classifier_config = {
            'input_dim': 768,
            'num_cell_types': 50
        }
        generator_config = {
            'image_dim': 768,
            'scrna_dim': 2000,
            'output_dim': 2000
        }
        
        model = RAGSTModel(
            vision_config=vision_config,
            classifier_config=classifier_config,
            generator_config=generator_config
        )
        
        # Test forward pass without retrieval
        images = torch.randn(2, 3, 224, 224)
        output = model(images)
        
        assert output.shape == (2, 2000)
    
    def test_ragst_model_with_retrieval(self):
        """Test RAG-ST model with retrieval module."""
        # Create dummy retrieval data
        n_refs = 1000
        embedding_dim = 50
        n_genes = 2000
        
        scrna_embeddings = np.random.randn(n_refs, embedding_dim).astype(np.float32)
        scrna_expressions = np.random.exponential(1.0, (n_refs, n_genes)).astype(np.float32)
        cell_type_labels = np.random.randint(0, 10, n_refs)
        
        retrieval_data = {
            'scrna_embeddings': scrna_embeddings,
            'scrna_expressions': scrna_expressions,
            'cell_type_labels': cell_type_labels,
            'embedding_dim': embedding_dim,
            'top_k': 5
        }
        
        vision_config = {'backbone': 'vit-base', 'output_dim': 768}
        classifier_config = {'input_dim': 768, 'num_cell_types': 10}
        generator_config = {'image_dim': 768, 'scrna_dim': n_genes, 'output_dim': n_genes}
        
        model = RAGSTModel(
            vision_config=vision_config,
            classifier_config=classifier_config,
            generator_config=generator_config,
            retrieval_data=retrieval_data
        )
        
        # Test forward pass with retrieval
        images = torch.randn(2, 3, 224, 224)
        output = model(images, return_intermediate=True)
        
        assert 'gene_expression' in output
        assert 'cell_type_probs' in output
        assert output['gene_expression'].shape == (2, n_genes)
        assert output['cell_type_probs'].shape == (2, 10)


class TestBaselines:
    """Test cases for baseline models."""
    
    def test_all_baselines(self):
        """Test all baseline models."""
        model_names = [
            'direct_regression',
            'cnn',
            'attention_pooling',
            'multiscale',
            'linear'
        ]
        
        batch_size = 2
        num_genes = 100  # Smaller for faster testing
        images = torch.randn(batch_size, 3, 224, 224)
        
        for model_name in model_names:
            model = create_baseline_model(model_name, num_genes=num_genes)
            
            # Test forward pass
            output = model(images)
            assert output.shape == (batch_size, num_genes)
            assert not torch.isnan(output).any()
    
    def test_ensemble_baseline(self):
        """Test ensemble baseline specifically."""
        model = create_baseline_model('ensemble', num_genes=50, num_models=2)
        images = torch.randn(1, 3, 224, 224)
        
        output = model(images)
        assert output.shape == (1, 50)
        
        # Check that ensemble weights are learnable
        assert model.ensemble_weights.requires_grad


# tests/test_data.py

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import h5py

from ragst.data.hest_dataset import HESTDataset, CellxGeneDataset, create_data_loaders


class TestHESTDataset:
    """Test cases for HEST dataset."""
    
    @pytest.fixture
    def dummy_hest_data(self):
        """Create dummy HEST data for testing."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)
        
        # Create directory structure
        (data_dir / 'images').mkdir()
        (data_dir / 'expressions').mkdir()
        
        # Create dummy samples
        n_samples = 10
        n_genes = 100
        
        for i in range(n_samples):
            sample_id = f"sample_{i:03d}"
            
            # Create dummy image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(data_dir / 'images' / f"{sample_id}.png")
            
            # Create dummy expression
            expression = np.random.exponential(1.0, n_genes).astype(np.float32)
            
            with h5py.File(data_dir / 'expressions' / f"{sample_id}.h5", 'w') as f:
                f.create_dataset('expression', data=expression)
                f.create_dataset('gene_names', 
                               data=[f"Gene_{j}".encode() for j in range(n_genes)])
                f.create_dataset('cell_type', data=b'T_cell')
        
        yield str(data_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_hest_dataset_creation(self, dummy_hest_data):
        """Test HEST dataset creation."""
        dataset = HESTDataset(
            data_dir=dummy_hest_data,
            split='train',
            max_genes=100
        )
        
        assert len(dataset) > 0
        assert len(dataset.gene_names) == 100
    
    def test_hest_dataset_getitem(self, dummy_hest_data):
        """Test getting items from HEST dataset."""
        dataset = HESTDataset(
            data_dir=dummy_hest_data,
            split='train',
            max_genes=100
        )
        
        sample = dataset[0]
        
        assert 'image' in sample
        assert 'expression' in sample
        assert 'cell_type' in sample
        assert 'sample_id' in sample
        
        assert sample['image'].shape == (3, 224, 224)
        assert sample['expression'].shape == (100,)
        assert isinstance(sample['cell_type'].item(), int)
    
    def test_hest_dataset_transforms(self, dummy_hest_data):
        """Test dataset with augmentations."""
        dataset = HESTDataset(
            data_dir=dummy_hest_data,
            split='train',
            augment=True,
            max_genes=100
        )
        
        # Get same sample multiple times to check augmentation
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Images should be different due to augmentation
        assert not torch.equal(sample1['image'], sample2['image'])


class TestCellxGeneDataset:
    """Test cases for CellxGene dataset."""
    
    def test_cellxgene_dataset_dummy(self):
        """Test CellxGene dataset with dummy data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = CellxGeneDataset(
                data_dir=temp_dir,
                tissue_types=['liver', 'lung'],
                max_genes=100,
                max_cells_per_type=50
            )
            
            assert len(dataset) > 0
            assert len(dataset.cell_type_mapping) > 0
            
            # Test getting embeddings
            embeddings, expressions, labels = dataset.get_embeddings_and_expressions()
            
            assert embeddings.shape[0] == expressions.shape[0]
            assert len(labels) == expressions.shape[0]
            assert expressions.shape[1] <= 100  # Max genes


class TestDataLoaders:
    """Test data loader creation."""
    
    def test_create_data_loaders(self, dummy_hest_data):
        """Test data loader creation."""
        config = {
            'data_dir': dummy_hest_data,
            'batch_size': 2,
            'num_workers': 0,  # Use 0 for testing
            'max_genes': 100
        }
        
        data_loaders = create_data_loaders(config)
        
        assert 'train' in data_loaders
        assert 'val' in data_loaders
        assert 'test' in data_loaders
        
        # Test a batch
        train_loader = data_loaders['train']
        batch = next(iter(train_loader))
        
        assert 'image' in batch
        assert 'expression' in batch
        assert batch['image'].shape[0] <= 2  # Batch size
        assert batch['expression'].shape[1] == 100  # Number of genes


# tests/test_training.py

import pytest
import torch
import tempfile
from unittest.mock import Mock, patch

from ragst.training.trainer import RAGSTTrainer
from ragst.models.ragst_model import RAGSTModel


class TestRAGSTTrainer:
    """Test cases for RAG-ST trainer."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        config = {
            'data': {
                'data_dir': 'dummy',
                'batch_size': 2,
                'num_workers': 0,
                'max_genes': 100
            },
            'training': {
                'stage1_lr': 1e-3,
                'stage2_lr': 1e-4,
                'weight_decay': 1e-4,
                'stage1_epochs': 1,
                'stage2_epochs': 1
            },
            'output_dir': tempfile.mkdtemp(),
            'use_wandb': False
        }
        
        # Create model
        vision_config = {'backbone': 'vit-base', 'output_dim': 768}
        classifier_config = {'input_dim': 768, 'num_cell_types': 10}
        generator_config = {'image_dim': 768, 'scrna_dim': 100, 'output_dim': 100}
        
        model = RAGSTModel(
            vision_config=vision_config,
            classifier_config=classifier_config,
            generator_config=generator_config
        )
        
        # Mock data loaders to avoid file I/O
        with patch('ragst.training.trainer.create_data_loaders') as mock_loaders:
            mock_loaders.return_value = {
                'train': Mock(),
                'val': Mock(),
                'test': Mock()
            }
            
            trainer = RAGSTTrainer(config, model, device='cpu')
            
            assert trainer.model is not None
            assert trainer.stage1_optimizer is not None
            assert trainer.stage2_optimizer is not None


# tests/test_evaluation.py

import pytest
import torch
import numpy as np
from ragst.evaluation.metrics import (
    compute_correlation_metrics,
    compute_mse_metrics,
    compute_biological_metrics
)


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_correlation_metrics(self):
        """Test correlation metrics computation."""
        # Create dummy data with known correlation
        n_samples, n_genes = 100, 50
        
        # Perfect correlation case
        predictions = np.random.randn(n_samples, n_genes)
        targets = predictions.copy()
        
        metrics = compute_correlation_metrics(predictions, targets)
        
        assert metrics['pearson_mean'] > 0.99
        assert metrics['spearman_mean'] > 0.99
    
    def test_mse_metrics(self):
        """Test MSE metrics computation."""
        predictions = np.random.randn(100, 50)
        targets = predictions + np.random.normal(0, 0.1, (100, 50))  # Add small noise
        
        metrics = compute_mse_metrics(predictions, targets)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mse'] > 0
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
    
    def test_biological_metrics(self):
        """Test biological interpretation metrics."""
        n_samples, n_genes = 20, 100
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Create data where top genes are preserved
        predictions = np.random.exponential(1.0, (n_samples, n_genes))
        targets = predictions + np.random.normal(0, 0.1, (n_samples, n_genes))
        
        metrics = compute_biological_metrics(predictions, targets, gene_names)
        
        assert 'top_10_overlap' in metrics
        assert 'range_correlation' in metrics
        assert 'variance_correlation' in metrics
        assert 0 <= metrics['top_10_overlap'] <= 1


# tests/test_integration.py

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from ragst.models.ragst_model import RAGSTModel
from ragst.models.baselines import create_baseline_model


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_inference(self):
        """Test end-to-end inference pipeline."""
        # Create model
        vision_config = {'backbone': 'vit-base', 'output_dim': 256}
        classifier_config = {'input_dim': 256, 'num_cell_types': 5}
        generator_config = {'image_dim': 256, 'scrna_dim': 50, 'output_dim': 50}
        
        model = RAGSTModel(
            vision_config=vision_config,
            classifier_config=classifier_config,
            generator_config=generator_config
        )
        
        # Test inference
        model.eval()
        with torch.no_grad():
            images = torch.randn(1, 3, 224, 224)
            output = model(images)
            
            assert output.shape == (1, 50)
            assert not torch.isnan(output).any()
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model
            vision_config = {'backbone': 'vit-base', 'output_dim': 256}
            classifier_config = {'input_dim': 256, 'num_cell_types': 5}
            generator_config = {'image_dim': 256, 'scrna_dim': 50, 'output_dim': 50}
            
            model1 = RAGSTModel(
                vision_config=vision_config,
                classifier_config=classifier_config,
                generator_config=generator_config
            )
            
            # Save model
            checkpoint_path = Path(temp_dir) / 'model.pth'
            model1.save_checkpoint(
                str(checkpoint_path),
                epoch=1,
                metrics={'test_metric': 0.5}
            )
            
            # Load model
            model2 = RAGSTModel.load_pretrained(str(checkpoint_path))
            
            # Test that models produce same output
            model1.eval()
            model2.eval()
            
            with torch.no_grad():
                images = torch.randn(1, 3, 224, 224)
                output1 = model1(images)
                output2 = model2(images)
                
                assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_baseline_comparison(self):
        """Test that we can compare against baselines."""
        baselines = ['direct_regression', 'linear']
        images = torch.randn(2, 3, 224, 224)
        
        results = {}
        
        for baseline_name in baselines:
            model = create_baseline_model(baseline_name, num_genes=50)
            model.eval()
            
            with torch.no_grad():
                output = model(images)
                results[baseline_name] = output
        
        # Check that different models produce different outputs
        assert not torch.allclose(results['direct_regression'], results['linear'])


# Run tests with pytest command
if __name__ == "__main__":
    pytest.main([__file__])