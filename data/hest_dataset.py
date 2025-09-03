# ragst/data/hest_dataset.py

import os
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Optional, Tuple, Union
import scanpy as sc
from pathlib import Path

class HESTDataset(Dataset):
    """Dataset class for HEST-1K histology and spatial transcriptomics data."""
    
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 image_size: int = 224,
                 max_genes: int = 2000,
                 normalize_expression: bool = True,
                 cell_type_mapping: Optional[Dict] = None,
                 augment: bool = True):
        """
        Initialize HEST dataset.
        
        Args:
            data_dir: Root directory containing HEST data
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size for preprocessing
            max_genes: Maximum number of genes to include
            normalize_expression: Whether to normalize gene expression
            cell_type_mapping: Mapping from cell type names to indices
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_genes = max_genes
        self.normalize_expression = normalize_expression
        self.cell_type_mapping = cell_type_mapping or {}
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.samples = self._load_sample_list()
        
        # Setup image transforms
        self.transform = self._setup_transforms(augment)
        
        # Load gene information
        self.gene_names = self._load_gene_names()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        metadata_path = self.data_dir / "metadata.csv"
        if metadata_path.exists():
            return pd.read_csv(metadata_path)
        else:
            # Create dummy metadata if not available
            return pd.DataFrame()
    
    def _load_sample_list(self) -> List[Dict]:
        """Load list of samples for the current split."""
        split_file = self.data_dir / f"{self.split}_samples.json"
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                samples = json.load(f)
        else:
            # Fallback: scan directory for available samples
            samples = self._scan_samples()
            
        return samples
    
    def _scan_samples(self) -> List[Dict]:
        """Scan data directory to find available samples."""
        samples = []
        
        # Look for paired image and expression files
        image_dir = self.data_dir / "images"
        expr_dir = self.data_dir / "expressions"
        
        if image_dir.exists() and expr_dir.exists():
            for img_file in image_dir.glob("*.png"):
                sample_id = img_file.stem
                expr_file = expr_dir / f"{sample_id}.h5"
                
                if expr_file.exists():
                    samples.append({
                        'sample_id': sample_id,
                        'image_path': str(img_file),
                        'expression_path': str(expr_file)
                    })
        
        # Split samples (80/10/10 train/val/test)
        n_samples = len(samples)
        if self.split == "train":
            samples = samples[:int(0.8 * n_samples)]
        elif self.split == "val":
            samples = samples[int(0.8 * n_samples):int(0.9 * n_samples)]
        else:  # test
            samples = samples[int(0.9 * n_samples):]
            
        return samples
    
    def _setup_transforms(self, augment: bool) -> transforms.Compose:
        """Setup image preprocessing transforms."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ]
        
        if augment and self.split == "train":
            # Add augmentations for training
            transform_list = [
                transforms.Resize((self.image_size + 32, self.image_size + 32)),
                transforms.RandomCrop((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
            ] + transform_list[1:]  # Skip initial resize
        
        return transforms.Compose(transform_list)
    
    def _load_gene_names(self) -> List[str]:
        """Load gene names from the first available sample."""
        if self.samples:
            expr_path = self.samples[0]['expression_path']
            with h5py.File(expr_path, 'r') as f:
                if 'gene_names' in f:
                    gene_names = [name.decode() for name in f['gene_names'][:]]
                else:
                    # Create dummy gene names
                    n_genes = f['expression'].shape[1]
                    gene_names = [f"Gene_{i}" for i in range(n_genes)]
        else:
            gene_names = [f"Gene_{i}" for i in range(self.max_genes)]
        
        return gene_names[:self.max_genes]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample_info = self.samples[idx]
        
        # Load image
        image = Image.open(sample_info['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Load gene expression
        with h5py.File(sample_info['expression_path'], 'r') as f:
            expression = f['expression'][:]
            
            # Get cell type if available
            cell_type = 0  # Default cell type
            if 'cell_type' in f:
                cell_type_name = f['cell_type'][()].decode()
                cell_type = self.cell_type_mapping.get(cell_type_name, 0)
        
        # Process expression data
        expression = torch.tensor(expression[:self.max_genes], dtype=torch.float32)
        
        if self.normalize_expression:
            # Log1p normalization
            expression = torch.log1p(expression)
            
            # Optional: standardize
            expression = (expression - expression.mean()) / (expression.std() + 1e-8)
        
        return {
            'image': image,
            'expression': expression,
            'cell_type': torch.tensor(cell_type, dtype=torch.long),
            'sample_id': sample_info['sample_id']
        }


class CellxGeneDataset(Dataset):
    """Dataset class for CellxGene Census single-cell reference data."""
    
    def __init__(self,
                 data_dir: str,
                 tissue_types: Optional[List[str]] = None,
                 cell_types: Optional[List[str]] = None,
                 max_genes: int = 2000,
                 max_cells_per_type: int = 1000):
        """
        Initialize CellxGene dataset.
        
        Args:
            data_dir: Directory containing processed CellxGene data
            tissue_types: List of tissue types to include
            cell_types: List of cell types to include
            max_genes: Maximum number of genes
            max_cells_per_type: Maximum cells per cell type
        """
        self.data_dir = Path(data_dir)
        self.tissue_types = tissue_types
        self.cell_types = cell_types
        self.max_genes = max_genes
        self.max_cells_per_type = max_cells_per_type
        
        # Load processed data
        self.adata = self._load_anndata()
        self.cell_type_mapping = self._create_cell_type_mapping()
        
        print(f"Loaded {self.adata.n_obs} cells with {self.adata.n_vars} genes")
        print(f"Cell types: {len(self.cell_type_mapping)}")
    
    def _load_anndata(self) -> sc.AnnData:
        """Load processed AnnData object."""
        data_file = self.data_dir / "processed_cellxgene.h5ad"
        
        if data_file.exists():
            adata = sc.read_h5ad(data_file)
        else:
            # Create dummy data for testing
            n_cells = 10000
            n_genes = self.max_genes
            
            X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
            var_names = [f"Gene_{i}" for i in range(n_genes)]
            
            adata = sc.AnnData(X=X, var=pd.DataFrame(index=var_names))
            
            # Add cell type annotations
            cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'Neutrophil'] * (n_cells // 5)
            cell_types = cell_types[:n_cells]
            adata.obs['cell_type'] = cell_types
        
        # Filter by tissue and cell types if specified
        if self.tissue_types and 'tissue' in adata.obs:
            adata = adata[adata.obs['tissue'].isin(self.tissue_types)]
        
        if self.cell_types and 'cell_type' in adata.obs:
            adata = adata[adata.obs['cell_type'].isin(self.cell_types)]
        
        # Select top variable genes
        if adata.n_vars > self.max_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=self.max_genes)
            adata = adata[:, adata.var.highly_variable]
        
        # Sample cells per cell type to balance dataset
        if self.max_cells_per_type and 'cell_type' in adata.obs:
            sampled_indices = []
            for cell_type in adata.obs['cell_type'].unique():
                mask = adata.obs['cell_type'] == cell_type
                indices = np.where(mask)[0]
                
                if len(indices) > self.max_cells_per_type:
                    indices = np.random.choice(indices, self.max_cells_per_type, replace=False)
                
                sampled_indices.extend(indices)
            
            adata = adata[sampled_indices]
        
        return adata
    
    def _create_cell_type_mapping(self) -> Dict[str, int]:
        """Create mapping from cell type names to indices."""
        if 'cell_type' in self.adata.obs:
            unique_types = self.adata.obs['cell_type'].unique()
            return {cell_type: idx for idx, cell_type in enumerate(unique_types)}
        else:
            return {'Unknown': 0}
    
    def get_embeddings_and_expressions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get embeddings, expressions, and cell type labels for retrieval module."""
        # Compute embeddings using PCA
        sc.pp.pca(self.adata, n_comps=50)
        embeddings = self.adata.obsm['X_pca']
        
        # Get expressions (use log-normalized data)
        if hasattr(self.adata, 'raw') and self.adata.raw is not None:
            expressions = self.adata.raw.X.toarray()
        else:
            expressions = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        
        # Get cell type labels
        cell_type_labels = np.array([
            self.cell_type_mapping.get(ct, 0) 
            for ct in self.adata.obs.get('cell_type', ['Unknown'] * self.adata.n_obs)
        ])
        
        return embeddings, expressions, cell_type_labels
    
    def __len__(self) -> int:
        return self.adata.n_obs
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single cell profile."""
        if hasattr(self.adata.X, 'toarray'):
            expression = self.adata.X[idx].toarray().flatten()
        else:
            expression = self.adata.X[idx]
        
        cell_type_name = self.adata.obs.iloc[idx].get('cell_type', 'Unknown')
        cell_type = self.cell_type_mapping.get(cell_type_name, 0)
        
        return {
            'expression': torch.tensor(expression, dtype=torch.float32),
            'cell_type': torch.tensor(cell_type, dtype=torch.long),
            'cell_id': str(self.adata.obs.index[idx])
        }


def create_data_loaders(config: Dict) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    
    # Create datasets
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = HESTDataset(
            data_dir=config['data_dir'],
            split=split,
            image_size=config.get('image_size', 224),
            max_genes=config.get('max_genes', 2000),
            normalize_expression=config.get('normalize_expression', True),
            cell_type_mapping=config.get('cell_type_mapping'),
            augment=(split == 'train')
        )
    
    # Create data loaders
    data_loaders = {}
    
    for split, dataset in datasets.items():
        batch_size = config.get('batch_size', 32)
        if split == 'train':
            batch_size = config.get('train_batch_size', batch_size)
        
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    return data_loaders


class DataProcessor:
    """Utility class for data preprocessing and preparation."""
    
    @staticmethod
    def prepare_cellxgene_data(raw_data_dir: str, 
                              output_dir: str,
                              tissue_types: List[str],
                              min_cells: int = 10,
                              min_genes: int = 200):
        """Process raw CellxGene data for training."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and process data
        adata_list = []
        
        for tissue in tissue_types:
            tissue_file = Path(raw_data_dir) / f"{tissue}.h5ad"
            if tissue_file.exists():
                adata = sc.read_h5ad(tissue_file)
                
                # Basic filtering
                sc.pp.filter_cells(adata, min_genes=min_genes)
                sc.pp.filter_genes(adata, min_cells=min_cells)
                
                # Normalize
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                
                adata.obs['tissue'] = tissue
                adata_list.append(adata)
        
        if adata_list:
            # Concatenate all tissues
            combined_adata = sc.concat(adata_list, join='outer')
            
            # Find highly variable genes
            sc.pp.highly_variable_genes(combined_adata, n_top_genes=2000)
            
            # Save processed data
            combined_adata.write(output_path / "processed_cellxgene.h5ad")
            
            print(f"Processed data saved to {output_path}")
            print(f"Total cells: {combined_adata.n_obs}")
            print(f"Total genes: {combined_adata.n_vars}")
        else:
            print("No data files found for specified tissues")
    
    @staticmethod
    def split_hest_data(data_dir: str, 
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1):
        """Split HEST data into train/val/test sets."""
        data_path = Path(data_dir)
        
        # Find all available samples
        image_dir = data_path / "images"
        expr_dir = data_path / "expressions"
        
        samples = []
        if image_dir.exists() and expr_dir.exists():
            for img_file in image_dir.glob("*.png"):
                sample_id = img_file.stem
                expr_file = expr_dir / f"{sample_id}.h5"
                
                if expr_file.exists():
                    samples.append({
                        'sample_id': sample_id,
                        'image_path': str(img_file),
                        'expression_path': str(expr_file)
                    })
        
        # Shuffle and split
        np.random.shuffle(samples)
        n_samples = len(samples)
        
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        splits = {
            'train': samples[:train_end],
            'val': samples[train_end:val_end],
            'test': samples[val_end:]
        }
        
        # Save split files
        for split_name, split_samples in splits.items():
            split_file = data_path / f"{split_name}_samples.json"
            with open(split_file, 'w') as f:
                json.dump(split_samples, f, indent=2)
        
        print(f"Data split completed:")
        for split_name, split_samples in splits.items():
            print(f"  {split_name}: {len(split_samples)} samples")
    
    @staticmethod
    def compute_dataset_statistics(data_dir: str) -> Dict:
        """Compute dataset statistics for normalization."""
        dataset = HESTDataset(data_dir, split='train', augment=False)
        
        # Compute image statistics
        image_means = []
        image_stds = []
        expression_stats = []
        
        data_loader = DataLoader(dataset, batch_size=32, num_workers=4)
        
        for batch in data_loader:
            images = batch['image']
            expressions = batch['expression']
            
            # Image statistics (per channel)
            for c in range(3):
                image_means.append(images[:, c].mean().item())
                image_stds.append(images[:, c].std().item())
            
            # Expression statistics
            expression_stats.extend(expressions.cpu().numpy())
        
        stats = {
            'image_mean': [np.mean(image_means[c::3]) for c in range(3)],
            'image_std': [np.mean(image_stds[c::3]) for c in range(3)],
            'expression_mean': np.mean(expression_stats),
            'expression_std': np.std(expression_stats),
            'n_samples': len(dataset)
        }
        
        return stats