# scripts/download_hest.py

import os
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile

def download_file(url: str, filepath: str, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {Path(filepath).name}") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_archive(archive_path: str, extract_to: str):
    """Extract archive file."""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    
    print(f"Extracting {archive_path.name} to {extract_to}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def download_hest_dataset(output_dir: str, sample_subset: str = None):
    """
    Download HEST-1K dataset.
    
    Args:
        output_dir: Directory to save the dataset
        sample_subset: Subset of samples to download ('small', 'medium', 'full')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define download URLs (these would be actual URLs in practice)
    # For demo purposes, using placeholder URLs
    base_urls = {
        'small': 'https://example.com/hest-small.tar.gz',
        'medium': 'https://example.com/hest-medium.tar.gz', 
        'full': 'https://example.com/hest-1k-full.tar.gz'
    }
    
    subset = sample_subset or 'small'
    
    if subset not in base_urls:
        raise ValueError(f"Unknown subset: {subset}. Choose from {list(base_urls.keys())}")
    
    url = base_urls[subset]
    filename = f"hest_{subset}.tar.gz"
    filepath = output_dir / filename
    
    print(f"Downloading HEST-1K dataset ({subset} subset)")
    print(f"URL: {url}")
    print(f"Output: {filepath}")
    
    # Download the dataset
    try:
        download_file(url, str(filepath))
        print(f"Download completed: {filepath}")
        
        # Extract the dataset
        extract_archive(str(filepath), str(output_dir))
        print(f"Extraction completed to {output_dir}")
        
        # Clean up archive file
        os.remove(filepath)
        print(f"Cleaned up archive file: {filename}")
        
        # Verify the structure
        verify_hest_structure(output_dir)
        
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        
        # For demo purposes, create dummy data structure
        print("Creating dummy data structure for development...")
        create_dummy_hest_data(output_dir, subset)


def verify_hest_structure(data_dir: Path):
    """Verify the HEST dataset structure."""
    expected_dirs = ['images', 'expressions', 'metadata']
    
    for dirname in expected_dirs:
        dir_path = data_dir / dirname
        if dir_path.exists():
            print(f"✓ Found {dirname} directory")
            
            # Count files
            if dirname == 'images':
                count = len(list(dir_path.glob('*.png')))
                print(f"  - {count} image files")
            elif dirname == 'expressions':
                count = len(list(dir_path.glob('*.h5')))
                print(f"  - {count} expression files")
        else:
            print(f"✗ Missing {dirname} directory")


def create_dummy_hest_data(output_dir: Path, subset: str):
    """Create dummy HEST data for development and testing."""
    import numpy as np
    import h5py
    from PIL import Image
    
    # Define number of samples based on subset
    n_samples = {'small': 100, 'medium': 500, 'full': 1000}[subset]
    
    print(f"Creating {n_samples} dummy samples...")
    
    # Create directories
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'expressions').mkdir(exist_ok=True)
    (output_dir / 'metadata').mkdir(exist_ok=True)
    
    # Generate dummy data
    gene_names = [f"Gene_{i:04d}" for i in range(2000)]
    cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'Neutrophil', 
                  'Epithelial', 'Endothelial', 'Fibroblast', 'Smooth_muscle']
    
    metadata_records = []
    
    for i in tqdm(range(n_samples), desc="Creating dummy samples"):
        sample_id = f"sample_{i:06d}"
        
        # Create dummy histology image (224x224 RGB)
        # Simulate tissue-like appearance
        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add some structure to make it look more tissue-like
        center_x, center_y = np.random.randint(50, 174, 2)
        y, x = np.ogrid[:224, :224]
        mask = ((x - center_x)**2 + (y - center_y)**2) < 30**2
        img_array[mask] = img_array[mask] * 0.7 + 50  # Darker center regions
        
        img = Image.fromarray(img_array)
        img.save(output_dir / 'images' / f"{sample_id}.png")
        
        # Create dummy gene expression
        # Simulate realistic expression patterns
        base_expression = np.random.exponential(1.0, 2000)
        
        # Add some highly expressed genes
        n_high_expr = np.random.randint(50, 200)
        high_expr_indices = np.random.choice(2000, n_high_expr, replace=False)
        base_expression[high_expr_indices] *= np.random.uniform(5, 20, n_high_expr)
        
        # Add noise
        expression = base_expression + np.random.normal(0, 0.1, 2000)
        expression = np.maximum(expression, 0)  # Non-negative
        
        # Save expression data
        with h5py.File(output_dir / 'expressions' / f"{sample_id}.h5", 'w') as f:
            f.create_dataset('expression', data=expression.astype(np.float32))
            f.create_dataset('gene_names', 
                           data=[name.encode('utf-8') for name in gene_names])
            
            # Add cell type annotation
            cell_type = np.random.choice(cell_types)
            f.create_dataset('cell_type', data=cell_type.encode('utf-8'))
        
        # Metadata record
        metadata_records.append({
            'sample_id': sample_id,
            'cell_type': cell_type,
            'tissue': np.random.choice(['liver', 'lung', 'brain', 'heart']),
            'patient_id': f"patient_{i // 10:03d}",
            'x_coord': np.random.uniform(0, 1000),
            'y_coord': np.random.uniform(0, 1000)
        })
    
    # Save metadata
    import pandas as pd
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"Created dummy HEST dataset with {n_samples} samples")
    print(f"Data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download HEST dataset')
    parser.add_argument('--output_dir', type=str, default='data/hest',
                       help='Output directory for dataset')
    parser.add_argument('--subset', type=str, choices=['small', 'medium', 'full'],
                       default='small', help='Dataset subset to download')
    parser.add_argument('--create_dummy', action='store_true',
                       help='Create dummy data instead of downloading')
    
    args = parser.parse_args()
    
    if args.create_dummy:
        output_dir = Path(args.output_dir)
        create_dummy_hest_data(output_dir, args.subset)
    else:
        download_hest_dataset(args.output_dir, args.subset)


if __name__ == '__main__':
    main()


# scripts/prepare_cellxgene.py

import argparse
import pandas as pd
import scanpy as sc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cellxgene_census

def download_cellxgene_data(tissue_types: list, 
                           output_dir: str,
                           max_cells_per_tissue: int = 10000):
    """
    Download and process CellxGene Census data for specified tissues.
    
    Args:
        tissue_types: List of tissue types to download
        output_dir: Output directory
        max_cells_per_tissue: Maximum cells per tissue type
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Connecting to CellxGene Census...")
    
    try:
        # Connect to CellxGene Census
        census = cellxgene_census.open_soma()
        
        all_adata = []
        
        for tissue in tissue_types:
            print(f"Downloading data for {tissue}...")
            
            # Query for tissue-specific data
            # This is a simplified example - actual implementation would use proper Census API
            query = f"tissue_general == '{tissue}'"
            
            # Get expression data
            obs_df = census["census_data"]["homo_sapiens"].obs.read(
                value_filter=query,
                column_names=["soma_joinid", "cell_type", "tissue_general"]
            ).concat().to_pandas()
            
            if len(obs_df) == 0:
                print(f"No data found for {tissue}")
                continue
            
            # Sample cells if too many
            if len(obs_df) > max_cells_per_tissue:
                obs_df = obs_df.sample(n=max_cells_per_tissue, random_state=42)
            
            # Get expression matrix
            var_df = census["census_data"]["homo_sapiens"].var.read().concat().to_pandas()
            
            # Select highly variable genes
            selected_genes = var_df.head(2000)  # Top 2000 genes
            
            X = census["census_data"]["homo_sapiens"].X["raw"].read(
                obs_joinids=obs_df["soma_joinid"].values,
                var_joinids=selected_genes["soma_joinid"].values
            ).coos().concat().to_scipy_csr()
            
            # Create AnnData object
            adata = sc.AnnData(
                X=X,
                obs=obs_df.set_index("soma_joinid"),
                var=selected_genes.set_index("soma_joinid")
            )
            
            adata.obs['tissue'] = tissue
            all_adata.append(adata)
            
            print(f"Downloaded {adata.n_obs} cells for {tissue}")
        
        census.close()
        
    except Exception as e:
        print(f"Error accessing CellxGene Census: {e}")
        print("Creating dummy CellxGene data for development...")
        all_adata = create_dummy_cellxgene_data(tissue_types, max_cells_per_tissue)
    
    if all_adata:
        # Combine all tissues
        print("Combining and processing data...")
        combined_adata = sc.concat(all_adata, join='outer')
        
        # Basic preprocessing
        sc.pp.filter_cells(combined_adata, min_genes=200)
        sc.pp.filter_genes(combined_adata, min_cells=10)
        
        # Normalize and log transform
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(combined_adata, n_top_genes=2000)
        combined_adata.raw = combined_adata
        combined_adata = combined_adata[:, combined_adata.var.highly_variable]
        
        # Compute PCA for embeddings
        sc.pp.pca(combined_adata, n_comps=50)
        
        # Save processed data
        output_file = output_dir / "processed_cellxgene.h5ad"
        combined_adata.write(output_file)
        
        print(f"Processed CellxGene data saved to {output_file}")
        print(f"Final dataset: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
        
        # Save cell type mapping
        cell_types = combined_adata.obs['cell_type'].unique()
        cell_type_mapping = {ct: i for i, ct in enumerate(cell_types)}
        
        import json
        with open(output_dir / "cell_type_mapping.json", 'w') as f:
            json.dump(cell_type_mapping, f, indent=2)
        
        print(f"Cell type mapping saved with {len(cell_types)} types")


def create_dummy_cellxgene_data(tissue_types: list, max_cells_per_tissue: int):
    """Create dummy CellxGene data for development."""
    
    cell_types = [
        'T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'Neutrophil',
        'Epithelial_cell', 'Endothelial_cell', 'Fibroblast', 
        'Smooth_muscle_cell', 'Macrophage'
    ]
    
    all_adata = []
    
    for tissue in tissue_types:
        print(f"Creating dummy data for {tissue}...")
        
        n_cells = min(max_cells_per_tissue, np.random.randint(5000, 10000))
        n_genes = 2000
        
        # Generate realistic gene expression using negative binomial
        X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
        
        # Create gene names
        var_names = [f"Gene_{i:04d}" for i in range(n_genes)]
        
        # Create cell annotations
        obs_data = {
            'cell_type': np.random.choice(cell_types, n_cells),
            'tissue_general': tissue,
            'donor_id': [f"donor_{i//100:03d}" for i in range(n_cells)],
            'sex': np.random.choice(['male', 'female'], n_cells),
            'age': np.random.randint(20, 80, n_cells)
        }
        
        # Create AnnData
        adata = sc.AnnData(
            X=X,
            obs=pd.DataFrame(obs_data),
            var=pd.DataFrame(index=var_names)
        )
        
        all_adata.append(adata)
        
        print(f"Created {n_cells} dummy cells for {tissue}")
    
    return all_adata


def main():
    parser = argparse.ArgumentParser(description='Prepare CellxGene Census data')
    parser.add_argument('--tissue_types', nargs='+', 
                       default=['liver', 'lung', 'brain', 'heart'],
                       help='Tissue types to download')
    parser.add_argument('--output_dir', type=str, default='data/cellxgene',
                       help='Output directory')
    parser.add_argument('--max_cells_per_tissue', type=int, default=10000,
                       help='Maximum cells per tissue type')
    parser.add_argument('--create_dummy', action='store_true',
                       help='Create dummy data instead of downloading')
    
    args = parser.parse_args()
    
    if args.create_dummy:
        print("Creating dummy CellxGene data...")
        all_adata = create_dummy_cellxgene_data(args.tissue_types, args.max_cells_per_tissue)
        
        # Process and save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if all_adata:
            combined_adata = sc.concat(all_adata, join='outer')
            
            # Basic processing
            sc.pp.normalize_total(combined_adata, target_sum=1e4)
            sc.pp.log1p(combined_adata)
            sc.pp.highly_variable_genes(combined_adata, n_top_genes=2000)
            sc.pp.pca(combined_adata, n_comps=50)
            
            # Save
            combined_adata.write(output_dir / "processed_cellxgene.h5ad")
            
            # Save cell type mapping
            cell_types = combined_adata.obs['cell_type'].unique()
            cell_type_mapping = {ct: i for i, ct in enumerate(cell_types)}
            
            import json
            with open(output_dir / "cell_type_mapping.json", 'w') as f:
                json.dump(cell_type_mapping, f, indent=2)
            
            print(f"Dummy data created: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
    else:
        download_cellxgene_data(args.tissue_types, args.output_dir, args.max_cells_per_tissue)


if __name__ == '__main__':
    main()


# scripts/split_data.py

import argparse
import json
import numpy as np
from pathlib import Path
from ragst.data.hest_dataset import DataProcessor

def main():
    parser = argparse.ArgumentParser(description='Split HEST data into train/val/test')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='HEST data directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Split data
    DataProcessor.split_hest_data(
        args.data_dir,
        args.train_ratio,
        args.val_ratio
    )
    
    print("Data splitting completed!")


if __name__ == '__main__':
    main()


# scripts/compute_stats.py

import argparse
import json
from ragst.data.hest_dataset import DataProcessor

def main():
    parser = argparse.ArgumentParser(description='Compute dataset statistics')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='HEST data directory')
    parser.add_argument('--output', type=str, default='data_statistics.json',
                       help='Output file for statistics')
    
    args = parser.parse_args()
    
    print("Computing dataset statistics...")
    stats = DataProcessor.compute_dataset_statistics(args.data_dir)
    
    # Save statistics
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {args.output}")
    print(f"Dataset contains {stats['n_samples']} samples")
    print(f"Image statistics: mean={stats['image_mean']}, std={stats['image_std']}")
    print(f"Expression statistics: mean={stats['expression_mean']:.4f}, std={stats['expression_std']:.4f}")


if __name__ == '__main__':
    main()