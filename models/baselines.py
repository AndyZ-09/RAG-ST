# ragst/models/baselines.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np
from transformers import ViTModel, ViTConfig


class DirectRegressionBaseline(nn.Module):
    """
    Simple baseline: directly regress from image features to gene expression.
    This represents the standard encoder-decoder approach mentioned in the paper.
    """
    
    def __init__(self,
                 backbone: str = "vit-base",
                 num_genes: int = 2000,
                 hidden_dims: List[int] = [1024, 512, 256]):
        super().__init__()
        
        # Vision encoder
        if backbone == "vit-base":
            config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
            self.vision_encoder = ViTModel(config)
            encoder_dim = config.hidden_size
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Direct regression layers
        layers = []
        prev_dim = encoder_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer for gene expression
        layers.append(nn.Linear(prev_dim, num_genes))
        
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input histology images [B, C, H, W]
            
        Returns:
            Predicted gene expressions [B, num_genes]
        """
        # Extract image features
        vision_outputs = self.vision_encoder(images)
        image_features = vision_outputs.last_hidden_state[:, 0]  # CLS token
        
        # Direct regression to gene expression
        gene_expression = self.regressor(image_features)
        
        return gene_expression


class CNNBaseline(nn.Module):
    """
    CNN-based baseline model using ResNet architecture.
    """
    
    def __init__(self,
                 num_genes: int = 2000,
                 pretrained: bool = True):
        super().__init__()
        
        # Use ResNet50 as backbone
        import torchvision.models as models
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        backbone_dim = 2048
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_genes)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN baseline."""
        features = self.backbone(images)
        gene_expression = self.regressor(features)
        return gene_expression


class AttentionPoolingBaseline(nn.Module):
    """
    Baseline using attention pooling over image patches.
    """
    
    def __init__(self,
                 backbone: str = "vit-base",
                 num_genes: int = 2000,
                 attention_dim: int = 256):
        super().__init__()
        
        # Vision encoder
        if backbone == "vit-base":
            config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
            self.vision_encoder = ViTModel(config)
            encoder_dim = config.hidden_size
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable query for pooling
        self.pooling_query = nn.Parameter(torch.randn(1, 1, encoder_dim))
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_genes)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention pooling."""
        batch_size = images.size(0)
        
        # Extract patch features
        vision_outputs = self.vision_encoder(images)
        patch_features = vision_outputs.last_hidden_state  # [B, num_patches, dim]
        
        # Attention pooling
        query = self.pooling_query.expand(batch_size, -1, -1)  # [B, 1, dim]
        pooled_features, _ = self.attention_pooling(query, patch_features, patch_features)
        pooled_features = pooled_features.squeeze(1)  # [B, dim]
        
        # Predict gene expression
        gene_expression = self.regressor(pooled_features)
        
        return gene_expression


class MultiScaleBaseline(nn.Module):
    """
    Multi-scale baseline processing images at different resolutions.
    """
    
    def __init__(self,
                 num_genes: int = 2000,
                 scales: List[int] = [224, 112, 56]):
        super().__init__()
        
        self.scales = scales
        
        # Separate encoders for different scales
        self.encoders = nn.ModuleList()
        encoder_dims = []
        
        for scale in scales:
            # Simple CNN encoder for each scale
            encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 5, 2, 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 256)
            )
            self.encoders.append(encoder)
            encoder_dims.append(256)
        
        # Fusion layer
        total_dim = sum(encoder_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_genes)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing."""
        features = []
        
        for encoder, scale in zip(self.encoders, self.scales):
            # Resize image to target scale
            resized = F.interpolate(images, size=(scale, scale), 
                                  mode='bilinear', align_corners=False)
            
            # Extract features
            scale_features = encoder(resized)
            features.append(scale_features)
        
        # Concatenate multi-scale features
        combined_features = torch.cat(features, dim=1)
        
        # Predict gene expression
        gene_expression = self.fusion(combined_features)
        
        return gene_expression


class EnsembleBaseline(nn.Module):
    """
    Ensemble baseline combining multiple architectures.
    """
    
    def __init__(self,
                 num_genes: int = 2000,
                 num_models: int = 3):
        super().__init__()
        
        # Create multiple base models
        self.models = nn.ModuleList([
            DirectRegressionBaseline(num_genes=num_genes),
            CNNBaseline(num_genes=num_genes),
            AttentionPoolingBaseline(num_genes=num_genes)
        ][:num_models])
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        predictions = []
        
        for model in self.models:
            pred = model(images)
            predictions.append(pred)
        
        # Weighted combination
        stacked_preds = torch.stack(predictions, dim=0)  # [num_models, B, num_genes]
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        
        ensemble_pred = (stacked_preds * weights).sum(dim=0)
        
        return ensemble_pred


class LinearBaseline(nn.Module):
    """
    Simple linear baseline for quick comparison.
    """
    
    def __init__(self, num_genes: int = 2000):
        super().__init__()
        
        # Global average pooling on raw images
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Single linear layer
        self.linear = nn.Linear(3, num_genes)  # 3 RGB channels
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear baseline."""
        # Global average pooling
        pooled = self.global_pool(images).flatten(1)  # [B, 3]
        
        # Linear prediction
        gene_expression = self.linear(pooled)
        
        return gene_expression


def create_baseline_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to create baseline models."""
    
    model_registry = {
        'direct_regression': DirectRegressionBaseline,
        'cnn': CNNBaseline,
        'attention_pooling': AttentionPoolingBaseline,
        'multiscale': MultiScaleBaseline,
        'ensemble': EnsembleBaseline,
        'linear': LinearBaseline
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Unknown baseline model: {model_name}. "
                        f"Available models: {list(model_registry.keys())}")
    
    model_class = model_registry[model_name]
    
    # Filter kwargs to only include relevant parameters
    import inspect
    sig = inspect.signature(model_class.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    return model_class(**valid_kwargs)


# Example usage and training script for baselines
if __name__ == "__main__":
    # Demo of baseline models
    batch_size = 4
    num_genes = 2000
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    models_to_test = [
        'direct_regression',
        'cnn', 
        'attention_pooling',
        'multiscale',
        'linear'
    ]
    
    print("Testing baseline models:")
    
    for model_name in models_to_test:
        print(f"\n{model_name}:")
        
        try:
            model = create_baseline_model(model_name, num_genes=num_genes)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params:,}")
            
            # Test forward pass
            with torch.no_grad():
                output = model(dummy_images)
                print(f"  Output shape: {output.shape}")
                print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nAll baseline models tested successfully!")