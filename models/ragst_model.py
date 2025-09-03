# ragst/models/ragst_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import ViTModel, ViTConfig
import faiss

class VisionEncoder(nn.Module):
    """Vision encoder for histology image patches."""
    
    def __init__(self, 
                 backbone: str = "vit-base", 
                 pretrained: bool = True,
                 freeze_layers: int = 0,
                 output_dim: int = 768):
        super().__init__()
        
        self.backbone = backbone
        self.output_dim = output_dim
        
        if backbone == "vit-base":
            config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
            self.encoder = ViTModel(config)
            encoder_dim = config.hidden_size
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Projection layer to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Freeze early layers if specified
        if freeze_layers > 0:
            for param in list(self.encoder.parameters())[:freeze_layers]:
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Batch of histology images [B, C, H, W]
            
        Returns:
            Image embeddings [B, output_dim]
        """
        # Extract features using backbone
        outputs = self.encoder(images)
        
        # Use CLS token for global representation
        features = outputs.last_hidden_state[:, 0]  # [B, encoder_dim]
        
        # Project to desired dimension
        embeddings = self.projection(features)  # [B, output_dim]
        
        return embeddings


class CellTypeClassifier(nn.Module):
    """Multi-class classifier for cell type prediction from image embeddings."""
    
    def __init__(self, 
                 input_dim: int = 768,
                 num_cell_types: int = 100,
                 hidden_dims: List[int] = [512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_cell_types))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict cell type probabilities.
        
        Args:
            embeddings: Image embeddings [B, input_dim]
            
        Returns:
            Cell type logits [B, num_cell_types]
        """
        logits = self.classifier(embeddings)
        return logits


class RetrievalModule(nn.Module):
    """FAISS-based retrieval module for scRNA-seq profiles."""
    
    def __init__(self, 
                 scrna_embeddings: np.ndarray,
                 scrna_expressions: np.ndarray,
                 cell_type_labels: np.ndarray,
                 embedding_dim: int = 768,
                 top_k: int = 10):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.scrna_expressions = scrna_expressions
        self.cell_type_labels = cell_type_labels
        
        # Build FAISS index for efficient retrieval
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = scrna_embeddings / np.linalg.norm(
            scrna_embeddings, axis=1, keepdims=True
        )
        self.index.add(normalized_embeddings.astype(np.float32))
    
    def retrieve(self, 
                 cell_type_probs: torch.Tensor,
                 query_embeddings: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k scRNA-seq profiles based on cell type probabilities.
        
        Args:
            cell_type_probs: Cell type probability distributions [B, num_cell_types]
            query_embeddings: Optional query embeddings for similarity search
            
        Returns:
            retrieved_expressions: Retrieved gene expressions [B, top_k, num_genes]
            retrieval_weights: Weights for retrieved samples [B, top_k]
        """
        batch_size = cell_type_probs.size(0)
        num_genes = self.scrna_expressions.shape[1]
        
        retrieved_expressions = []
        retrieval_weights = []
        
        for i in range(batch_size):
            probs = cell_type_probs[i].cpu().numpy()
            
            # Sample cell types based on probabilities
            top_cell_types = np.argsort(probs)[-5:]  # Top 5 cell types
            
            # Collect candidates from top cell types
            candidates = []
            for cell_type in top_cell_types:
                mask = self.cell_type_labels == cell_type
                if mask.sum() > 0:
                    indices = np.where(mask)[0]
                    candidates.extend(indices[:min(self.top_k * 2, len(indices))])
            
            if len(candidates) < self.top_k:
                # Fallback: use all available candidates
                candidates = list(range(min(self.top_k, len(self.cell_type_labels))))
            
            # Select top-k from candidates
            selected_indices = np.random.choice(candidates, 
                                              size=min(self.top_k, len(candidates)), 
                                              replace=False)
            
            # Get expressions and compute weights
            expressions = self.scrna_expressions[selected_indices]
            weights = np.ones(len(selected_indices)) / len(selected_indices)
            
            # Pad if necessary
            if len(selected_indices) < self.top_k:
                pad_size = self.top_k - len(selected_indices)
                expressions = np.vstack([expressions, np.zeros((pad_size, num_genes))])
                weights = np.concatenate([weights, np.zeros(pad_size)])
            
            retrieved_expressions.append(expressions)
            retrieval_weights.append(weights)
        
        retrieved_expressions = torch.tensor(np.stack(retrieved_expressions), dtype=torch.float32)
        retrieval_weights = torch.tensor(np.stack(retrieval_weights), dtype=torch.float32)
        
        return retrieved_expressions, retrieval_weights


class GeneratorModule(nn.Module):
    """Transformer-based generator for gene expression prediction."""
    
    def __init__(self,
                 image_dim: int = 768,
                 scrna_dim: int = 2000,  # Number of genes
                 output_dim: int = 2000,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.image_dim = image_dim
        self.scrna_dim = scrna_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.scrna_proj = nn.Linear(scrna_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Cross-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, 
                image_embeddings: torch.Tensor,
                retrieved_expressions: torch.Tensor,
                retrieval_weights: torch.Tensor) -> torch.Tensor:
        """
        Generate gene expression from image and retrieved scRNA-seq profiles.
        
        Args:
            image_embeddings: Image embeddings [B, image_dim]
            retrieved_expressions: Retrieved gene expressions [B, top_k, scrna_dim]
            retrieval_weights: Weights for retrieved samples [B, top_k]
            
        Returns:
            Generated gene expressions [B, output_dim]
        """
        batch_size, top_k, _ = retrieved_expressions.shape
        
        # Project inputs to hidden dimension
        image_features = self.image_proj(image_embeddings).unsqueeze(1)  # [B, 1, hidden_dim]
        scrna_features = self.scrna_proj(retrieved_expressions.view(-1, self.scrna_dim))  # [B*top_k, hidden_dim]
        scrna_features = scrna_features.view(batch_size, top_k, self.hidden_dim)  # [B, top_k, hidden_dim]
        
        # Combine image and scRNA features
        combined_features = torch.cat([image_features, scrna_features], dim=1)  # [B, 1+top_k, hidden_dim]
        
        # Add positional embeddings
        seq_len = combined_features.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        combined_features = combined_features + pos_emb
        
        # Apply transformer
        transformed_features = self.transformer(combined_features)  # [B, 1+top_k, hidden_dim]
        
        # Use first token (image token) for final prediction
        output_features = transformed_features[:, 0, :]  # [B, hidden_dim]
        
        # Generate final gene expression
        gene_expression = self.output_proj(output_features)  # [B, output_dim]
        
        return gene_expression


class RAGSTModel(nn.Module):
    """Complete RAG-ST model combining all components."""
    
    def __init__(self,
                 vision_config: Dict,
                 classifier_config: Dict,
                 generator_config: Dict,
                 retrieval_data: Optional[Dict] = None):
        super().__init__()
        
        # Initialize components
        self.vision_encoder = VisionEncoder(**vision_config)
        self.cell_type_classifier = CellTypeClassifier(**classifier_config)
        self.generator = GeneratorModule(**generator_config)
        
        # Initialize retrieval module if data provided
        self.retrieval_module = None
        if retrieval_data is not None:
            self.retrieval_module = RetrievalModule(**retrieval_data)
    
    def forward(self, 
                images: torch.Tensor,
                return_intermediate: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Complete forward pass through RAG-ST model.
        
        Args:
            images: Batch of histology images [B, C, H, W]
            return_intermediate: Whether to return intermediate outputs
            
        Returns:
            Gene expression predictions [B, num_genes] or dict with intermediate outputs
        """
        # Stage 1: Extract image features and predict cell types
        image_embeddings = self.vision_encoder(images)
        cell_type_logits = self.cell_type_classifier(image_embeddings)
        cell_type_probs = F.softmax(cell_type_logits, dim=-1)
        
        # Stage 2: Retrieve and generate
        if self.retrieval_module is not None:
            retrieved_expressions, retrieval_weights = self.retrieval_module.retrieve(cell_type_probs)
            gene_expression = self.generator(image_embeddings, retrieved_expressions, retrieval_weights)
        else:
            # Fallback: direct prediction without retrieval
            gene_expression = self.generator.output_proj(image_embeddings)
        
        if return_intermediate:
            return {
                'gene_expression': gene_expression,
                'image_embeddings': image_embeddings,
                'cell_type_probs': cell_type_probs,
                'cell_type_logits': cell_type_logits
            }
        
        return gene_expression
    
    @classmethod
    def load_pretrained(cls, checkpoint_path: str) -> 'RAGSTModel':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract configurations
        model_config = checkpoint['model_config']
        model = cls(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def save_checkpoint(self, 
                       checkpoint_path: str, 
                       epoch: int, 
                       optimizer_state: Optional[Dict] = None,
                       metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vision_config': self.vision_encoder.__dict__,
                'classifier_config': self.cell_type_classifier.__dict__,
                'generator_config': self.generator.__dict__
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)