"""
Two-Tower Model for Candidate Generation
==========================================
Dual-encoder architecture producing user and item embeddings
for ANN-based candidate retrieval.

Architecture:
- User Tower: user features → 256 → 128 → 128-dim L2-normalized embedding
- Item Tower: item features → 256 → 128 → 128-dim L2-normalized embedding
- Trained with: in-batch negatives + hard negative mining
- Loss: sampled softmax (equivalent to InfoNCE/NT-Xent)
- Optimizer: AdamW with cosine LR schedule

Embedding usage:
- User embedding: computed ONLINE (real-time user tower inference)
- Item embedding: computed OFFLINE (batch every 4h, stored in Milvus)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TwoTowerConfig:
    # Embedding dimensions
    user_id_vocab_size: int = 5_000_000     # hashed user IDs
    item_id_vocab_size: int = 15_000_000    # hashed item IDs
    category_vocab_size: int = 5_000
    brand_vocab_size: int = 50_000
    city_vocab_size: int = 10_000
    device_vocab_size: int = 10
    
    sparse_embedding_dim: int = 64
    final_embedding_dim: int = 128
    
    # Architecture
    user_dense_features: int = 12     # numerical user features
    item_dense_features: int = 10     # numerical item features
    hidden_dims: list[int] = None     # [256, 128]
    dropout: float = 0.1
    
    # Training
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 10
    warmup_steps: int = 1000
    temperature: float = 0.05         # softmax temperature
    hard_negative_ratio: float = 0.3  # fraction of hard negatives
    
    # Negative sampling
    num_hard_negatives: int = 3       # hard negatives per positive
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

class EmbeddingLayer(nn.Module):
    """
    Shared embedding lookup layer for sparse features.
    Uses feature hashing to handle high-cardinality features.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight[1:])  # skip padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size,) tensor of feature indices
        Returns:
            (batch_size, embedding_dim) tensor
        """
        return self.embedding(x)


class Tower(nn.Module):
    """
    One side of the Two-Tower model (user or item).
    
    Architecture:
    - Sparse features → Embedding lookup → Concatenate
    - Dense features → BatchNorm
    - [Sparse_concat || Dense_normed] → MLP → L2-normalized embedding
    """
    
    def __init__(
        self,
        sparse_specs: list[tuple[str, int]],  # [(name, vocab_size), ...]
        num_dense_features: int,
        sparse_embedding_dim: int = 64,
        hidden_dims: list[int] = None,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Sparse feature embeddings
        self.sparse_embeddings = nn.ModuleDict({
            name: EmbeddingLayer(vocab_size, sparse_embedding_dim)
            for name, vocab_size in sparse_specs
        })
        
        # Dense feature normalization
        self.dense_bn = nn.BatchNorm1d(num_dense_features) if num_dense_features > 0 else None
        
        # MLP
        input_dim = len(sparse_specs) * sparse_embedding_dim + num_dense_features
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        sparse_features: dict[str, torch.Tensor],
        dense_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            sparse_features: {feature_name: (batch_size,) index tensor}
            dense_features: (batch_size, num_dense) float tensor
        
        Returns:
            (batch_size, output_dim) L2-normalized embedding
        """
        # Embed sparse features
        sparse_embeds = []
        for name, emb_layer in self.sparse_embeddings.items():
            if name in sparse_features:
                sparse_embeds.append(emb_layer(sparse_features[name]))
        
        parts = []
        if sparse_embeds:
            parts.append(torch.cat(sparse_embeds, dim=1))
        
        if dense_features is not None and self.dense_bn is not None:
            normed_dense = self.dense_bn(dense_features)
            parts.append(normed_dense)
        
        x = torch.cat(parts, dim=1)
        x = self.mlp(x)
        
        # L2 normalize for cosine similarity / inner product
        x = F.normalize(x, p=2, dim=1)
        
        return x


# ---------------------------------------------------------------------------
# Two-Tower Model
# ---------------------------------------------------------------------------

class TwoTowerModel(pl.LightningModule):
    """
    Two-Tower (Dual Encoder) model for candidate generation.
    
    Training:
    - In-batch negatives: other items in same batch serve as negatives (free!)
    - Hard negatives: items retrieved by ANN but not interacted with
    - Loss: sampled softmax with temperature scaling
    
    Inference:
    - User tower: runs ONLINE at serving time
    - Item tower: runs OFFLINE, embeddings stored in Milvus
    """
    
    def __init__(self, config: TwoTowerConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # User tower
        self.user_tower = Tower(
            sparse_specs=[
                ("user_id", config.user_id_vocab_size),
                ("city", config.city_vocab_size),
                ("device", config.device_vocab_size),
            ],
            num_dense_features=config.user_dense_features,
            sparse_embedding_dim=config.sparse_embedding_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.final_embedding_dim,
            dropout=config.dropout,
        )
        
        # Item tower
        self.item_tower = Tower(
            sparse_specs=[
                ("item_id", config.item_id_vocab_size),
                ("category", config.category_vocab_size),
                ("brand", config.brand_vocab_size),
            ],
            num_dense_features=config.item_dense_features,
            sparse_embedding_dim=config.sparse_embedding_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.final_embedding_dim,
            dropout=config.dropout,
        )
        
        self.temperature = config.temperature
    
    def forward(
        self,
        user_sparse: dict[str, torch.Tensor],
        user_dense: torch.Tensor,
        item_sparse: dict[str, torch.Tensor],
        item_dense: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute user and item embeddings.
        
        Returns:
            (user_embedding, item_embedding) each (batch_size, embedding_dim)
        """
        user_emb = self.user_tower(user_sparse, user_dense)
        item_emb = self.item_tower(item_sparse, item_dense)
        return user_emb, item_emb
    
    def compute_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        hard_neg_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sampled softmax loss with in-batch negatives.
        
        For each (user, positive_item) pair:
        - All other items in the batch are negatives
        - Optional: hard negatives appended
        
        Equivalent to InfoNCE / NT-Xent loss.
        """
        batch_size = user_emb.shape[0]
        
        # Similarity matrix: (batch_size, batch_size)
        # Each row: similarity of user_i with all items
        logits = torch.mm(user_emb, item_emb.t()) / self.temperature
        
        if hard_neg_emb is not None:
            # Append hard negative similarities
            hard_logits = torch.mm(user_emb, hard_neg_emb.t()) / self.temperature
            logits = torch.cat([logits, hard_logits], dim=1)
        
        # Labels: diagonal entries are positives
        labels = torch.arange(batch_size, device=logits.device)
        
        # Cross-entropy loss (softmax over all negatives)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        user_emb, item_emb = self.forward(
            batch["user_sparse"], batch["user_dense"],
            batch["item_sparse"], batch["item_dense"],
        )
        
        hard_neg_emb = None
        if "hard_neg_sparse" in batch:
            hard_neg_emb = self.item_tower(
                batch["hard_neg_sparse"], batch["hard_neg_dense"]
            )
        
        loss = self.compute_loss(user_emb, item_emb, hard_neg_emb)
        
        # Metrics
        with torch.no_grad():
            sim = torch.mm(user_emb, item_emb.t())
            correct = (sim.argmax(dim=1) == torch.arange(len(sim), device=sim.device)).float()
            accuracy = correct.mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int):
        user_emb, item_emb = self.forward(
            batch["user_sparse"], batch["user_dense"],
            batch["item_sparse"], batch["item_dense"],
        )
        
        loss = self.compute_loss(user_emb, item_emb)
        
        # Recall@K metrics
        with torch.no_grad():
            sim = torch.mm(user_emb, item_emb.t())
            labels = torch.arange(len(sim), device=sim.device)
            
            for k in [10, 50, 100]:
                topk = sim.topk(min(k, sim.shape[1]), dim=1).indices
                recall = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
                self.log(f"val_recall@{k}", recall, prog_bar=(k == 100))
        
        self.log("val_loss", loss, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.warmup_steps,
            T_mult=2,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def get_user_embedding(
        self,
        user_sparse: dict[str, torch.Tensor],
        user_dense: torch.Tensor,
    ) -> np.ndarray:
        """
        Get user embedding for online serving.
        
        This method runs the user tower inference ONLY,
        which produces the query vector for ANN search.
        """
        self.eval()
        with torch.no_grad():
            emb = self.user_tower(user_sparse, user_dense)
        return emb.cpu().numpy()
    
    def get_item_embeddings_batch(
        self,
        item_sparse: dict[str, torch.Tensor],
        item_dense: torch.Tensor,
    ) -> np.ndarray:
        """
        Get item embeddings for batch indexing.
        
        Used in offline pipeline to compute embeddings for all 10M items,
        which are then loaded into Milvus ANN index.
        """
        self.eval()
        with torch.no_grad():
            emb = self.item_tower(item_sparse, item_dense)
        return emb.cpu().numpy()


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    """
    Dataset of (user, item) interaction pairs with negative sampling.
    
    Data format (from logged serving features):
    - user_sparse_features: dict of feature indices
    - user_dense_features: array of numerical features
    - item_sparse_features: dict of feature indices
    - item_dense_features: array of numerical features
    - label: 1 (positive interaction) / 0 (negative)
    
    Negative sampling strategy:
    - In-batch negatives are computed in the loss function (free)
    - Hard negatives are pre-mined from serving logs (items retrieved but not clicked)
    """
    
    def __init__(
        self,
        interaction_data: list[dict[str, Any]],
        hard_negatives: dict[str, list[dict]] | None = None,
    ):
        self.data = interaction_data
        self.hard_negatives = hard_negatives or {}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        result = {
            "user_sparse": {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in sample["user_sparse"].items()
            },
            "user_dense": torch.tensor(
                sample["user_dense"], dtype=torch.float32
            ),
            "item_sparse": {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in sample["item_sparse"].items()
            },
            "item_dense": torch.tensor(
                sample["item_dense"], dtype=torch.float32
            ),
        }
        
        # Add hard negatives if available
        user_id = str(sample.get("user_id", ""))
        if user_id in self.hard_negatives:
            neg = self.hard_negatives[user_id]
            if neg:
                import random
                neg_sample = random.choice(neg)
                result["hard_neg_sparse"] = {
                    k: torch.tensor(v, dtype=torch.long)
                    for k, v in neg_sample["item_sparse"].items()
                }
                result["hard_neg_dense"] = torch.tensor(
                    neg_sample["item_dense"], dtype=torch.float32
                )
        
        return result


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def train_two_tower(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    config: TwoTowerConfig | None = None,
):
    """
    Train Two-Tower model.
    
    Pipeline:
    1. Load training data (logged features from serving)
    2. Train with in-batch + hard negatives
    3. Evaluate recall@K on validation set
    4. Export model checkpoint + item embeddings
    """
    config = config or TwoTowerConfig()
    
    # In production: load from Parquet/Delta Lake
    # train_data = load_interaction_data(train_data_path)
    # val_data = load_interaction_data(val_data_path)
    
    # Placeholder data for skeleton
    print(f"Training Two-Tower model")
    print(f"Config: {config}")
    print(f"Train data: {train_data_path}")
    print(f"Output: {output_dir}")
    
    model = TwoTowerModel(config)
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",  # mixed precision for faster training
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        default_root_dir=output_dir,
    )
    
    # trainer.fit(model, train_dataloader, val_dataloader)
    
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Two-Tower model")
    parser.add_argument("--train-data", required=True, help="Path to training data")
    parser.add_argument("--val-data", required=True, help="Path to validation data")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    config = TwoTowerConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
    )
    
    train_two_tower(args.train_data, args.val_data, args.output_dir, config)
