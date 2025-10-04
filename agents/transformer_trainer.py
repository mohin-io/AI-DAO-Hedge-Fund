"""
Fine-tuned Transformer Model Trainer
Advanced training pipeline with real market data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime
import json

from agents.transformer_predictor import TransformerPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    """PyTorch Dataset for market sequences"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerTrainer:
    """
    Advanced Transformer training pipeline with:
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Model checkpointing
    - Performance metrics
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = TransformerPredictor(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion_reg = nn.MSELoss()
        self.criterion_cls = nn.BCEWithLogitsLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def prepare_data_loaders(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int = 32,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders

        Args:
            data: Dictionary with 'X_train', 'X_test', 'y_train', 'y_test'
            batch_size: Batch size for training
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, test_loader)
        """
        train_dataset = MarketDataset(data['X_train'], data['y_train'])
        test_dataset = MarketDataset(data['X_test'], data['y_test'])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )

        logger.info(f"Created DataLoaders: {len(train_dataset)} train, {len(test_dataset)} test samples")

        return train_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            cls_logits, reg_output = self.model(X)

            # Convert regression targets to classification labels
            cls_labels = (y > 0).float().unsqueeze(1)

            # Combined loss
            cls_loss = self.criterion_cls(cls_logits, cls_labels)
            reg_loss = self.criterion_reg(reg_output.squeeze(), y)
            loss = 0.5 * cls_loss + 0.5 * reg_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(cls_logits) > 0.5).float()
            correct_predictions += (predictions == cls_labels).sum().item()
            total_samples += len(y)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                # Forward pass
                cls_logits, reg_output = self.model(X)

                # Classification labels
                cls_labels = (y > 0).float().unsqueeze(1)

                # Combined loss
                cls_loss = self.criterion_cls(cls_logits, cls_labels)
                reg_loss = self.criterion_reg(reg_output.squeeze(), y)
                loss = 0.5 * cls_loss + 0.5 * reg_loss

                # Metrics
                total_loss += loss.item()
                predictions = (torch.sigmoid(cls_logits) > 0.5).float()
                correct_predictions += (predictions == cls_labels).sum().item()
                total_samples += len(y)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        patience: int = 10,
        save_dir: str = "models/transformer"
    ):
        """
        Complete training pipeline with early stopping and checkpointing

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience
            save_dir: Directory to save model checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        logger.info("=" * 80)
        logger.info("STARTING TRANSFORMER TRAINING")
        logger.info("=" * 80)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning Rate: {learning_rate}")
        logger.info(f"Batch Size: {train_loader.batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info("=" * 80)

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'history': self.history
                }

                checkpoint_path = save_path / f"best_model.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        # Save final model
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(final_checkpoint, save_path / "final_model.pt")

        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best Validation Accuracy: {max(self.history['val_accuracy']):.4f}")
        logger.info(f"Models saved to: {save_path}")
        logger.info("=" * 80)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        logger.info(f"✓ Loaded checkpoint from {checkpoint_path}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data

        Args:
            X: Input sequences (n_samples, seq_length, n_features)

        Returns:
            Tuple of (classification_predictions, regression_predictions)
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            cls_logits, reg_output = self.model(X_tensor)
            cls_pred = torch.sigmoid(cls_logits).cpu().numpy()
            reg_pred = reg_output.cpu().numpy()

        return cls_pred, reg_pred


if __name__ == "__main__":
    from agents.market_data_loader import MarketDataLoader

    # Load and prepare data
    loader = MarketDataLoader()
    data = loader.prepare_training_data(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        sequence_length=60,
        prediction_horizon=1,
        test_split=0.2
    )

    # Initialize trainer
    input_dim = data['X_train'].shape[2]
    trainer = TransformerTrainer(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )

    # Create data loaders
    train_loader, val_loader = trainer.prepare_data_loaders(
        data,
        batch_size=32,
        num_workers=0
    )

    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        learning_rate=1e-4,
        patience=10,
        save_dir="models/transformer"
    )

    print("\n✓ Training complete!")
