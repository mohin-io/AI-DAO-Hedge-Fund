"""
Ensemble ML Model: LSTM + GRU + Transformer
Combined architecture for superior market predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """
    LSTM-based market predictor
    Long Short-Term Memory for capturing long-term dependencies
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            predictions: (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


class GRUPredictor(nn.Module):
    """
    GRU-based market predictor
    Gated Recurrent Unit for efficient sequence modeling
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            predictions: (batch_size, 1)
        """
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


class EnsemblePredictor(nn.Module):
    """
    Ensemble model combining LSTM, GRU, and Transformer
    Uses weighted averaging for final predictions
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dropout: float = 0.2,
        ensemble_method: str = 'weighted'  # 'weighted', 'attention', 'voting'
    ):
        super().__init__()

        self.ensemble_method = ensemble_method

        # Individual models
        self.lstm = LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        self.gru = GRUPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # Simplified Transformer (from existing TransformerPredictor)
        self.transformer_embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.transformer_fc = nn.Linear(d_model, 1)

        # Ensemble weights
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(3) / 3)  # Equal initial weights
        elif ensemble_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            predictions: (batch_size, 1)
            individual_preds: Dictionary of individual model predictions
        """
        # LSTM prediction
        lstm_pred = self.lstm(x)

        # GRU prediction
        gru_pred = self.gru(x)

        # Transformer prediction
        transformer_embed = self.transformer_embedding(x)
        transformer_out = self.transformer_encoder(transformer_embed)
        transformer_pred = self.transformer_fc(transformer_out[:, -1, :])

        # Store individual predictions
        individual_preds = {
            'lstm': lstm_pred.detach().cpu().numpy(),
            'gru': gru_pred.detach().cpu().numpy(),
            'transformer': transformer_pred.detach().cpu().numpy()
        }

        # Ensemble combination
        if self.ensemble_method == 'weighted':
            # Weighted average
            weights = torch.softmax(self.weights, dim=0)
            ensemble_pred = (
                weights[0] * lstm_pred +
                weights[1] * gru_pred +
                weights[2] * transformer_pred
            )

        elif self.ensemble_method == 'attention':
            # Stack predictions
            stacked = torch.stack([lstm_pred, gru_pred, transformer_pred], dim=-1)
            stacked = stacked.squeeze(1)  # (batch_size, 3)

            # Compute attention weights
            attention_weights = self.attention(stacked)

            # Weighted combination
            ensemble_pred = torch.sum(stacked * attention_weights, dim=-1, keepdim=True)

        elif self.ensemble_method == 'voting':
            # Simple averaging
            ensemble_pred = (lstm_pred + gru_pred + transformer_pred) / 3

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        return ensemble_pred, individual_preds

    def get_model_weights(self):
        """Get current ensemble weights"""
        if self.ensemble_method == 'weighted':
            weights = torch.softmax(self.weights, dim=0)
            return {
                'lstm': weights[0].item(),
                'gru': weights[1].item(),
                'transformer': weights[2].item()
            }
        return None


class EnsembleTrainer:
    """
    Training pipeline for ensemble model
    """

    def __init__(
        self,
        model: EnsemblePredictor,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step"""
        self.model.train()

        X = X.to(self.device)
        y = y.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        predictions, _ = self.model(X)
        loss = self.criterion(predictions.squeeze(), y)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, dict]:
        """Validation step"""
        self.model.eval()

        X = X.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            predictions, individual_preds = self.model(X)
            loss = self.criterion(predictions.squeeze(), y)

            # Calculate individual model losses
            individual_losses = {}
            for model_name, preds in individual_preds.items():
                preds_tensor = torch.from_numpy(preds).to(self.device)
                model_loss = self.criterion(preds_tensor.squeeze(), y)
                individual_losses[model_name] = model_loss.item()

        return loss.item(), individual_losses

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        learning_rate: float = 1e-4
    ):
        """Complete training loop"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        logger.info("=" * 80)
        logger.info("STARTING ENSEMBLE MODEL TRAINING")
        logger.info("=" * 80)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning Rate: {learning_rate}")
        logger.info(f"Ensemble Method: {self.model.ensemble_method}")
        logger.info("=" * 80)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            train_losses = []
            for X_batch, y_batch in train_loader:
                loss = self.train_step(X_batch, y_batch)
                train_losses.append(loss)

            avg_train_loss = np.mean(train_losses)

            # Validation
            val_losses = []
            for X_batch, y_batch in val_loader:
                val_loss, individual_losses = self.validate(X_batch, y_batch)
                val_losses.append(val_loss)

            avg_val_loss = np.mean(val_losses)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

                # Log ensemble weights
                weights = self.model.get_model_weights()
                if weights:
                    logger.info(f"  Ensemble Weights: {weights}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_ensemble_model.pt')

        logger.info("\n" + "=" * 80)
        logger.info("ENSEMBLE TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("ENSEMBLE MODEL: LSTM + GRU + TRANSFORMER")
    print("=" * 80)

    # Initialize model
    input_dim = 50  # Number of features
    model = EnsemblePredictor(
        input_dim=input_dim,
        hidden_dim=128,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        ensemble_method='weighted'
    )

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture:")
    print(f"  - LSTM: 2 layers, 128 hidden units")
    print(f"  - GRU: 2 layers, 128 hidden units")
    print(f"  - Transformer: 3 encoder layers, 8 attention heads")
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 32
    seq_length = 60
    dummy_input = torch.randn(batch_size, seq_length, input_dim)

    ensemble_pred, individual_preds = model(dummy_input)

    print(f"\nForward Pass Test:")
    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {ensemble_pred.shape}")
    print(f"  Individual Models: {list(individual_preds.keys())}")

    print("\n✓ Ensemble model initialized successfully!")
    print("=" * 80)
