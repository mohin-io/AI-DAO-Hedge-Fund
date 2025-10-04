"""
Transformer-based Market Predictor
Uses attention mechanisms for time-series forecasting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerPredictor(nn.Module):
    """
    Transformer-based market prediction model

    Architecture:
    - Multi-head self-attention for capturing temporal dependencies
    - Feed-forward layers for feature extraction
    - Positional encoding for sequence ordering
    - Classification head for market direction prediction
    """

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        sequence_length: int = 60,
        num_classes: int = 3  # Up, Down, Neutral
    ):
        super().__init__()

        self.d_model = d_model
        self.sequence_length = sequence_length

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )

        # Regression head (for price prediction)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        predict_price: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            predict_price: If True, predict price change; else predict direction

        Returns:
            predictions: (batch_size, num_classes) or (batch_size, 1)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Transpose for transformer: (seq_len, batch, d_model)
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        encoded = self.transformer_encoder(x)  # (seq_len, batch, d_model)

        # Take the last time step
        last_hidden = encoded[-1]  # (batch, d_model)

        # Predict
        if predict_price:
            output = self.regressor(last_hidden)  # (batch, 1)
        else:
            output = self.classifier(last_hidden)  # (batch, num_classes)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for interpretability

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            Attention weights from all heads
        """
        # This requires modifying the transformer to return attention weights
        # For simplicity, we'll implement this later
        pass


class MarketPredictor:
    """Wrapper class for training and inference"""

    def __init__(
        self,
        model: TransformerPredictor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

    def setup_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Setup optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_step(
        self,
        x: torch.Tensor,
        y_direction: Optional[torch.Tensor] = None,
        y_price: Optional[torch.Tensor] = None
    ) -> Tuple[float, float]:
        """
        Single training step

        Args:
            x: Input features (batch_size, seq_len, input_dim)
            y_direction: Direction labels (batch_size,) - 0=down, 1=neutral, 2=up
            y_price: Price change targets (batch_size, 1)

        Returns:
            classification_loss, regression_loss
        """
        self.model.train()

        x = x.to(self.device)
        cls_loss = 0.0
        reg_loss = 0.0

        # Classification
        if y_direction is not None:
            y_direction = y_direction.to(self.device)
            pred_direction = self.model(x, predict_price=False)
            cls_loss = self.criterion_cls(pred_direction, y_direction)

        # Regression
        if y_price is not None:
            y_price = y_price.to(self.device)
            pred_price = self.model(x, predict_price=True)
            reg_loss = self.criterion_reg(pred_price, y_price)

        # Combined loss
        total_loss = cls_loss + reg_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return cls_loss.item(), reg_loss.item()

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions

        Args:
            x: Input features (batch_size, seq_len, input_dim)
            return_probs: If True, return probabilities; else return class predictions

        Returns:
            direction_preds: (batch_size, 3) probabilities or (batch_size,) classes
            price_preds: (batch_size, 1) predicted price changes
        """
        self.model.eval()

        x = x.to(self.device)

        # Direction prediction
        pred_direction = self.model(x, predict_price=False)
        if return_probs:
            direction_preds = torch.softmax(pred_direction, dim=-1).cpu().numpy()
        else:
            direction_preds = torch.argmax(pred_direction, dim=-1).cpu().numpy()

        # Price prediction
        pred_price = self.model(x, predict_price=True)
        price_preds = pred_price.cpu().numpy()

        return direction_preds, price_preds

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


def prepare_market_data(
    prices: np.ndarray,
    volumes: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare market data for transformer input

    Args:
        prices: Array of prices (n_samples,)
        volumes: Array of volumes (n_samples,)
        sequence_length: Length of input sequences

    Returns:
        X: Input sequences (n_samples - seq_len, seq_len, n_features)
        y_direction: Direction labels (n_samples - seq_len,)
        y_price: Price change targets (n_samples - seq_len, 1)
    """
    # Calculate features
    returns = np.diff(prices) / prices[:-1]
    log_returns = np.log(prices[1:] / prices[:-1])
    volume_change = np.diff(volumes) / volumes[:-1]

    # Pad to match length
    returns = np.concatenate([[0], returns])
    log_returns = np.concatenate([[0], log_returns])
    volume_change = np.concatenate([[0], volume_change])

    # Stack features
    features = np.stack([
        prices / np.max(prices),  # Normalized prices
        volumes / np.max(volumes),  # Normalized volumes
        returns,
        log_returns,
        volume_change
    ], axis=-1)

    # Create sequences
    X = []
    y_direction = []
    y_price = []

    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])

        # Direction label
        future_return = returns[i]
        if future_return > 0.01:  # 1% up
            direction = 2
        elif future_return < -0.01:  # 1% down
            direction = 0
        else:
            direction = 1  # Neutral

        y_direction.append(direction)
        y_price.append([future_return])

    return np.array(X), np.array(y_direction), np.array(y_price)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    model = TransformerPredictor(
        input_dim=5,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        sequence_length=60
    )

    predictor = MarketPredictor(model)
    predictor.setup_optimizer(lr=1e-4)

    # Generate synthetic data
    prices = 100 + np.cumsum(np.random.randn(1000) * 2)
    volumes = np.random.randint(1000, 10000, 1000)

    X, y_dir, y_price = prepare_market_data(prices, volumes, sequence_length=60)

    print(f"Data shape: X={X.shape}, y_dir={y_dir.shape}, y_price={y_price.shape}")

    # Training example
    for epoch in range(5):
        cls_loss, reg_loss = predictor.train_step(
            torch.FloatTensor(X[:32]),
            torch.LongTensor(y_dir[:32]),
            torch.FloatTensor(y_price[:32])
        )
        print(f"Epoch {epoch}: cls_loss={cls_loss:.4f}, reg_loss={reg_loss:.4f}")

    # Prediction example
    test_x = torch.FloatTensor(X[32:64])
    dir_preds, price_preds = predictor.predict(test_x)
    print(f"Direction predictions shape: {dir_preds.shape}")
    print(f"Price predictions shape: {price_preds.shape}")
