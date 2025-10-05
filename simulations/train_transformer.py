"""
Complete Transformer Training Pipeline
Trains market prediction model on real data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from agents.transformer_trainer import TransformerTrainer
from agents.market_data_loader import MarketDataLoader
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""

    logger.info("=" * 80)
    logger.info("TRANSFORMER MARKET PREDICTOR - TRAINING PIPELINE")
    logger.info("=" * 80)

    # Configuration
    config = {
        'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
        'start_date': '2020-01-01',
        'end_date': '2024-12-31',
        'sequence_length': 60,
        'prediction_horizon': 1,
        'test_split': 0.2,

        # Model hyperparameters
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,

        # Training hyperparameters
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 1e-4,
        'patience': 15,

        # Paths
        'save_dir': 'models/transformer',
        'data_cache': 'data/market_data_cache.pkl'
    }

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    # Step 1: Load and prepare data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING AND PREPARING DATA")
    logger.info("=" * 80)

    data_loader = MarketDataLoader()

    try:
        # Try to load cached data
        logger.info("Checking for cached data...")
        import pickle
        with open(config['data_cache'], 'rb') as f:
            data = pickle.load(f)
        logger.info("✓ Loaded cached data")

    except FileNotFoundError:
        logger.info("No cache found. Downloading and preparing data...")

        data = data_loader.prepare_training_data(
            tickers=config['tickers'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon'],
            test_split=config['test_split']
        )

        # Cache the data
        logger.info(f"Caching data to {config['data_cache']}...")
        Path('data').mkdir(exist_ok=True)
        with open(config['data_cache'], 'wb') as f:
            pickle.dump(data, f)
        logger.info("✓ Data cached successfully")

    # Data statistics
    logger.info("\nData Statistics:")
    logger.info(f"  Training samples: {len(data['X_train'])}")
    logger.info(f"  Test samples: {len(data['X_test'])}")
    logger.info(f"  Sequence length: {data['X_train'].shape[1]}")
    logger.info(f"  Features per timestep: {data['X_train'].shape[2]}")
    logger.info(f"  Total trainable examples: {len(data['X_train']) * data['X_train'].shape[1]}")

    # Step 2: Initialize trainer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: INITIALIZING TRANSFORMER MODEL")
    logger.info("=" * 80)

    input_dim = data['X_train'].shape[2]

    trainer = TransformerTrainer(
        input_dim=input_dim,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )

    # Count parameters
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    logger.info(f"Model Architecture:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")

    # Step 3: Create data loaders
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CREATING DATA LOADERS")
    logger.info("=" * 80)

    train_loader, test_loader = trainer.prepare_data_loaders(
        data,
        batch_size=config['batch_size'],
        num_workers=0
    )

    logger.info(f"Data Loaders Created:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    logger.info(f"  Batch size: {config['batch_size']}")

    # Step 4: Train model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: TRAINING MODEL")
    logger.info("=" * 80)

    start_time = datetime.now()

    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        save_dir=config['save_dir']
    )

    end_time = datetime.now()
    training_duration = end_time - start_time

    # Step 5: Evaluate model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: FINAL EVALUATION")
    logger.info("=" * 80)

    # Load best model
    best_model_path = Path(config['save_dir']) / 'best_model.pt'
    trainer.load_checkpoint(str(best_model_path))

    # Evaluate on test set
    test_loss, test_acc = trainer.validate(test_loader)

    logger.info("Final Test Results:")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Training Duration: {training_duration}")

    # Make sample predictions
    logger.info("\nSample Predictions:")
    sample_X = data['X_test'][:5]
    sample_y_true = data['y_test'][:5]

    cls_pred, reg_pred = trainer.predict(sample_X)

    for i in range(5):
        direction = "UP" if cls_pred[i][0] > 0.5 else "DOWN"
        confidence = cls_pred[i][0] if cls_pred[i][0] > 0.5 else 1 - cls_pred[i][0]
        predicted_return = reg_pred[i][0]
        actual_return = sample_y_true[i]

        logger.info(f"  Sample {i+1}:")
        logger.info(f"    Predicted Direction: {direction} (confidence: {confidence:.2%})")
        logger.info(f"    Predicted Return: {predicted_return:+.4f}")
        logger.info(f"    Actual Return: {actual_return:+.4f}")
        logger.info(f"    Error: {abs(predicted_return - actual_return):.4f}")

    # Step 6: Save configuration
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: SAVING CONFIGURATION")
    logger.info("=" * 80)

    final_config = {
        **config,
        'training_duration': str(training_duration),
        'final_test_loss': float(test_loss),
        'final_test_accuracy': float(test_acc),
        'total_parameters': total_params,
        'training_date': datetime.now().isoformat()
    }

    config_path = Path(config['save_dir']) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=2)

    logger.info(f"✓ Configuration saved to {config_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info(f"  ✓ Model trained on {len(config['tickers'])} tickers")
    logger.info(f"  ✓ {len(data['X_train'])} training samples processed")
    logger.info(f"  ✓ Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"  ✓ Final test accuracy: {test_acc:.4f}")
    logger.info(f"  ✓ Duration: {training_duration}")
    logger.info(f"  ✓ Models saved to: {config['save_dir']}")
    logger.info("=" * 80)

    return trainer, data


if __name__ == "__main__":
    try:
        trainer, data = main()
        logger.info("\n✓ Training pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\n✗ Training interrupted by user")

    except Exception as e:
        logger.error(f"\n✗ Training failed with error: {e}", exc_info=True)
        raise
