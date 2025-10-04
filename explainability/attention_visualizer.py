"""
Attention Visualization for Transformer Models
Visualizes attention weights and patterns for interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualizes attention patterns from transformer-based trading models

    Features:
    - Attention heatmaps per head and layer
    - Token importance visualization
    - Temporal attention patterns
    - Multi-head comparison
    """

    def __init__(self, output_dir: str = "explainability/attention_plots"):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Attention Heatmap",
        save_name: Optional[str] = None,
        cmap: str = "viridis"
    ):
        """
        Plot attention weights as heatmap

        Args:
            attention_weights: (seq_len, seq_len) attention matrix
            feature_names: Names for sequence positions
            title: Plot title
            save_name: Filename to save plot
            cmap: Colormap name
        """
        try:
            plt.figure(figsize=(12, 10))

            # Create heatmap
            sns.heatmap(
                attention_weights,
                cmap=cmap,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=feature_names if feature_names else False,
                yticklabels=feature_names if feature_names else False
            )

            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Key Position', fontsize=12)
            plt.ylabel('Query Position', fontsize=12)
            plt.tight_layout()

            # Save
            if save_name:
                save_path = self.output_dir / save_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Attention heatmap saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting attention heatmap: {e}")

    def plot_multi_head_attention(
        self,
        attention_weights: np.ndarray,
        num_heads: int,
        feature_names: Optional[List[str]] = None,
        title: str = "Multi-Head Attention",
        save_name: Optional[str] = None
    ):
        """
        Visualize attention from multiple heads

        Args:
            attention_weights: (num_heads, seq_len, seq_len) tensor
            num_heads: Number of attention heads
            feature_names: Names for sequence positions
            title: Plot title
            save_name: Filename to save plot
        """
        try:
            # Calculate grid size
            cols = min(4, num_heads)
            rows = (num_heads + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if num_heads == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(rows, cols)

            for head_idx in range(num_heads):
                row = head_idx // cols
                col = head_idx % cols
                ax = axes[row, col]

                # Plot heatmap for this head
                im = ax.imshow(
                    attention_weights[head_idx],
                    cmap='viridis',
                    aspect='auto'
                )

                ax.set_title(f'Head {head_idx + 1}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Key Position', fontsize=8)
                ax.set_ylabel('Query Position', fontsize=8)

                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Set tick labels if provided
                if feature_names:
                    ax.set_xticks(range(len(feature_names)))
                    ax.set_yticks(range(len(feature_names)))
                    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=6)
                    ax.set_yticklabels(feature_names, fontsize=6)

            # Remove empty subplots
            for idx in range(num_heads, rows * cols):
                row = idx // cols
                col = idx % cols
                fig.delaxes(axes[row, col])

            plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()

            # Save
            if save_name:
                save_path = self.output_dir / save_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Multi-head attention plot saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting multi-head attention: {e}")

    def plot_temporal_attention_pattern(
        self,
        attention_weights: np.ndarray,
        time_steps: Optional[List[str]] = None,
        title: str = "Temporal Attention Pattern",
        save_name: Optional[str] = None
    ):
        """
        Visualize how attention focuses over time

        Args:
            attention_weights: (seq_len, seq_len) or (num_steps, seq_len, seq_len)
            time_steps: Labels for time steps
            title: Plot title
            save_name: Filename to save plot
        """
        try:
            if attention_weights.ndim == 2:
                # Single time step
                attention_weights = attention_weights[np.newaxis, :, :]

            num_steps = attention_weights.shape[0]
            seq_len = attention_weights.shape[1]

            # Calculate attention distribution over positions
            avg_attention_per_position = attention_weights.mean(axis=1)  # (num_steps, seq_len)

            plt.figure(figsize=(14, 6))

            # Plot as heatmap
            plt.subplot(1, 2, 1)
            sns.heatmap(
                avg_attention_per_position.T,
                cmap='YlOrRd',
                cbar_kws={'label': 'Average Attention Weight'},
                xticklabels=time_steps if time_steps else False,
                yticklabels=[f'Pos {i}' for i in range(seq_len)]
            )
            plt.title('Attention Distribution Over Time', fontsize=12, fontweight='bold')
            plt.xlabel('Time Step', fontsize=10)
            plt.ylabel('Sequence Position', fontsize=10)

            # Plot as line chart
            plt.subplot(1, 2, 2)
            for pos in range(min(seq_len, 10)):  # Plot top 10 positions
                plt.plot(
                    avg_attention_per_position[:, pos],
                    label=f'Position {pos}',
                    marker='o',
                    linewidth=2,
                    markersize=4
                )

            plt.title('Attention Weight Trends', fontsize=12, fontweight='bold')
            plt.xlabel('Time Step', fontsize=10)
            plt.ylabel('Average Attention Weight', fontsize=10)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.3)

            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            # Save
            if save_name:
                save_path = self.output_dir / save_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Temporal attention pattern saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting temporal attention: {e}")

    def plot_token_importance(
        self,
        attention_weights: np.ndarray,
        token_names: List[str],
        title: str = "Token Importance from Attention",
        save_name: Optional[str] = None
    ):
        """
        Visualize token importance based on attention weights

        Args:
            attention_weights: (seq_len, seq_len) attention matrix
            token_names: Names of tokens/features
            title: Plot title
            save_name: Filename to save plot
        """
        try:
            # Calculate token importance as sum of attention received
            token_importance = attention_weights.sum(axis=0)  # Sum over query positions

            # Normalize
            token_importance = token_importance / token_importance.sum()

            # Sort by importance
            sorted_indices = np.argsort(token_importance)[::-1]
            sorted_importance = token_importance[sorted_indices]
            sorted_names = [token_names[i] for i in sorted_indices]

            # Plot
            plt.figure(figsize=(12, 6))

            # Bar plot
            plt.subplot(1, 2, 1)
            bars = plt.barh(range(len(sorted_names)), sorted_importance)

            # Color bars by importance
            colors = plt.cm.viridis(sorted_importance / sorted_importance.max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Normalized Attention Weight', fontsize=10)
            plt.title('Feature Importance Ranking', fontsize=12, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)

            # Cumulative plot
            plt.subplot(1, 2, 2)
            cumulative = np.cumsum(sorted_importance)
            plt.plot(range(len(cumulative)), cumulative, marker='o', linewidth=2)
            plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
            plt.xlabel('Number of Features', fontsize=10)
            plt.ylabel('Cumulative Attention Weight', fontsize=10)
            plt.title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            # Save
            if save_name:
                save_path = self.output_dir / save_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Token importance plot saved to {save_path}")

            plt.close()

            return {
                'token_names': sorted_names,
                'importance': sorted_importance.tolist(),
                'cumulative': cumulative.tolist()
            }

        except Exception as e:
            logger.error(f"Error plotting token importance: {e}")
            return None

    def plot_layer_attention_comparison(
        self,
        layer_attentions: Dict[str, np.ndarray],
        feature_names: Optional[List[str]] = None,
        title: str = "Attention Patterns Across Layers",
        save_name: Optional[str] = None
    ):
        """
        Compare attention patterns across different layers

        Args:
            layer_attentions: Dict mapping layer name to attention weights
            feature_names: Names for sequence positions
            title: Plot title
            save_name: Filename to save plot
        """
        try:
            num_layers = len(layer_attentions)

            fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
            if num_layers == 1:
                axes = [axes]

            for idx, (layer_name, attention) in enumerate(layer_attentions.items()):
                ax = axes[idx]

                # Plot heatmap
                im = ax.imshow(attention, cmap='viridis', aspect='auto')
                ax.set_title(f'{layer_name}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Key Position', fontsize=8)
                ax.set_ylabel('Query Position', fontsize=8)

                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Set tick labels
                if feature_names:
                    ax.set_xticks(range(len(feature_names)))
                    ax.set_yticks(range(len(feature_names)))
                    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=6)
                    ax.set_yticklabels(feature_names, fontsize=6)

            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            # Save
            if save_name:
                save_path = self.output_dir / save_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Layer comparison plot saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting layer comparison: {e}")

    def extract_attention_from_model(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from transformer model

        Args:
            model: PyTorch transformer model
            input_tensor: Input tensor to model
            layer_names: Names of layers to extract attention from

        Returns:
            Dict mapping layer names to attention weights
        """
        try:
            attention_weights = {}
            hooks = []

            def attention_hook(module, input, output, name):
                """Hook to capture attention weights"""
                if isinstance(output, tuple) and len(output) > 1:
                    # Assume second output is attention weights
                    attn = output[1]
                elif hasattr(output, 'attentions'):
                    attn = output.attentions
                else:
                    return

                # Convert to numpy
                if torch.is_tensor(attn):
                    attention_weights[name] = attn.detach().cpu().numpy()

            # Register hooks
            for name, module in model.named_modules():
                if 'attention' in name.lower() or 'attn' in name.lower():
                    if layer_names is None or name in layer_names:
                        hook = module.register_forward_hook(
                            lambda m, i, o, n=name: attention_hook(m, i, o, n)
                        )
                        hooks.append(hook)

            # Forward pass
            model.eval()
            with torch.no_grad():
                _ = model(input_tensor)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            return attention_weights

        except Exception as e:
            logger.error(f"Error extracting attention: {e}")
            return {}

    def create_attention_summary(
        self,
        attention_weights: np.ndarray,
        feature_names: List[str],
        save_name: str = "attention_summary.png"
    ):
        """
        Create comprehensive summary of attention patterns

        Args:
            attention_weights: (seq_len, seq_len) or (num_heads, seq_len, seq_len)
            feature_names: Names of features/tokens
            save_name: Filename to save summary
        """
        try:
            if attention_weights.ndim == 3:
                # Average over heads
                avg_attention = attention_weights.mean(axis=0)
            else:
                avg_attention = attention_weights

            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Main heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            sns.heatmap(
                avg_attention,
                cmap='viridis',
                ax=ax1,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cbar_kws={'label': 'Attention Weight'}
            )
            ax1.set_title('Average Attention Heatmap', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Key Features')
            ax1.set_ylabel('Query Features')

            # 2. Column-wise attention (feature importance)
            ax2 = fig.add_subplot(gs[0, 1])
            col_attention = avg_attention.sum(axis=0)
            col_attention = col_attention / col_attention.sum()
            bars = ax2.bar(range(len(feature_names)), col_attention)
            ax2.set_xticks(range(len(feature_names)))
            ax2.set_xticklabels(feature_names, rotation=45, ha='right')
            ax2.set_ylabel('Normalized Attention Weight')
            ax2.set_title('Feature Importance', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)

            # Color bars
            colors = plt.cm.viridis(col_attention / col_attention.max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # 3. Row-wise attention distribution
            ax3 = fig.add_subplot(gs[1, 0])
            row_attention = avg_attention.sum(axis=1)
            im = ax3.imshow(row_attention.reshape(-1, 1), cmap='YlOrRd', aspect='auto')
            ax3.set_yticks(range(len(feature_names)))
            ax3.set_yticklabels(feature_names)
            ax3.set_xticks([])
            ax3.set_title('Query Attention Distribution', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

            # 4. Attention statistics
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')

            # Calculate statistics
            stats_text = f"""
Attention Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Max Attention: {avg_attention.max():.4f}
Min Attention: {avg_attention.min():.4f}
Mean Attention: {avg_attention.mean():.4f}
Std Attention: {avg_attention.std():.4f}

Sparsity: {(avg_attention < 0.01).sum() / avg_attention.size * 100:.2f}%

Top 3 Most Attended Features:
"""
            top_3_idx = col_attention.argsort()[::-1][:3]
            for i, idx in enumerate(top_3_idx, 1):
                stats_text += f"{i}. {feature_names[idx]}: {col_attention[idx]:.4f}\n"

            ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')

            plt.suptitle('Attention Analysis Summary', fontsize=16, fontweight='bold')

            # Save
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention summary saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error creating attention summary: {e}")


def generate_default_feature_names(seq_len: int) -> List[str]:
    """Generate default feature names for sequence"""
    return [f"T-{seq_len - i}" for i in range(seq_len)]


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    visualizer = AttentionVisualizer()

    # Create synthetic attention weights
    seq_len = 10
    num_heads = 4

    # Single head attention
    attention = np.random.rand(seq_len, seq_len)
    attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize

    feature_names = [
        'Price', 'Volume', 'RSI', 'MACD', 'BB_upper',
        'BB_lower', 'SMA_20', 'EMA_12', 'ATR', 'OBV'
    ]

    # Plot single attention heatmap
    visualizer.plot_attention_heatmap(
        attention,
        feature_names=feature_names,
        title="Trading Feature Attention",
        save_name="single_attention.png"
    )

    # Multi-head attention
    multi_head_attention = np.random.rand(num_heads, seq_len, seq_len)
    for i in range(num_heads):
        multi_head_attention[i] = multi_head_attention[i] / multi_head_attention[i].sum(axis=1, keepdims=True)

    visualizer.plot_multi_head_attention(
        multi_head_attention,
        num_heads=num_heads,
        feature_names=feature_names,
        title="Multi-Head Trading Attention",
        save_name="multi_head_attention.png"
    )

    # Token importance
    visualizer.plot_token_importance(
        attention,
        token_names=feature_names,
        title="Trading Feature Importance",
        save_name="feature_importance.png"
    )

    # Comprehensive summary
    visualizer.create_attention_summary(
        multi_head_attention,
        feature_names=feature_names,
        save_name="attention_summary.png"
    )

    print("Attention visualizations generated successfully!")
