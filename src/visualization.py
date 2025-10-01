"""
Visualization module for SAR imagery and change detection results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List, Dict
from pathlib import Path


def plot_sar_image(
    image: np.ndarray,
    title: str = "SAR Image",
    cmap: str = 'gray',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Plot a single SAR image.

    Args:
        image: SAR image
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for color scale (None = auto)
        vmax: Maximum value for color scale (None = auto)
        colorbar: Whether to show colorbar
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Auto-scale if not provided
    if vmin is None:
        vmin = np.percentile(image, 1)
    if vmax is None:
        vmax = np.percentile(image, 99)

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_before_after(
    image_before: np.ndarray,
    image_after: np.ndarray,
    change_map: Optional[np.ndarray] = None,
    titles: Optional[Tuple[str, str, str]] = None,
    cmap: str = 'gray',
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Create side-by-side before/after comparison with optional change map.

    Args:
        image_before: Earlier image
        image_after: Later image
        change_map: Optional change detection map
        titles: Tuple of (before_title, after_title, change_title)
        cmap: Colormap for SAR images
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure
    """
    if titles is None:
        titles = ("Before", "After", "Changes")

    # Determine number of subplots
    n_plots = 3 if change_map is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Common color scale for before/after
    vmin = min(np.percentile(image_before, 1), np.percentile(image_after, 1))
    vmax = max(np.percentile(image_before, 99), np.percentile(image_after, 99))

    # Plot before
    im1 = axes[0].imshow(image_before, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot after
    im2 = axes[1].imshow(image_after, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot change map if provided
    if change_map is not None:
        im3 = axes[2].imshow(change_map, cmap='hot')
        axes[2].set_title(titles[2], fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axes


def plot_change_heatmap(
    change_magnitude: np.ndarray,
    title: str = "Change Magnitude",
    cmap: str = 'RdYlGn_r',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Plot change magnitude as a heatmap.

    Args:
        change_magnitude: Change magnitude array
        title: Plot title
        cmap: Colormap (default: red=increase, green=decrease)
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(change_magnitude, cmap=cmap)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Change Magnitude', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_detections(
    image: np.ndarray,
    detections: List[Dict],
    title: str = "Detected Objects",
    max_detections: int = 50,
    cmap: str = 'gray',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Plot SAR image with detected objects annotated.

    Args:
        image: SAR image
        detections: List of detection dictionaries with 'bbox' and 'centroid'
        title: Plot title
        max_detections: Maximum number of detections to plot
        cmap: Colormap for image
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot image
    vmin, vmax = np.percentile(image, [1, 99])
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title} ({len(detections)} detected)", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Plot detections (limit to top N)
    for i, det in enumerate(detections[:max_detections]):
        # Extract bounding box
        bbox = det['bbox']
        y_min, x_min, y_max, x_max = bbox

        # Draw rectangle
        rect = mpatches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            alpha=0.7
        )
        ax.add_patch(rect)

        # Add label
        centroid = det['centroid']
        ax.plot(centroid[1], centroid[0], 'r+', markersize=10, markeredgewidth=2)

        # Add detection number for top 10
        if i < 10:
            ax.text(
                x_min, y_min - 5,
                str(i + 1),
                color='yellow',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7)
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_change_rgb(
    image_before: np.ndarray,
    image_after: np.ndarray,
    title: str = "Multi-temporal RGB Composite",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Create RGB composite for change visualization.

    Red channel: After image
    Green: Before image
    Blue: Before image

    Changes appear as red (increase) or cyan (decrease).

    Args:
        image_before: Earlier image (normalized 0-1)
        image_after: Later image (normalized 0-1)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure
    """
    from src.preprocessing import normalize_intensity

    # Normalize images to 0-1
    before_norm = normalize_intensity(image_before)
    after_norm = normalize_intensity(image_after)

    # Create RGB composite
    rgb = np.zeros((*before_norm.shape, 3))
    rgb[:, :, 0] = after_norm  # Red: After
    rgb[:, :, 1] = before_norm  # Green: Before
    rgb[:, :, 2] = before_norm  # Blue: Before

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(rgb)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', label='Increase (new bright targets)'),
        mpatches.Patch(color='cyan', label='Decrease (removed targets)'),
        mpatches.Patch(color='gray', label='No change')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def create_summary_plot(
    image_before: np.ndarray,
    image_after: np.ndarray,
    change_map: np.ndarray,
    change_magnitude: np.ndarray,
    detections_before: List[Dict],
    detections_after: List[Dict],
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Create comprehensive summary plot with all analysis results.

    Args:
        image_before: Earlier SAR image
        image_after: Later SAR image
        change_map: Binary change map
        change_magnitude: Change magnitude
        detections_before: Detections in before image
        detections_after: Detections in after image
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    vmin, vmax = np.percentile(np.concatenate([image_before.flatten(), image_after.flatten()]), [1, 99])

    # Before image with detections
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_before, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Before ({len(detections_before)} objects)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # After image with detections
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image_after, cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title(f'After ({len(detections_after)} objects)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Change map
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(change_map, cmap='hot')
    ax3.set_title('Change Detection Map', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Change magnitude heatmap
    ax4 = fig.add_subplot(gs[1, :])
    im = ax4.imshow(change_magnitude, cmap='RdYlGn_r', aspect='auto')
    ax4.set_title('Change Magnitude Heatmap', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.023, pad=0.04, label='Change (dB)')

    plt.suptitle('Sentinel-1 SAR Change Detection Analysis', fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
