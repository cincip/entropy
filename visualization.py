from typing import List, Tuple, Optional
import math


def ascii_heatmap(
    entropies: List[float],
    width: int = 80,
    height: int = 20,
    title: str = "Entropy Heatmap",
) -> str:
    if not entropies:
        return ""
    
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    
    entropy_range = max_entropy - min_entropy
    if entropy_range == 0:
        entropy_range = 1.0
    
    chars = " .:-=@#"
    
    output_lines = []
    
    output_lines.append(title.center(width))
    output_lines.append("-" * width)
    
    sample_indices = [
        int(i * len(entropies) / width) for i in range(width)
    ]
    sampled_entropies = [entropies[idx] for idx in sample_indices]
    
    for row in range(height, 0, -1):
        threshold_low = min_entropy + (entropy_range * (row - 1) / height)
        threshold_high = min_entropy + (entropy_range * row / height)
        
        line = ""
        for entropy in sampled_entropies:
            if entropy < threshold_low:
                line += " "
            elif entropy < threshold_high:
                char_idx = min(len(chars) - 1, int((entropy - min_entropy) / entropy_range * len(chars)))
                line += chars[char_idx]
            else:
                line += chars[-1]
        
        output_lines.append(line)
    
    output_lines.append("-" * width)
    
    legend = f"Min: {min_entropy:.3f} bits | Max: {max_entropy:.3f} bits"
    output_lines.append(legend)
    output_lines.append(f"Position range: 0 to {len(entropies) - 1}")
    
    return "\n".join(output_lines)


def ascii_sparkline(entropies: List[float], width: int = 80) -> str:
    if not entropies:
        return ""
    
    sparkline_chars = "▁▂▃▄▅▆▇█"
    
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    entropy_range = max(max_entropy - min_entropy, 0.001)
    
    sample_indices = [
        int(i * len(entropies) / width) for i in range(width)
    ]
    sampled_entropies = [entropies[idx] for idx in sample_indices]
    
    sparkline = ""
    for entropy in sampled_entropies:
        normalized = (entropy - min_entropy) / entropy_range
        char_idx = min(len(sparkline_chars) - 1, int(normalized * len(sparkline_chars)))
        sparkline += sparkline_chars[char_idx]
    
    return sparkline


def plot_entropy_with_matplotlib(
    entropies: List[float],
    window_size: int,
    title: str = "Shannon Entropy over Sliding Window",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    positions = list(range(len(entropies)))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(positions, entropies, marker='o', markersize=3, linewidth=1.5, alpha=0.7)
    ax.fill_between(positions, entropies, alpha=0.2)
    
    ax.set_xlabel("Position")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title(f"{title} (window size: {window_size})")
    ax.grid(True, alpha=0.3)
    
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    mean_entropy = sum(entropies) / len(entropies)
    
    stats_text = f"Min: {min_entropy:.3f} | Max: {max_entropy:.3f} | Mean: {mean_entropy:.3f}"
    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()