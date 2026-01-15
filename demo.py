"""
Demo script for Shannon entropy over sliding windows.

How to use:
  python demo.py                           # Run all demos
  python demo.py --text                    # Text entropy demo
  python demo.py --bytes                   # Binary data demo
  python demo.py --matplotlib              # Show matplotlib plot
"""

import sys
import argparse
from entropy import SlidingWindowEntropy, shannon_entropy, compute_entropy_range
from visualization import ascii_heatmap, ascii_sparkline, plot_entropy_with_matplotlib


def demo_text_entropy():
    print("\n" + "=" * 80)
    print("DEMO 1: Text Entropy Analysis")
    print("=" * 80)
    text_structured = "AAAAAABBBBBBCCCCCC"
    text_random = "ABJXKDMQZPWLFHGBCX"
    text_natural = "the quick brown fox jumps over the lazy dog"
    
    samples = [
        ("Structured (AAAAAABBBBBBCCCCCC)", text_structured),
        ("Random (ABJXKDMQZPWLFHGBCX)", text_random),
        ("Natural Language", text_natural),
    ]
    
    for label, text in samples:
        entropy = shannon_entropy(text)
        print(f"\n{label}")
        print(f"  Text: {text}")
        print(f"  Shannon Entropy: {entropy:.4f} bits")
        print(f"  Max possible (uniform): {len(set(text.encode('utf-8')))**0.5:.4f} bits")


def demo_sliding_window():
    print("\n" + "=" * 80)
    print("DEMO 2: Sliding Window Entropy")
    print("=" * 80)
    
    data = (
        "AAAA" * 5 +
        "ABAB" * 5 +
        "XYZW" * 5 +
        "JKQMXPBVFGHZWLD" * 2
    )
    
    window_size = 8
    calc = SlidingWindowEntropy(window_size)
    entropies = calc.compute(data)
    
    print(f"\nData: {data}")
    print(f"Data length: {len(data)}")
    print(f"Window size: {window_size}")
    print(f"Number of windows: {len(entropies)}")
    
    min_ent, max_ent = compute_entropy_range(data, window_size)
    print(f"\nEntropy range: {min_ent:.4f} to {max_ent:.4f} bits")
    
    print("\nSample entropy values:")
    positions_to_show = [0, len(entropies) // 4, len(entropies) // 2, 
                         3 * len(entropies) // 4, len(entropies) - 1]
    
    for pos in positions_to_show:
        if pos < len(entropies):
            window_data = data[pos:pos + window_size]
            print(f"  Position {pos:3d}: {window_data:20s} → {entropies[pos]:.4f} bits")


def demo_visualizations():
    """Demonstrate ASCII visualizations."""
    print("\n" + "=" * 80)
    print("DEMO 3: ASCII Visualizations")
    print("=" * 80)
    
    data = (
        "AAAA" * 3 +
        "AABB" * 3 +
        "ABCD" * 3 +
        "ABCDEFGHIJ" * 2
    )
    
    window_size = 6
    calc = SlidingWindowEntropy(window_size)
    entropies = calc.compute(data)
    
    print("\nSparkline visualization:")
    sparkline = ascii_sparkline(entropies, width=60)
    print(sparkline)
    
    print("\n\nHeatmap visualization:")
    heatmap = ascii_heatmap(
        entropies,
        width=70,
        height=12,
        title=f"Entropy Heatmap (Window size: {window_size})"
    )
    print(heatmap)


def demo_binary_data():
    print("\n" + "=" * 80)
    print("DEMO 4: Binary Data Entropy")
    print("=" * 80)
    
    patterns = [
        (b"\x00" * 16, "All zeros"),
        (b"\xFF" * 16, "All ones"),
        (b"\x00\xFF" * 8, "Alternating 0x00 and 0xFF"),
        (bytes(range(16)), "Sequential bytes 0-15"),
    ]
    
    window_size = 4
    calc = SlidingWindowEntropy(window_size)
    
    for data, description in patterns:
        print(f"\n{description}")
        print(f"  Hex: {data.hex()}")
        entropies = calc.compute(data)
        print(f"  Entropy values: {', '.join(f'{e:.4f}' for e in entropies[:5])}")
        if len(entropies) > 5:
            print(f"  ... (showing first 5 of {len(entropies)} windows)")


def demo_compression_correlation():
    print("\n" + "=" * 80)
    print("DEMO 5: Entropy and Compressibility")
    print("=" * 80)
    
    import zlib
    
    data_samples = [
        ("Repetitive", "AAAA" * 50),
        ("Pattern", "ABCD" * 50),
        ("Random-like", "JKQMXPBVFGHZWLDSRTUVWXYZABC" * 7),
    ]
    
    for label, text in data_samples:
        data_bytes = text.encode('utf-8')
        entropy = shannon_entropy(data_bytes)
        
        compressed = zlib.compress(data_bytes)
        original_size = len(data_bytes)
        compressed_size = len(compressed)
        compression_ratio = 100 * (1 - compressed_size / original_size)
        
        print(f"\n{label}")
        print(f"  Original size: {original_size} bytes")
        print(f"  Compressed size: {compressed_size} bytes")
        print(f"  Compression ratio: {compression_ratio:.1f}%")
        print(f"  Shannon entropy: {entropy:.4f} bits")
        print(f"  Max compression (theory): {100 * (1 - entropy / 8):.1f}% (for 8-bit symbols)")


def main():
    parser = argparse.ArgumentParser(
        description="Shannon entropy sliding window demonstrations"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run text entropy demo only"
    )
    parser.add_argument(
        "--bytes",
        action="store_true",
        help="Run binary data demo only"
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="Run sliding window demo only"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Run visualization demo only"
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Run compression correlation demo only"
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Generate matplotlib plot (requires matplotlib)"
    )
    
    args = parser.parse_args()
    
    run_all = not any([args.text, args.bytes, args.window, args.viz, 
                       args.compression, args.matplotlib])
    
    if args.text or run_all:
        demo_text_entropy()
    if args.bytes or run_all:
        demo_binary_data()
    if args.window or run_all:
        demo_sliding_window()
    if args.viz or run_all:
        demo_visualizations()
    if args.compression or run_all:
        demo_compression_correlation()
    
    if args.matplotlib:
        try:
            data = (
                "A" * 15 +
                "AB" * 15 +
                "ABCD" * 15 +
                "ABCDEFGHIJKLMNOP" * 5
            )
            window_size = 12
            calc = SlidingWindowEntropy(window_size)
            entropies = calc.compute(data)
            
            plot_entropy_with_matplotlib(
                entropies,
                window_size,
                title="Shannon Entropy Analysis",
                figsize=(14, 7)
            )
        except ImportError as e:
            print(f"\n{e}")
            print("Install matplotlib with: pip install matplotlib")
    
    print("\n" + "=" * 80)
    print("Demos completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
