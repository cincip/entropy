from entropy import SlidingWindowEntropy, shannon_entropy, compute_entropy_range
from visualization import ascii_heatmap, ascii_sparkline


def example_text_classification():
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Text Classification by Entropy")
    print("=" * 80)
    
    texts = {
        "DNA sequence (repetitive)": "ATGCATGCATGCTAGCTAGC",
        "English prose": "the quick brown fox jumps over the lazy dog repeatedly",
        "Random letters": "qwxzpbmkjfghdslrtynv",
        "Code comments": "# TODO fix this bug. FIXME also check here",
        "Encoded data": "0f8a3b2c5e7d1f4a9b6c",
    }
    
    window_size = 5
    
    for label, text in texts.items():
        entropy = shannon_entropy(text)
        calc = SlidingWindowEntropy(window_size)
        
        try:
            entropies = calc.compute(text)
            avg_entropy = sum(entropies) / len(entropies)
            min_entropy = min(entropies)
            max_entropy = max(entropies)
            
            print(f"\n{label}")
            print(f"  Text: {text}")
            print(f"  Overall entropy: {entropy:.4f} bits")
            print(f"  Avg window entropy: {avg_entropy:.4f} bits")
            print(f"  Range: {min_entropy:.4f} to {max_entropy:.4f} bits")
        except ValueError as e:
            print(f"\n{label}")
            print(f"  Text: {text}")
            print(f"  Note: Text too short for window size {window_size}")


def example_signal_analysis():
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Signal Analysis - Detecting Noise Levels")
    print("=" * 80)

    signals = {
        "Very clean (low noise)": "C" * 30 + "N" + "C" * 30,
        "Moderate noise": "C" * 10 + "N" * 10 + "C" * 10 + "N" * 10 + "C" * 10,
        "High noise": "CNCNCNCNCNCNCNCNCNCN" * 3,
        "Very noisy (noise dominant)": "N" * 30 + "C" + "N" * 30,
    }
    
    window_size = 10
    calc = SlidingWindowEntropy(window_size)
    
    for label, signal in signals.items():
        entropies = calc.compute(signal)
        avg = sum(entropies) / len(entropies)
        
        print(f"\n{label}")
        print(f"  Signal: {signal[:40]}...")
        print(f"  Average entropy: {avg:.4f} bits")
        print(f"  Sparkline: {ascii_sparkline(entropies, width=50)}")


def example_file_format_detection():
    print("\n" + "=" * 80)
    print("EXAMPLE 3: File Format Detection")
    print("=" * 80)
    
    files = {
        "Text file (mostly ASCII)": "abcdefghijklmnopqrstuvwxyz " * 5,
        "Binary file (varied bytes)": bytes(range(256)) + bytes(range(255, -1, -1)),
        "Compressed file (high entropy)": bytes([i ^ ((i + 1) * 17) % 256 for i in range(80)]),
        "Encrypted data (max entropy)": bytes([(i * 59 + 73) % 256 for i in range(80)]),
    }
    
    window_size = 16
    calc = SlidingWindowEntropy(window_size)
    
    for label, data in files.items():
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        entropies = calc.compute(data)
        avg = sum(entropies) / len(entropies)
        entropy_min = min(entropies)
        entropy_max = max(entropies)
        
        print(f"\n{label}")
        print(f"  Data size: {len(data)} bytes")
        print(f"  Entropy stats:")
        print(f"    Min: {entropy_min:.4f} bits")
        print(f"    Max: {entropy_max:.4f} bits")
        print(f"    Avg: {avg:.4f} bits")


def example_anomaly_detection():
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Anomaly Detection")
    print("=" * 80)
    
    normal_prefix = "AAAA" * 10
    anomaly = "BBBBBBBBB"
    normal_suffix = "AAAA" * 10
    data = normal_prefix + anomaly + normal_suffix
    
    window_size = 4
    calc = SlidingWindowEntropy(window_size)
    entropies_with_pos = calc.compute_with_positions(data)
    
    entropies = [e for _, e in entropies_with_pos]
    avg_entropy = sum(entropies) / len(entropies)
    threshold = avg_entropy + 0.5
    
    print(f"\nData: {data}")
    print(f"Window size: {window_size}")
    print(f"Average entropy: {avg_entropy:.4f} bits")
    print(f"Anomaly threshold: {threshold:.4f} bits")
    
    print("\nAnomaly candidates:")
    for pos, entropy in entropies_with_pos:
        if entropy > threshold:
            window_data = data[pos:pos+window_size]
            print(f"  Position {pos:2d}: {window_data:10s} entropy={entropy:.4f} "
                  f"(ANOMALY)" if entropy > avg_entropy + 0.2 else f" (normal)")


def example_compression_analysis():
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Entropy vs Compression Potential")
    print("=" * 80)
    
    import zlib
    
    patterns = {
        "Very repetitive": "A" * 100,
        "Pattern 1": "AB" * 50,
        "Pattern 2": "ABCD" * 25,
        "Pattern 3": "ABCDEFGH" * 12 + "AB" * 2,
    }
    
    print("\nCorrelation between entropy and compression ratio:")
    print(f"{'Pattern':<20} {'Entropy':<12} {'Compress %':<12}")
    print("-" * 44)
    
    for label, data in patterns.items():
        data_bytes = data.encode('utf-8')
        entropy = shannon_entropy(data_bytes)
        compressed = zlib.compress(data_bytes)
        
        ratio = 100 * (1 - len(compressed) / len(data_bytes))
        print(f"{label:<20} {entropy:<12.4f} {ratio:<12.1f}%")
    
    print("\nNote: Higher entropy → Lower compression potential")


def example_multi_scale_analysis():
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Multi-Scale Analysis")
    print("=" * 80)
    
    data = (
        "A" * 8 +
        "B" * 8 +
        "AB" * 8 +
        "ABCD" * 4
    )
    
    print(f"\nData: {data}")
    print("\nEntropy at different scales:")
    print(f"{'Window Size':<15} {'Avg Entropy':<15} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    
    for window_size in [1, 2, 4, 8, 16]:
        calc = SlidingWindowEntropy(window_size)
        entropies = calc.compute(data)
        
        avg = sum(entropies) / len(entropies)
        min_e = min(entropies)
        max_e = max(entropies)
        
        print(f"{window_size:<15} {avg:<15.4f} {min_e:<10.4f} {max_e:<10.4f}")


def main():
    print("\n" + "=" * 80)
    print("Shannon Entropy Sliding Window - Practical Examples")
    print("=" * 80)
    
    example_text_classification()
    example_signal_analysis()
    example_file_format_detection()
    example_anomaly_detection()
    example_compression_analysis()
    example_multi_scale_analysis()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
