import math
import unittest
from entropy import shannon_entropy, SlidingWindowEntropy, compute_entropy_range


class TestShannonEntropy(unittest.TestCase):
    def test_uniform_distribution(self):
        entropy = shannon_entropy("AABB")
        self.assertAlmostEqual(entropy, 1.0, places=5)
    
    def test_single_symbol(self):
        entropy = shannon_entropy("AAAA")
        self.assertAlmostEqual(entropy, 0.0, places=5)
    
    def test_completely_random_four_symbols(self):
        entropy = shannon_entropy("ABCD")
        self.assertAlmostEqual(entropy, 2.0, places=5)
    
    def test_empty_data(self):
        entropy = shannon_entropy("")
        self.assertEqual(entropy, 0.0)
    
    def test_text_input(self):
        entropy_text = shannon_entropy("AA")
        entropy_bytes = shannon_entropy(b"AA")
        self.assertAlmostEqual(entropy_text, entropy_bytes, places=5)
    
    def test_binary_data(self):
        entropy = shannon_entropy(b"\x00\x00\x00\x00")
        self.assertAlmostEqual(entropy, 0.0, places=5)
        entropy = shannon_entropy(b"\x00\xFF\x00\xFF")
        self.assertAlmostEqual(entropy, 1.0, places=5)
    
    def test_entropy_properties(self):
        entropy = shannon_entropy("random_string_12345")
        self.assertGreaterEqual(entropy, 0.0)
        unique_count = len(set("ABCDE"))
        max_entropy = math.log2(unique_count)
        entropy = shannon_entropy("ABCDEABCDE")
        self.assertLessEqual(entropy, max_entropy + 0.001)


class TestSlidingWindowEntropy(unittest.TestCase):
    def test_window_size_validation(self):
        with self.assertRaises(ValueError):
            SlidingWindowEntropy(0)
        
        with self.assertRaises(ValueError):
            SlidingWindowEntropy(-1)
    
    def test_window_size_larger_than_data(self):
        calc = SlidingWindowEntropy(10)
        with self.assertRaises(ValueError):
            calc.compute("short")
    
    def test_sliding_window_count(self):
        data = "AAABBBCCC"
        window_size = 3
        calc = SlidingWindowEntropy(window_size)
        entropies = calc.compute(data)
        expected_count = len(data) - window_size + 1
        self.assertEqual(len(entropies), expected_count)
        self.assertEqual(len(entropies), 7)
    
    def test_sliding_window_values(self):
        data = "AABBCCDD"
        window_size = 2
        calc = SlidingWindowEntropy(window_size)
        entropies = calc.compute(data)
        expected_entropies = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        for computed, expected in zip(entropies, expected_entropies):
            self.assertAlmostEqual(computed, expected, places=5)
    
    def test_compute_with_positions(self):
        data = "AAAABBBBCCCC"
        window_size = 4
        calc = SlidingWindowEntropy(window_size)
        results = calc.compute_with_positions(data)
        self.assertEqual(len(results), len(data) - window_size + 1)
        for position, entropy in results:
            self.assertIsInstance(position, int)
            self.assertIsInstance(entropy, float)
            self.assertGreaterEqual(entropy, 0.0)
    
    def test_increasing_entropy_pattern(self):
        data = "AAAA" + "ABAB" + "ABCD"
        window_size = 4
        calc = SlidingWindowEntropy(window_size)
        entropies = calc.compute(data)
        self.assertLess(entropies[0], entropies[-1])


class TestComputeEntropyRange(unittest.TestCase):
    def test_entropy_range(self):
        data = "AAAA" + "ABCD"
        window_size = 2
        min_ent, max_ent = compute_entropy_range(data, window_size)
        self.assertLess(min_ent, 0.5)
        self.assertGreaterEqual(max_ent, 1.0)
    
    def test_uniform_entropy_range(self):
        data = "AAAA" * 10
        window_size = 2
        min_ent, max_ent = compute_entropy_range(data, window_size)
        self.assertAlmostEqual(min_ent, 0.0, places=5)
        self.assertAlmostEqual(max_ent, 0.0, places=5)


class TestDataTypeHandling(unittest.TestCase):
    def test_text_vs_bytes_consistency(self):
        text = "hello world"
        entropy_text = shannon_entropy(text)
        entropy_bytes = shannon_entropy(text.encode('utf-8'))
        self.assertAlmostEqual(entropy_text, entropy_bytes, places=5)
    
    def test_utf8_handling(self):
        text = "café"
        entropy = shannon_entropy(text)
        self.assertGreaterEqual(entropy, 0.0)
    
    def test_sliding_window_with_bytes(self):
        data = b"\x00\xFF\x00\xFF\x00\xFF"
        window_size = 2
        calc = SlidingWindowEntropy(window_size)
        entropies = calc.compute(data)
        for entropy in entropies:
            self.assertAlmostEqual(entropy, 1.0, places=5)


class TestEdgeCases(unittest.TestCase):
    def test_single_byte_data(self):
        calc = SlidingWindowEntropy(1)
        entropies = calc.compute("A")
        self.assertEqual(len(entropies), 1)
        self.assertAlmostEqual(entropies[0], 0.0, places=5)
    
    def test_large_window(self):
        data = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
        window_size = 100
        calc = SlidingWindowEntropy(window_size)
        entropies = calc.compute(data)
        self.assertGreater(len(entropies), 0)
    
    def test_null_bytes(self):
        data = b"\x00\x01\x02\x03\x04"
        entropy = shannon_entropy(data)
        expected = math.log2(5)
        self.assertAlmostEqual(entropy, expected, places=5)


def run_tests():
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
