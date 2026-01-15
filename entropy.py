import math
from collections import Counter
from typing import List, Tuple, Union


def shannon_entropy(data: Union[bytes, str]) -> float:
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if len(data) == 0:
        return 0.0
    
    frequencies = Counter(data)
    entropy = 0.0
    
    for count in frequencies.values():
        probability = count / len(data)
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


class SlidingWindowEntropy:
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
    
    def compute(self, data: Union[bytes, str]) -> List[float]:
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if len(data) < self.window_size:
            raise ValueError(
                f"Data length ({len(data)}) must be >= window_size ({self.window_size})"
            )
        
        entropies = []
        for i in range(len(data) - self.window_size + 1):
            window = data[i:i + self.window_size]
            entropy = shannon_entropy(window)
            entropies.append(entropy)
        
        return entropies
    
    def compute_with_positions(
        self, data: Union[bytes, str]
    ) -> List[Tuple[int, float]]:
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        results = []
        entropies = self.compute(data)
        for i, entropy in enumerate(entropies):
            results.append((i, entropy))
        
        return results


def compute_entropy_range(
    data: Union[bytes, str], window_size: int
) -> Tuple[float, float]:
    calc = SlidingWindowEntropy(window_size)
    entropies = calc.compute(data)
    
    if not entropies:
        return (0.0, 0.0)
    
    return (min(entropies), max(entropies))
