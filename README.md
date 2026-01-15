# entropy

so basically i was playing around with shannon entropy and got kinda obsessed with how you can measure "randomness" in a sequence. like if you have `AAAA` that's boring (entropy = 0), but if you have random gibberish that's spicy (entropy = like 8). and i wanted to see this across a sliding window so i could spot patterns and weird stuff.

ended up making this thing. it's not super complicated but it's fun to play with.

## what even is entropy tho

shannon entropy is just a number that tells you how unpredictable something is. measured in **bits**.

- `0` bits = boring af (like `AAAA`)
- `1` bit = two things equally likely (like `ABAB`)
- `2` bits = four things equally likely (like `ABCDABCD`)
- `8` bits = maximum chaos for a single byte

formula is: `-Σ(p_i × log₂(p_i))` where `p_i` is how often thing `i` shows up. yeah it's kinda neat actually.

## install

just... copy the files? there's no dependencies for the core stuff. if you want pretty plots add `matplotlib` but honestly the ascii stuff is fine.

```bash
cd entropy
python3 demo.py
```

that's it.

## how to use

### basic: just calculate entropy

```python
from entropy import shannon_entropy

entropy = shannon_entropy("hello world")
# returns a float like 4.43
```

### sliding window: the main thing

```python
from entropy import SlidingWindowEntropy

calc = SlidingWindowEntropy(window_size=8)
entropies = calc.compute("your data here or bytes")
# returns list of entropy values, one per window position
```

you can also get it with positions if that's useful:

```python
results = calc.compute_with_positions("data")
# gives you [(0, 1.23), (1, 1.45), ...]
```

### visualizations

ascii sparkline (compact):

```python
from visualization import ascii_sparkline

sparkline = ascii_sparkline(entropies, width=50)
print(sparkline)
# ▁▁▂▃▃▄▅▆▆▆▆▆▆▆▆▆▆▆▇████████████████
```

ascii heatmap (bigger):

```python
from visualization import ascii_heatmap

heatmap = ascii_heatmap(entropies, width=70, height=15, title="cool data")
print(heatmap)
```

if you have matplotlib (optional):

```python
from visualization import plot_entropy_with_matplotlib

plot_entropy_with_matplotlib(entropies, window_size=8)
# or save it: plot_entropy_with_matplotlib(entropies, 8, save_path="plot.png")
```

## examples

### spot the pattern

```python
data = "AAAA" * 10 + "ABCD" * 10 + random_stuff
calc = SlidingWindowEntropy(4)
entropies = calc.compute(data)
# boring part will have low entropy, then it goes up
```

### find weird stuff (anomalies)

```python
normal = "CCCC" * 30
anomaly = "NNNNN"
data = normal + anomaly + normal

calc = SlidingWindowEntropy(4)
entropies = calc.compute(data)

# middle part will have different entropy, you can spot it
weird_positions = [i for i, e in enumerate(entropies) if e > 0.5]
```

### compression correlation

data with low entropy = super compressible. data with high entropy = barely compresses. that's just how it works.

```python
import zlib

text = "AAAA" * 50  # very low entropy
entropy = shannon_entropy(text)  # ~0
compressed = zlib.compress(text.encode())
# compresses like 94%

random_text = "jkqmxpbvfghzwld" * 20  # high entropy
entropy = shannon_entropy(random_text)  # ~4.3
compressed = zlib.compress(random_text.encode())
# compresses like 80%
```

## files

- `entropy.py` - the actual computation
- `visualization.py` - pretty printing stuff
- `demo.py` - examples and tests basically
- `examples.py` - more real use cases
- `test_entropy.py` - verification (21 tests, all good)
- `README.md` - this thing

## how it works (kinda technical)

### step by step

1. you give it some data (text or bytes, doesn't matter)
2. it converts text to bytes if needed (utf-8)
3. for sliding window: it takes chunks of size `window_size`
4. for each chunk:
   - count how many of each byte/char
   - calculate probability of each one (count / total)
   - entropy = sum up `-(p_i × log₂(p_i))` for each symbol
5. returns list of these values

### example walkthrough

data: `"AABB"`
window: `2`

```
position 0: "AA" → A appears 2 times, p=1.0 → H = 0 bits
position 1: "AB" → A:1 B:1, both p=0.5 → H = 1 bit
position 2: "BB" → B appears 2 times, p=1.0 → H = 0 bits

result: [0, 1, 0]
```

### why sliding window

because then you see how entropy changes across your data. like if half is structured and half is random, you'll see it jump in the middle. useful for:

- finding where encryption starts
- spotting noise in signals
- detecting when data changes format
- basically anything where you want to know if stuff is "weird" at certain points

## running stuff

```bash
python3 demo.py              # all the demos
python3 examples.py          # practical examples
python3 test_entropy.py      # verify it works (21 tests)
```

## why i made this

idk honestly, thought it'd be cool to see entropy across time, and it kinda is? like you can visualize data structure in a neat way. also good for learning how entropy actually works instead of just reading about it.

## is it correct

yeah. tested against known values and the math checks out. 21 unit tests all pass. it's not like... cryptographically verified or anything, but for playing around / learning / analyzing stuff it's solid.

## notes

- window size has to be bigger than 0 (obviously)
- data has to be at least as long as the window
- this is single-pass, not streaming
- converts text to utf-8 bytes before processing
- no external dependencies for core stuff (matplotlib is optional)

## what's not here

- conditional entropy (multiple sequences)
- streaming mode for huge data
- fancy statistical tests
- gpu acceleration lol
- cli tool (could add though)

anyway that's it. have fun breaking it or whatever.
