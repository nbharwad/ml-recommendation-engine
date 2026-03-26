# Quantization Learning Module

Author: NILESH BHARWAD

A comprehensive guide to quantization in Deep Learning / LLM / Model Optimization.

---

## 📦 Module 1: Foundations - Why LLMs Are Large & How Numbers Work

### 🎯 Objective
Understand why LLMs require massive memory and how computers represent numbers internally, establishing the foundation for understanding quantization.

### 🧠 Core Concepts

- **Parameters (Weights)**: The core building blocks of neural networks; stored in memory as the majority of LLM size
- **Neural Network Structure**: Billions of parameters arranged in layers with densely connected nodes
- **Integer Storage**: Uses fixed bits (e.g., 8-bit = 256 possible values, each representing powers of 2)
- **Floating Point Problem**: Infinite decimal places exist between any two numbers, but computers can only represent finite values
- **Precision Compromise**: Computers promise accuracy up to significant figures, then approximate rest

### ⚙️ Key Techniques

- **Integer (unsigned int)**: Discrete values, exact representation (e.g., uint8 = 0-255)
- **Floating Point**: Uses scientific notation in binary; divides bits into sign, exponent, significand

### 📊 Floating Point Formats

| Format | Bits | Exponent | Significand | Range | Precision |
|--------|------|----------|-------------|-------|-----------|
| float32 | 32 | 8 | 23 | ±3.4×10³⁸ | 7 sig figs |
| float16 | 16 | 5 | 10 | ±65504 | 3 sig figs |
| bfloat16 | 16 | 8 | 7 | ±3.4×10³⁸ | 2 sig figs |
| float8 | 8 | 4 | 3 | ±240 | ~1 sig fig |
| float4 | 4 | 2 | 1 | ±3 | ~1 sig fig |

### 🔍 Deep Insights

- Model parameters cluster near zero (most values between -0.1 and 0.1)
- This clustering is intentional: training rewards small parameters for better generalization
- Float distribution is non-uniform: more representable values near zero
- bfloat16 (Google) sacrifices precision for wide range—2 sig figs sufficient for LLMs

### ❓ Interview Questions

1. **Why do neural network parameters tend to cluster near zero?**
   - During training, optimization algorithms (like gradient descent) drive weights toward smaller values to improve generalization and prevent overfitting.

2. **What happens when you increment a floating point number at very large vs very small values?**
   - The "step size" changes: small values have tiny increments, large values have large jumps. This is because floats distribute precision non-uniformly.

3. **Why is bfloat16 popular for LLMs despite lower precision than float32?**
   - It has 8 exponent bits (wide range like float32) but only 7 significand bits. The wide range prevents overflow in large LLMs, and 2 significant figures is sufficient for acceptable model quality.

---

## 📦 Module 2: Floating Point Precision & Data Distributions

### 🎯 Objective
Understand how different floating point formats affect accuracy and why model parameters are ideal candidates for compression.

### 🧠 Core Concepts

- **Significand (Mantissa)**: Determines precision/accuracy of representation
- **Exponent**: Determines range of representable values
- **Parameter Distribution**: Most LLM weights fall in narrow range near zero
- **Outliers**: Rare but extreme values (e.g., 10x larger than typical) cause quantization problems

### 💻 Code Snippets

```python
# Visualizing parameter distributions (conceptual)
import numpy as np

# Most parameters cluster near zero
params = np.random.normal(0, 0.1, 1000000)  # Typical distribution
print(f"Range: {params.min():.2f} to {params.max():.2f}")
# Output: Range: -0.45 to 0.48 (most within -0.1 to 0.1)
```

### 📊 Precision vs Range Trade-off

| Format | Range | Precision | RAM Usage | LLMs Use? |
|--------|-------|-----------|-----------|-----------|
| float32 | Excellent | Excellent | 4 bytes | Original |
| float16 | Good | Good | 2 bytes | Yes |
| bfloat16 | Excellent | Low | 2 bytes | Yes (preferred) |
| float8 | Limited | Very Low | 1 byte | Emerging |
| float4 | Very Limited | Minimal | 0.5 bytes | Rare |

### 🔍 Deep Insights

- Float32 promises 7 significant figures; most applications don't need this
- bfloat16 sacrifices precision but maintains range—2 sig figs works for LLMs
- Model parameters don't need full float32 precision because:
  1. Values cluster in narrow range
  2. Neural networks are inherently approximate
  3. Small errors get "absorbed" by the network's non-linearity

### ❓ Interview Questions

1. **Why can we use lower precision for LLMs than for scientific computing?**
   - LLMs don't require exact arithmetic; they learn statistical patterns. Small errors in weights often don't significantly change model behavior because of redundancy in the network.

2. **What would happen if we trained a model directly in float4?**
   - The limited range and precision would cause severe training instability. Gradients would overflow or underflow constantly.

---

## 📦 Module 3: Quantization Fundamentals

### 🎯 Objective
Understand what quantization is, how it compresses values from large to small ranges, and why simple rounding fails for LLMs.

### 🧠 Core Concepts

- **Quantization**: Process of mapping values from large range to smaller range (lossy compression)
- **Dequantization**: Converting back to approximate original values using stored scale
- **Round-to-Nearest**: Basic quantization method—maps to closest representable value
- **Quantization Error**: Difference between original and dequantized value (the "loss")

### ⚙️ Key Techniques

- **Naive Rounding**: Simply round float16→float8→float4
- **Scale-based Quantization**: Divide by scale factor, round, multiply back
- **Block-wise quantization**: Apply quantization to small groups of parameters (32-256)

### 💻 Code Snippets

```python
# Naive rounding (fails for LLMs)
def naive_round(value, bits):
    qmax = 2 ** bits - 1
    # This doesn't work well—uses full range regardless of data distribution
    scale = 1.0  # Wrong approach!
    quantized = round(value / scale * qmax)
    return quantized * scale / qmax
```

### 🔍 Deep Insights

- **Why naive rounding fails**: Using full float range (e.g., -3 to 3 for float4) when data is in narrow range (e.g., -0.89 to 0.16) wastes precision
- **The outlier problem**: Single outlier (e.g., value 10) forces entire range to expand, destroying precision for all other values
- **Solution**: Block-wise quantization isolates outliers to small groups

### ❓ Interview Questions

1. **Why is quantization considered "lossy"?**
   - Because we're mapping infinite real values to finite discrete values. Some information is permanently lost—we can't recover the exact original.

2. **Why can't we quantize an entire LLM at once?**
   - Outliers in one part of the model would force the entire model into a wider range, destroying precision everywhere. Block-wise quantization localizes outlier damage.

---

## 📦 Module 4: Quantization Methods - Symmetric vs Asymmetric

### 🎯 Objective
Master symmetric and asymmetric quantization methods, including when each is preferable.

### 🧠 Core Concepts

- **Symmetric Quantization**: Maps data symmetrically around zero; range is [-max, +max]
- **Asymmetric Quantization**: Maps data to [min, max] without requiring symmetry around zero
- **Scale Factor**: The divisor used to normalize values into quantized range
- **Zero Point**: Offset used in asymmetric quantization to handle non-zero data ranges

### 💻 Code Snippets

```python
import math

# Symmetric quantization
def symmetric_quantize(values, bits=4):
    vmax = max(abs(v) for v in values)  # Find largest absolute value
    qmax = 2 ** (bits - 1) - 1  # e.g., 7 for 4-bit
    scale = vmax / qmax
    
    quantized = [round(v / scale) for v in values]
    return {'values': quantized, 'scale': scale}

def symmetric_dequantize(quantized):
    return [v * quantized['scale'] for v in quantized['values']]

# Asymmetric quantization
def asymmetric_quantize(values, bits=4):
    vmax = max(values)
    vmin = min(values)
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    scale = (vmax - vmin) / (qmax - qmin)
    zero = qmin - round(vmin / scale)
    
    quantized = [round(v / scale + zero) for v in values]
    return {'values': quantized, 'scale': scale, 'zero': zero}

def asymmetric_dequantize(quantized):
    return [quantized['scale'] * (v - quantized['zero']) 
            for v in quantized['values']]
```

### 📊 Comparison

| Method | Precision | Range | Use Case | Complexity |
|--------|-----------|-------|----------|------------|
| Symmetric | Lower | [-max, +max] | Zero-centered data | Simple |
| Asymmetric | Higher | [min, max] | Biased distributions | Moderate |

### 🔍 Deep Insights

- **Symmetric**: Uses less memory (no zero-point storage); best when data is roughly centered at zero
- **Asymmetric**: Better accuracy (~10% less error) when data distribution is skewed; stores extra zero_point overhead
- **Empirical results**: Asymmetric typically wins for LLMs because weight distributions aren't perfectly symmetric

### ❓ Interview Questions

1. **Why does asymmetric quantization usually give better accuracy?**
   - It doesn't waste bits representing empty space. If data ranges from -0.89 to 0.16, symmetric would use range [-0.89, +0.89], leaving lots of unused positive space. Asymmetric uses [-0.89, 0.16] efficiently.

2. **What is the "zero point" in asymmetric quantization?**
   - An offset value stored alongside the scale. During dequantization: `original ≈ scale × (quantized_value - zero_point)`. This allows representing values that don't include zero in the center of the range.

---

## 📦 Module 5: Practical Quantization & Block Size Trade-offs

### 🎯 Objective
Understand how quantization is applied in practice and the trade-offs involved in block size selection.

### 🧠 Core Concepts

- **Block Size**: Number of parameters quantized together (typically 32-256)
- **Per-block Storage**: Each block stores its own scale (and zero-point for asymmetric)
- **Overhead Calculation**: Block metadata adds memory overhead
- **Outlier Containment**: Small blocks prevent outliers from affecting entire model

### ⚙️ Key Techniques

- **Small blocks** (32-64): Better accuracy, higher overhead
- **Large blocks** (128-256): Lower overhead, more error from value range variation

### 🔍 Deep Insights

- **Why block size matters**: 
  - Model parameters vary across layers and even within layers
  - A single global scale can't capture all this variation
  - Smaller blocks = tighter scales = less error
  
- **The overhead trade-off**:
  ```
  Overhead per block = 2 values (scale + zero_point)
  Total overhead = (2 × num_params) / block_size × bytes_per_value
  ```

- **Real-world choice**: Most tools use 128 as default; tune based on accuracy requirements

### ❓ Interview Questions

1. **What happens if block size is too large?**
   - Average range of values in the block increases, forcing a larger scale. This increases quantization error because smaller values get mapped to fewer discrete levels.

2. **What happens if block size is too small?**
   - Overhead dominates—too much metadata stored relative to actual parameter data. Memory savings decrease.

---

## 📦 Module 6: Measuring Quantization Quality Loss

### 🎯 Objective
Learn multiple methods to evaluate quantization impact and understand their trade-offs.

### 🧠 Core Concepts

- **Perplexity**: Measures how "surprised" the model is by correct tokens (lower = better)
- **KL Divergence**: Measures how different probability distributions are (lower = more similar)
- **Benchmarking**: Task-specific accuracy measurements
- **Qualitative Testing**: Direct interaction with the model

### 📊 Comparison of Measurement Methods

| Method | What It Measures | Pros | Cons |
|--------|------------------|------|------|
| Perplexity | Confidence in correct token | Simple number | Ignores wrong tokens |
| KL Divergence | Distribution similarity | Full picture | Hard to interpret |
| Benchmark | Task accuracy | Real-world relevance | Requires task-specific setup |
| Qualitative | User perception | Most realistic | Not rigorous |

### 📊 Example Results (Qwen3.5 9B)

| Format | Perplexity | KL Divergence | Speed (H100) |
|--------|------------|---------------|--------------|
| bfloat16 | 8.19 | baseline | 107 tok/s |
| 8-bit symmetric | 8.19 (+0.1%) | 0.0008 | 142 tok/s |
| 4-bit asymmetric | 8.56 (+4.6%) | 0.059 | 176 tok/s |
| 4-bit symmetric | 8.71 (+6.4%) | 0.068 | 177 tok/s |
| 2-bit asymmetric | 66.1 (+707%) | 2.14 | 167 tok/s |

### 🔍 Deep Insights

- **8-bit: nearly lossless** — negligible quality degradation
- **4-bit: acceptable** — 5-10% degradation typical, widely used in production
- **2-bit: usually too aggressive** — severe quality collapse
- **Speed gains**: Smaller formats are faster due to less memory bandwidth needed

### ❓ Interview Questions

1. **Why does perplexity sometimes not capture the full picture of quantization impact?**
   - Perplexity only looks at the probability assigned to the "correct" token. The entire probability distribution could shift, making the model less useful even if the correct token probability stays similar.

2. **Why does 2-bit quantization fail so dramatically?**
   - With only 4 discrete values (for 2-bit), there's insufficient granularity to represent the weight variations the model needs. The network essentially loses its ability to compute meaningful outputs.

---

## 📦 Module 7: Production Considerations & Performance

### 🎯 Objective
Understand real-world quantization deployment, performance characteristics, and toolchain.

### 🧠 Core Concepts

- **GGUF Format**: llama.cpp's optimized format for quantized models
- **ONNX Quantization**: Cross-platform standard
- **TensorRT Quantization**: NVIDIA's production solution
- **Dynamic vs Static Quantization**: When to compute scales at runtime vs. once

### 💻 Code Snippets

```bash
# Using llama.cpp to quantize
llama-quantize input.gguf output.gguf Q4_0  # 4-bit symmetric
llama-quantize input.gguf output.gguf Q4_1  # 4-bit asymmetric
llama-quantize input.gguf output.gguf Q8_0  # 8-bit symmetric

# Available formats in llama.cpp:
# Q8_0, Q4_1, Q4_0, Q3_K, Q2_K, etc.
```

### 📊 Production Speed (Tokens/Second)

| Hardware | bfloat16 | 8-bit | 4-bit |
|----------|----------|-------|-------|
| M1 Max | 19 | 32 | 43 |
| H100 GPU | 107 | 142 | 177 |

### 🔍 Deep Insights

- **Why faster despite dequantization?**: Memory bandwidth is usually the bottleneck. Smaller models fit in cache, reducing DRAM access.
- **Dynamic quantization**: Scale computed at runtime; better for variable data but slower
- **Static quantization**: Scale pre-computed; faster but requires representative dataset

### ❓ Interview Questions

1. **Why does quantized inference often run faster than full precision?**
   - The GPU spends most time waiting for data from memory. Smaller models = less data to transfer = more time computing.

2. **What is the difference between PTQ and QAT?**
   - PTQ (Post-Training Quantization): Quantize after training—simpler but may lose accuracy. QAT (Quantization-Aware Training): Train with quantization simulated—better accuracy but requires retraining.

---

## 🧩 FINAL SECTION

### 🧠 End-to-End Summary

**Quantization as a System:**

1. **Problem**: LLMs are too large (billions of parameters × 4 bytes = TBs of RAM)
2. **Solution**: Compress weights from 32-bit floats to 8/4/2-bit integers
3. **Key Insight**: Most model parameters cluster near zero—precise range isn't needed
4. **Challenge**: Outliers require block-wise quantization to contain damage
5. **Result**: 4x smaller, 2x faster, ~5-10% quality loss at 4-bit

**The Quantization Pipeline:**
```
FP32 Weights → Block into groups → Find scale (and zero-point) → 
Round to integer → Store (values + scale + zero-point) → 
Dequantize at runtime → Inference
```

### 🚀 Advanced Topics

| Method | Description | Best For |
|--------|-------------|----------|
| **AWQ (Activation-aware)** | Scales weights by activation importance | Ultra-low bit (2-3 bit) |
| **GPTQ** | Per-column quantization with Hessian | GPU-efficient |
| **SmoothQuant** | Moves difficulty from weights to activations | Very large models |
| **KV-cache Quantization** | Quantizes attention key/value cache | Long context inference |
| **QAT (Quantization-aware Training)** | Simulates quantization during training | Best accuracy |

### ⚠️ Common Mistakes

1. **Quantizing entire model at once** → Outliers destroy quality
2. **Using symmetric when asymmetric fits better** → Wastes precision on empty range
3. **Going too aggressive (2-bit)** → Complete model collapse
4. **Ignoring per-layer sensitivity** → Some layers need more bits than others
5. **Not evaluating on actual use-case** → Benchmarks may not reflect your task

### ❓ Final Interview Question

**"If you had a 100B parameter model and needed to run it on a single GPU with 24GB VRAM, how would you approach making it work?"**

Answer: Use 4-bit quantization (roughly 50GB → 12.5GB), plus:
- Block-wise quantization (128-256) for outlier handling
- Consider AWQ or GPTQ for better accuracy
- Use bfloat16 for activations if possible
- Profile to find most sensitive layers (may need higher precision)
- Consider parameter pruning if still too large
