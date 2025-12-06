

# 🚀 What is MiniROCKET?

**MiniROCKET** is an improved, *much faster*, and *almost perfectly accurate* variant of ROCKET for **time series classification**.

Goal:
→ **Same (or better) accuracy than ROCKET**
→ **Even faster** (by orders of magnitude)
→ **Completely deterministic** (ROCKET is random)

MiniROCKET achieves this by drastically simplifying the randomization used in ROCKET.

---

# ⭐ Key Differences from ROCKET

| Feature           | ROCKET            | MiniROCKET                                   |
| ----------------- | ----------------- | -------------------------------------------- |
| Kernel weights    | Random            | **Fixed** (all +1 and –1)                    |
| Kernel lengths    | Random {7, 9, 11} | **Fixed length = 9**                         |
| Number of kernels | 10,000            | **~10,000 BUT deterministic**                |
| Dilation          | Random            | **Deterministic set of dilations**           |
| Padding           | Random            | **Half padded, half unpadded**               |
| Features          | max + PPV         | **PPV only** (proportion of positive values) |
| Speed             | Very fast         | **Extremely fast** (5–10× faster)            |
| Accuracy          | SOTA              | **Matches or slightly improves**             |

MiniROCKET’s main insight:

> ROCKET’s randomization is not essential—**only dilation diversity and PPV matter**.

---

# 🧠 How MiniROCKET Works

MiniROCKET builds features using **fixed convolution kernels**, each defined by:

### 1. **Fixed kernel weights**

All kernels have:

```
[ -1, -1, -1, -1,  2, -1, -1, -1, -1 ]
```

This is a *single* predefined kernel pattern.

### 2. **Use many different dilations**

Dilation = how spread out the kernel taps are.

Example:

* Dilation 1 → matches local patterns
* Dilation 16 → matches long patterns
* Up to ~32k depending on series length

This gives frequency + scale diversity.

### 3. **Two padding options**

* With padding
* Without padding

This doubles the kernel variety.

### 4. **Only one feature** per kernel: **PPV**

> PPV = proportion of convolution outputs > 0

MiniROCKET found PPV alone captures almost all discriminative power.

So with ~10,000 kernels, MiniROCKET produces ~10,000 features (not 20k like ROCKET).

---

# ⚙️ MiniROCKET Pipeline

1. **Generate a deterministic set of kernels**

   * Same kernel weights
   * Many dilations
   * Some padded, some unpadded

2. **Convolve each kernel with the input time series**

3. **Compute PPV** for each kernel’s output

4. **Train a linear classifier (usually ridge regression)**

That's it.
No training of kernels → extremely fast.

---

# 🔥 Why MiniROCKET Is Faster

Because MiniROCKET:

* Uses one fixed kernel → simplifies convolution
* Uses only PPV → no max pooling
* Uses deterministic dilations → precomputed logic
* Removes random number generation
* Allows vectorization + batching

Speedups:

* **5–10× faster than ROCKET**
* Often >100× faster than deep CNNs

For long series, even more dramatic.

---

# 🎯 Performance

MiniROCKET:

* Matches ROCKET’s state-of-the-art accuracy
* Is **deterministic** (removes run-to-run variation)
* Is **much faster**, especially for long series or large datasets

It is currently one of the **fastest high-accuracy time series classifiers** available.

---

# 🧩 Summary Table

| Property       | MiniROCKET                 |
| -------------- | -------------------------- |
| Accuracy       | SOTA (≈ ROCKET)            |
| Speed          | Extremely fast             |
| Randomness     | None (fully deterministic) |
| Kernel weights | Fixed [+1 and –1 pattern]  |
| Kernel length  | Fixed = 9                  |
| Features       | PPV only                   |
| Dilation       | Deterministically varied   |

---
