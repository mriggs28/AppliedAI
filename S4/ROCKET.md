
# 🚀 What is ROCKET?

**ROCKET** stands for **RandOm Convolutional KErnel Transform**.
It is a **very fast** and **high-accuracy** method for **time series classification**.

At a high level, ROCKET does two things:

1. **Transforms time series** using a huge number of **random 1D convolutional kernels**
2. **Trains a simple linear classifier** (usually ridge regression or logistic regression) on the extracted features

Despite its simplicity, ROCKET achieves **state-of-the-art accuracy** while being **far faster** than deep learning or traditional time-series classifiers.

---

# 🧠 Why ROCKET Works

Unlike CNNs that *learn* convolutional kernels, ROCKET:

### ✓ Generates thousands of kernels *randomly*

Each kernel has random:

* length (7, 9, 11)
* weights (mean-centered Gaussian)
* bias
* dilation (sampled exponentially)
* padding (sometimes applied)

This gives **massive diversity**, letting kernels accidentally align with real patterns.

---

# ⚙️ Feature Extraction (Key Innovation)

For each random kernel, ROCKET computes **two features** from its convolution output (feature map):

### 1. **Max value**

Equivalent to global max pooling → detects the strongest match.

### 2. **PPV (Proportion of Positive Values)**

> The **most important feature** in ROCKET.
> It measures how often the kernel produces positive activations—i.e., how *prevalent* a pattern is.

These features are extremely cheap to compute.
With 10,000 kernels → **20,000 features per time series**.

---

# 🏎️ Speed

ROCKET is extremely fast because:

* Kernel weights are *not learned*
* Convolutions use very small kernels (≤ 11)
* Feature extraction is simple
* Final classifier is linear

Its runtime is **O(k · n · l)**
(k = kernels, n = number of samples, l = length of series)

Example results from the paper:

* Trains the largest UCR dataset in **6 minutes** (vs. hours or days for others)
* Scales to **1 million time series in ~1 hour**
* A tiny variant trains on 1 million series in **< 1 minute**

---

# 🎯 Accuracy

On the UCR benchmark archive, ROCKET:

* Achieves **state-of-the-art accuracy**
* Matches or beats:

  * HIVE-COTE
  * TS-CHIEF
  * InceptionTime
* Despite using only a single linear layer

Its performance comes from the **massive variety** of random convolutions and especially the **PPV feature**.

---

# 📦 Summary

| Property       | ROCKET                                                  |
| -------------- | ------------------------------------------------------- |
| Model type     | Random convolution transform + linear classifier        |
| Kernels        | Thousands, fully random, varied dilation/length/padding |
| Features       | Max + PPV                                               |
| Accuracy       | State-of-the-art on UCR datasets                        |
| Speed          | Extremely fast, linear scaling, CPU-friendly            |
| Scalability    | 1M+ time series easily                                  |
| Key advantages | Simplicity, speed, accuracy                             |

