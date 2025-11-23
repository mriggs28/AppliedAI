

# ✅ **1. Local vs Distributed Representations**

### **Local (One-Hot) Representation**

* Each concept = a vector with one “1” and the rest “0”.
* Example:

  * Cat = [1, 0, 0, 0]
  * Dog = [0, 1, 0, 0]
* **Properties**:

  * Mutually orthogonal.
  * No notion of similarity.
  * Very inefficient in high dimension.

### **Distributed / High-Dimensional Representation**

* Each concept = a long dense vector (e.g., 10,000 dimensions) of random values like ±1.
* Similar concepts can share structure, enabling:

  * **Robustness**: random noise barely changes similarity.
  * **Compactness**: you can store many patterns via superposition.
  * **Compositionality**: vectors can be combined algebraically (binding, permutation, superposition).

---

# ✅ **2. Random HD Vectors Are Nearly Orthogonal**

In a high-dimensional space (d ≈ 10,000), two random ±1 vectors:

$
\mathbb{E}[\text{cosine}(x,y)] = 0,\quad \text{Var}\approx\frac{1}{\sqrt{d}}
$

So cosine similarity ≈ 0 → almost orthogonal → nearly independent.

**Why useful?**

* Each vector behaves like a unique symbol.
* You can reliably separate many items.

---

# ✅ **3. Core HDC/VSA Operations**

### **Superposition**

* Purpose: *store multiple items together*.
* Operation: elementwise addition
  $
  H = A + B + C
  $
* Later retrieve using cosine similarity:
  ( $ \text{sim}(A, H) \approx \text{high} $).

### **Binding**

* Purpose: *associate two items* (e.g., role–filler pairs).
* Typical operation: elementwise multiplication
  $
  P = A \odot B
  $
* Properties:

  * Reversible: ( B = A \odot P )
  * Creates a vector dissimilar to A and B separately → avoids interference.

### **Similarity**

* Measure to compare vectors: cosine similarity or dot-product.
* Works well because HD vectors have predictable distributions.

---

# ✅ **4. Why Random ±1 Vectors Are Used**

* **Easy to generate**
* **Space-efficient**
* **Computationally cheap**: operations = adds & multiplies
* **Balanced**: equal number of +1 and –1 → zero-mean → no bias
* **Independent dimensions** → high capacity encoding

---

# ✅ **5. Centroid-Based HD Classification**

Training-free classification:

1. For each class, sum all its vectors:
   $
   C_k = \sum_{i \in \text{class k}} x_i
   $
2. Normalize centroids.
3. For a test vector x, compute cosine similarity to each centroid.
4. Predict class with highest similarity.

**Advantages**:

* No gradient descent.
* Extremely fast.
* Very robust.

---

# ✅ **6. n-grams vs Hypervectors**

### **n-grams**

* Represent sequences using one-hot vectors for words.
* Curse of dimensionality: number of possible n-grams blows up.

### **Hypervectors**

* Encode sequence structure via:

  * permutation (shifting)
  * binding
  * superposition
* Very compact and robust to noise.

---

# ✅ **7. Benefits of HD Representations**

* **Compositionality**: combine symbols algebraically.
* **Robustness**: noise affects only a few dimensions.
* **Scalability**: more dimensions = more capacity.
* **Symbolic reasoning**: supports role–filler, structures, sequences.

---

# 🧠 **SELF-ORGANIZING MAPS (SOMs)**

# ✅ **8. What is a Self-Organizing Map?**

SOM = unsupervised neural network that maps high-dimensional data → 2D grid while preserving topology.

Meaning:

* Nearby neurons represent similar data.
* Far neurons represent dissimilar data.

---

# ✅ **9. SOM Learning Algorithm**

For each sample:

1. **Compute BMU (Best Matching Unit)**
   Neuron whose weight vector is closest to input:
   $
   \text{BMU} = \arg\min_j |x - w_j|
   $

2. **Update BMU and neighbors**:
   $
   w_j(t+1) = w_j(t) + \eta(t),h_{j,\text{BMU}}(t),(x - w_j(t))
   $

3. **Decay**:

   * learning rate η(t)
   * neighborhood radius σ(t)

---

# ✅ **10. Neighborhood Function**

Typically Gaussian:

$
h_{j,\text{BMU}}(t) =
\exp\left(-\frac{d(j,\text{BMU})^2}{2\sigma(t)^2}\right)
$

Purpose:

* Spread learning across neighbors.
* Smooth the map early.
* Refine it later as σ shrinks.

---

# ✅ **11. Effects of SOM Hyperparameters**

* **Learning rate too high** → oscillations; unstable map.
* **Learning rate too low** → very slow convergence.
* **Radius decays too quickly** → poor global organization; map fragments.
* **Radius too large for too long** → everything becomes similar.

---

# ✅ **12. Grid Resolution**

* Larger grid = more neurons → higher map resolution.
* Reduces quantization error.
* Increases training time.

---

# ✅ **13. Biological Inspiration**

SOM mimics cortical maps:

* **Competition**: neurons compete to respond.
* **Cooperation**: neighbors also update.
* **Lateral inhibition**: “winner inhibits neighbors” mechanism.

---

# 🤖 **MLP + BACKPROPAGATION**

# ✅ **14. MLP Structure**

* Input layer → passes features.
* Hidden layers → nonlinear transformations.
* Output layer → prediction (e.g., Softmax).

Connections:
$
y = f(W_2 , f(W_1 x + b_1) + b_2)
$

---

# ✅ **15. Backpropagation Overview**

Goal: minimize loss L.

Steps:

1. Forward pass → compute predictions.
2. Compute loss L.
3. Use chain rule to compute gradients:
   $
   \frac{\partial L}{\partial W}
   $
4. Gradient descent update:
   $
   W \leftarrow W - \eta \frac{\partial L}{\partial W}
   $

---

# ✅ **16. Activation Functions**

| Activation  | Range   | Advantages                 |
| ----------- | ------- | -------------------------- |
| **Sigmoid** | (0, 1)  | Probabilistic output       |
| **tanh**    | (-1, 1) | Zero-centered              |
| **ReLU**    | [0, ∞)  | Avoids vanishing gradients |

---

# ✅ **17. Softmax**

Turns logits z into probabilities:

$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$

Used for multiclass classification.

---

# ✅ **18. Learning Rate Influence**

* **Too high** → divergence, oscillations.
* **Too low** → very slow learning or stuck in bad minima.

---

# ✅ **19. Weight Initialization**

Goal: avoid zero gradients & symmetry.

* **Xavier**: for tanh/sigmoid
* **He**: for ReLU

Ensures variance does not explode or vanish through layers.

---

# ✅ **20. Common Loss Functions**

* **Cross-entropy**: classification
* **MSE**: regression
* **Negative log-likelihood** = cross-entropy with Softmax

---

# ✅ **21. ReLU vs Sigmoid**

ReLU:

* No saturation for positive inputs.
* Prevents vanishing gradients.
* Faster convergence.

Sigmoid:

* Smooth but saturates → vanishing gradients.
* Used sometimes in output layers (binary classification).

---

# ✅ **22. Overfitting & Early Stopping**

* Too many epochs → model memorizes data.
* Early stopping monitors validation loss and stops when it increases.
* Prevents overfitting.

---

# ✅ **23. Avoiding Vanishing/Exploding Gradients**

Methods:

* ReLU activations
* Xavier/He initialization
* Batch normalization
* Gradient clipping

---

# 🎓 **LEARNING PARADIGM ASSOCIATIONS**

| Model           | Paradigm      | Explanation                         |
| --------------- | ------------- | ----------------------------------- |
| **SOM**         | Unsupervised  | Learns topology from unlabeled data |
| **MLP**         | Supervised    | Needs labeled data for backprop     |
| **HD Centroid** | Training-free | Just accumulates vectors            |

---

# ✅ **24. Conceptual Differences Between SOM and MLP**

* **SOM**:

  * Creates a map
  * Learns structure/topology
  * Unsupervised

* **MLP**:

  * Learns class boundaries
  * Requires labels
  * Optimizes a loss

---

# ✅ **25. Confusion Matrix Interpretation**

Confusion matrix M₍i,j₎:

* Rows = true labels
* Columns = predicted labels
* Diagonal = correct preds
* Off-diagonal = errors

Useful for analyzing:

* Which classes are confused
* Precision/recall
* Accuracy

---

# 🧠 **26. Advantages of HD Representations**

* **Scalability**: add more items easily.
* **Noise tolerance**: small corruption doesn't change cosine similarity much.
* **Explainability**: operations (binding/superposition) are interpretable.

---

# 🤝 **27. How HD, SOM, and Backprop Complement Each Other**

| Method             | Paradigm      | Strength                 |
| ------------------ | ------------- | ------------------------ |
| **HD Computing**   | Training-free | Fast, robust, symbolic   |
| **SOM**            | Unsupervised  | Topological mapping      |
| **MLP + Backprop** | Supervised    | Learn complex boundaries |

They cover the three fundamental learning paradigms.

---

# 🧮 **28. Numeric Examples You Must Know**

### **Cosine similarity**

$
\cos(x,y) = \frac{x\cdot y}{|x| |y|}
$

Example:
x = [1, -1, 1], y = [1, 1, -1]
→ dot = 1 - 1 - 1 = -1
→ cosine ≈ -1 / 3 = –0.33

---

### **SOM BMU Update Example**

If:

* w = [1, 1]
* x = [3, 2]
* learning rate = 0.5
* h = 1 (BMU)

Update:

$
w \leftarrow w + 0.5 ([3,2]-[1,1]) = [1,1] + 0.5[2,1] = [2,1.5]
$

---

### **Backprop Weight Update Example**

Simple rule:

$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$

If:

* gradient = 0.2
* η = 0.1

Then:
$
w \leftarrow w - 0.1 \times 0.2 = w - 0.02
$

