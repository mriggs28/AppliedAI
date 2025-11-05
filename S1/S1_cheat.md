# 🧠 **Quiz 1 Cheat Sheet — ML & Representation Learning**

---

## 🧩 **1. Supervised vs Unsupervised Learning**

| Type             | Data           | Goal                      | Examples                   |
| ---------------- | -------------- | ------------------------- | -------------------------- |
| **Supervised**   | Labeled (X, y) | Learn mapping ($f: X \to y$)  | Regression, Classification |
| **Unsupervised** | Unlabeled (X)  | Find structure / patterns | Clustering (K-Means), PCA  |

---

## ⚙️ **2. Pre-processing**

**Why:** Ensures fair comparison between features, avoids bias in distance-based algorithms.  
**Common methods:**

* **Z-score:** $x' = \dfrac{x - \mu}{\sigma}$ → mean 0, var 1  
* **Min–Max:** $x' = \dfrac{x - x_{\min}}{x_{\max}-x_{\min}}$ → scaled to $[0,1]$

✅ **Fit only on training set** → prevents **data leakage**

---

## 📏 **3. Distance Metrics**

| Metric                | Formula                             | Notes                             |
| --------------------- | ----------------------------------- | --------------------------------- |
| **L1 (Manhattan)**    | $\displaystyle \sum_i |x_i - y_i|$                | Robust to outliers                |
| **L2 (Euclidean)**    | $\displaystyle \sqrt{\sum_i (x_i - y_i)^2}$       | Sensitive to large values         |
| **Cosine similarity** | $\displaystyle \frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$              | Angle-based; ignores magnitude    |
| **Mahalanobis**       | $\displaystyle \sqrt{(\mathbf{x}-\mathbf{y})^\top \Sigma^{-1} (\mathbf{x}-\mathbf{y})}$       | Accounts for feature correlations |

---

## 🧮 **4. k-Nearest Neighbors (k-NN)**

* Predict label = majority among nearest $k$ samples.  
* **Small $k$:** low bias, high variance → overfits  
* **Large $k$:** high bias, low variance → smoother  
* Suffers from **curse of dimensionality** → distances lose meaning in high-D space.

---

## 🔁 **5. Cross-Validation (CV)**

* Split train set into **$k$ folds** → train on $k-1$, validate on 1 → average performance.  
* Used for **hyperparameter tuning** without touching the test set.

---

## 📊 **6. Confusion Matrix & Metrics**

|              | Pred + | Pred – |
| ------------ | ------ | ------ |
| **Actual +** | TP     | FN     |
| **Actual –** | FP     | TN     |

Formulas:

* Accuracy = $\dfrac{TP+TN}{\text{Total}}$  
* Precision = $\dfrac{TP}{TP+FP}$  
* Recall = $\dfrac{TP}{TP+FN}$  
* F1 = $2\cdot\dfrac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$

---

## ✴️ **7. Support Vector Machines (SVM)**

* Find hyperplane that **maximizes margin** between classes.  
* Constrained formulation (hard-margin) example: $$\begin{aligned}
&\min_{w,b}\ \tfrac{1}{2}\|w\|^2 \\
&\text{s.t.}\quad y_i (w^\top x_i + b) \ge 1\quad\forall i
\end{aligned}$$
* Decision rule: $w^\top x + b = 0$  
* **Support vectors:** points closest to boundary — define it.

---

## 🌌 **8. The Kernel Trick**

Enable nonlinear SVMs via **implicit feature mapping**:  
$$K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$$

Common kernels:
* Linear: $K(x,y)=x^\top y$  
* Polynomial: $K(x,y)=(x^\top y + c)^d$  
* RBF (Gaussian): $K(x,y)=\exp(-\gamma \|x-y\|^2)$

---

## 🔀 **9. Multi-Class SVM**

| Strategy              | \#Classifiers | Concept                   |
| --------------------- | -------------: | ------------------------- |
| **One-vs-Rest (OvR)** | N             | Each class vs all others  |
| **One-vs-One (OvO)**  | $\dfrac{N(N-1)}{2}$     | Pairwise class separation |

**OvO detail:** train one classifier for every pair of classes. For $N$ classes there are $\dfrac{N(N-1)}{2}$ unique pairs.

---

## 🧠 **10. Word Representations**

### ❌ One-hot encoding
* Sparse, high-dimensional  
* No semantic meaning

### ✅ Distributed embeddings
* Dense, low-dimensional (50–300D)  
* Capture word meaning via context

---

## 🔡 **11. Word2Vec Architectures**

| Model         | Predicts            | Notes                     |
| ------------- | ------------------- | ------------------------- |
| **Skip-Gram** | Context from target | Good for rare words       |
| **CBOW**      | Target from context | Faster for frequent words |

---

## ⚖️ **12. Embedding Dimensionality**

* Too low → poor representation  
* Too high → overfitting, slower training  
➡️ Usually 100–300 dimensions

---

## ⚙️ **13. Random Indexing (RI)**

* Assigns each word a **random sparse vector**  
* Builds context vectors by summing neighbors  
* Fast, memory-efficient alternative to Word2Vec

---

## 💬 **14. Measuring Similarity Between Embeddings**

* **Cosine similarity:** $\dfrac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ → most common  
* **Euclidean distance:** geometric closeness  
* **Dot product:** used directly in neural models

---

✅ **Summary Mnemonics**

* **S vs U:** labels vs patterns  
* **Scaling:** don’t leak info!  
* **k-NN:** watch $k$ + high dimensions  
* **CV:** reuse training only  
* **SVM:** margin, kernel, support vectors  
* **Embeddings:** dense, contextual, cosine-based
