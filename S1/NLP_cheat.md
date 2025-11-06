

## 🧭 1. Overview and Objectives

The presentation introduces **Natural Language Processing (NLP)** — a field of AI that enables machines to understand, interpret, and generate human language.
It explores:

* Common NLP tasks
* Word embedding techniques
* Traditional models (N-gram, LSA, Random Indexing)
* Neural network models (Word2Vec)
* Evaluation of word embeddings
* Practical applications and comparisons

---

## 🗣️ 2. Why NLP?

Language is described as the first great product of human cognition.
It allows:

* Collaboration and communication
* Expression of emotions, thoughts, and decisions
* Information storage through narratives

Thus, NLP is the bridge enabling **machines to communicate with humans**.

---

## 📋 3. Common NLP Tasks

1. **Text Classification** – Assigning texts to predefined categories.

   * Examples:

     * *Genre classification* (news, legal, scientific)
     * *Language identification*
     * *Spam detection*
2. **Information Extraction** – Identifying entities and relationships.
3. **Machine Reading/Translation** – Automatic understanding and translation of text.

All these require **understanding meaning and semantics**.

---

## 🔍 4. Meaning and Semantics

* NLP systems need to capture *semantics* — the meaning of words and sentences.
* Challenges: how to **represent meaning numerically**.
* The document mentions Ellie Pavlick’s work on **symbols and grounding** in large language models (LLMs), emphasizing constructing meaning from experience.

---

## 🔢 5. Simple Models: N-Gram Language Models

* A **language model** estimates the probability of a sequence of words.
* An **n-gram** is a contiguous sequence of n items (letters, syllables, or words).
* Used for:

  * Language identification
  * Text prediction
  * Statistical modeling of text patterns

While simple, n-gram models are limited in capturing long-range dependencies.

---

## 🧩 6. Word Embedding

Word embeddings represent words as **dense vectors** in continuous space.
These vectors encode **semantic relationships** between words:

* Similar words (e.g., “king” and “queen”) have similar vector representations.
* Relationships like *“Madrid – Spain + France ≈ Paris”* emerge.

Embeddings thus bridge discrete language with continuous mathematics.

### Key Ideas:

* Context defines meaning (“you shall know a word by the company it keeps”).
* Each word is mapped to a vector learned from co-occurrence in large corpora.
* Distances between vectors approximate **semantic similarity**.

---

## ⚖️ 7. Evaluation Methods for Word Embeddings

How to judge embedding quality:

### **Extrinsic Evaluation**

* Embeddings are used in downstream tasks.

  * Examples: POS tagging, Named Entity Recognition (NER), sentiment analysis.
* Measures **task performance** improvements.

### **Intrinsic Evaluation**

* Directly evaluates embeddings’ linguistic structure.

  * **Relatedness**: Compare cosine similarity vs. human judgments.
  * **Analogy**: Solve problems like *man:king :: woman:queen*.
  * **Categorization**: Cluster words and check category purity.
  * **Selectional preference**: Evaluate how typical a word is for a verb (e.g., *people eat apples*, not *apples eat people*).
  * **Coherence**: Check whether nearby words in embedding space are semantically related.

---

## 🧱 8. Word Embedding Models — Taxonomy

Two major approaches:

### 1. **Connectionist (Neural Network–based) Models**

* Represent knowledge as **weighted connections** between neurons.
* Example: **Word2Vec**

  * **Skip-Gram model**: Predicts context words given a target word.
  * **CBOW (Continuous Bag of Words)**: Predicts the target word given context words.
* Captures linguistic regularities through vector arithmetic.

### 2. **Distributional (Corpus-based) Models**

* Based on co-occurrence patterns in text.
* Famous principle: “You shall know a word by the company it keeps.”
* Examples: **Latent Semantic Analysis (LSA)** and **Random Indexing (RI)**.

---

## 🧮 9. Latent Semantic Analysis (LSA)

* Developed by **Landauer & Dumais (1997)**.
* Starts with a **term-document frequency matrix** (rows = words, columns = documents).
* Applies **Singular Value Decomposition (SVD)** to reduce dimensionality (usually to ~300).
* Captures higher-order semantic similarities (e.g., *boat* and *ship* become closer).
* Achieved ~51% on TOEFL synonym test — comparable to human-level vocabulary knowledge.

---

## 🎲 10. Random Indexing (RI)

* Alternative to LSA — **computationally simpler** (no SVD).
* Assigns random sparse vectors to documents.
* Words’ vectors are updated by summing the vectors of documents they appear in.
* Despite randomness, RI preserves semantic similarity remarkably well.
* TOEFL score ~52%.

### Improvement: **Random Indexing with Permutations (RP)**

* Uses *word windows* instead of documents as context.
* Adds permutations to encode **word order**.
* Achieved ~78% on TOEFL — approaching Word2Vec’s performance.

---

## 🧠 11. Neural Models: Word2Vec

* Developed by **Mikolov et al. (2013)**.
* Learns embeddings by predicting context (Skip-gram or CBOW).
* Produces powerful, generalizable semantic representations.
* Example:

  * vec(“Madrid”) - vec(“Spain”) + vec(“France”) ≈ vec(“Paris”).

These embeddings form the foundation for modern **transformer-based LLMs**.

---

## 🧪 12. Practical Assignment

The document concludes with a hands-on exercise:

* Use **Word2Vec** and **Random Indexing** on large text corpora.
* Perform **TOEFL synonym tasks**.
* Compare performance, accuracy, and speed between the two methods.

---

## 📚 13. References

The lecture cites major works in NLP and semantics:

* Russell & Norvig – *AI: A Modern Approach*
* Landauer & Dumais – *Latent Semantic Analysis*
* Mikolov et al. – *Word2Vec* papers
* Sahlgren et al. – *Encoding word order with permutations*
* Schnabel et al. – *Evaluation methods for word embeddings*

---

## 🧾 Summary of Key Concepts

| Concept        | Description                                                    | Example                        |
| -------------- | -------------------------------------------------------------- | ------------------------------ |
| **N-gram**     | Simple probabilistic model based on sequences                  | “to be or not to be”           |
| **LSA**        | Matrix decomposition to capture latent semantics               | “boat” and “ship” become close |
| **RI**         | Incremental random vector method                               | Faster but approximate         |
| **Word2Vec**   | Neural model learning semantic relationships                   | “king - man + woman ≈ queen”   |
| **Evaluation** | Intrinsic (similarity, analogy) / Extrinsic (task performance) | TOEFL synonym test             |

---
