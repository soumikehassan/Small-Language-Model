## 🧠 Backbone of Transformer Architecture

The **Transformer** is a groundbreaking neural network architecture based on self-attention mechanisms. Its **backbone** refers to the core layers and operations that make Transformers powerful and flexible for NLP and beyond.

---

### 🔩 Core Components

#### 1. **Input Embeddings**
- Converts tokens (words/subwords) into dense vectors.
- Includes **Positional Encoding** to retain the order of tokens, since Transformers are non-sequential by design.

#### 2. **Transformer Block (Backbone Unit)**
Each Transformer block is composed of:

- ✅ **Multi-Head Self-Attention**
  - Allows the model to attend to different parts of the input simultaneously.
  - Learns contextual relationships.

- ✅ **Layer Normalization**
  - Stabilizes training and accelerates convergence.

- ✅ **Feedforward Neural Network (FFN)**
  - A fully connected MLP applied to each position.
  - Commonly uses ReLU or GELU activation.

- ✅ **Residual Connections**
  - Skip connections to preserve gradients and improve learning in deeper models.

#### 3. **Stacked Layers**
- Transformers consist of **stacked encoder and/or decoder blocks**.
- Number of layers (depth) and attention heads are configurable.

---

### 🔄 Variants Based on Backbone

- **BERT** – Uses only the **encoder** part.
- **GPT** – Uses only the **decoder** part.
- **T5 / BART** – Use both encoder and decoder.

---

### 🧠 Summary

> The backbone of a Transformer is a **stack of attention + feedforward blocks**, enriched with **positional information** and **residual connections**. This architecture is what powers large language models like GPT, BERT, T5, etc.

---

📌 _Want to dive deeper? Read the original paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)_

