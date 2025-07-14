# ⚖️ Lawgorithm

**Semantic Similarity Detection of Legal Cases using Multi-Metric Fusion, Genetic Optimization & Graph Neural Networks**

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![HuggingFace Datasets](https://img.shields.io/badge/HuggingFace-Datasets-orange.svg)
![GNN](https://img.shields.io/badge/GNN-PyTorch%20Geometric-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📜 Overview

**Lawgorithm** is an intelligent legal case similarity engine that leverages:
- SentenceTransformer-based embeddings
- Five-layered similarity metrics (Cosine, Euclidean, Manhattan, Jaccard, BM25)
- Weight optimization via Genetic Algorithm (GA)
- Graph construction based on fused similarity
- Graph Neural Network (GCN variant)
- Explainability using GNNExplainer

It allows legal professionals or systems to retrieve semantically similar cases, explain results, and visualize importance of various features in legal judgments.

---

## 🧠 Key Features

- ✅ Embedding generation using `all-MiniLM-L6-v2`
- ✅ Multi-metric similarity computation:
  - Cosine
  - Euclidean
  - Manhattan
  - Jaccard
  - BM25
- ✅ Genetic Algorithm for optimized metric weighting
- ✅ Graph-based modeling of case relationships
- ✅ GCN-based model for learning and pattern generalization
- ✅ Semantic retrieval for unseen cases
- ✅ Explanation of predictions using GNNExplainer
- ✅ Performance visualizations: Fitness, Loss, Similarity, Feature Scores

---

## 📊 Sample Visualizations

- Fitness score evolution (GA)
- GCN model training loss
- Bar chart of top similar case scores
- Feature importance from GNNExplainer

---

## 🗃️ Dataset

- **Source**: [`ninadn/indian-legal`](https://huggingface.co/datasets/ninadn/indian-legal) (via HuggingFace Datasets)
- Sampled: 1000 case documents for prototype

---

## 🔧 Installation & Setup

### 1. Clone this repository

```bash
git clone https://github.com/your-username/Lawgorithm.git
cd Lawgorithm
```

### 2. Create Virtual Environment (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

**`requirements.txt` includes:**
```txt
transformers
sentence-transformers
datasets
torch
scikit-learn
rank-bm25
networkx
deap
matplotlib
seaborn
torch-geometric
```

---

## 🚀 How to Run

```bash
python lawgorithm_main.py
```

This script performs:
- Data loading and preprocessing
- Embedding generation
- Similarity matrix calculations
- Genetic Algorithm optimization
- Graph construction and GCN training
- Semantic similarity queries and explanations

---

## 🧪 Example Use

```python
case_text = "The appellant was convicted under Section 302..."
get_most_similar_cases(case_text, top_n=3)
explain_case(index_of_result)
```

---

## 🧠 Model Architecture

```text
             +-------------------+
             |  Case Embeddings  |  ← SentenceTransformer
             +-------------------+
                       ↓
    +-----------------------------------------------+
    |    Similarity Matrices (Cosine, Euclidean, etc)|
    +-----------------------------------------------+
                       ↓
       +-----------------------------------+
       |  Genetic Algorithm Weighted Fusion |
       +-----------------------------------+
                       ↓
         +---------------------------+
         |   Similarity-based Graph   |
         +---------------------------+
                       ↓
         +------------------------+
         |  GCN Model (GMN style) |
         +------------------------+
                       ↓
         +----------------------------+
         | GNNExplainer + Retrieval  |
         +----------------------------+
```

---

## 📌 Limitations

- Uses static features (degree centrality + noise); can be extended with legal-domain features.
- Limited to 1000-case sample for prototyping; performance on larger datasets may vary.
- No fine-tuned legal language model yet.

---

## 🛠️ Future Work

- 🧾 Add legal domain-specific embeddings (e.g., `legal-bert`)
- 🧠 Add summary-aware node features
- 🗂 Expand to entire dataset with batching
- 🔍 Improve visualization and UI for lawyer-facing deployment

---

## 👨‍⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💬 Acknowledgements

- [HuggingFace](https://huggingface.co/)
- [SentenceTransformers](https://www.sbert.net/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [DEAP](https://deap.readthedocs.io/)
