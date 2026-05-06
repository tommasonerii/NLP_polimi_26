# 🏅 QA & Retrieval Application (NLP Polimi '26)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![Git LFS](https://img.shields.io/badge/Git-LFS-green.svg)](https://git-lfs.com/)

This repository contains the final work for the **Natural Language Processing (2025/26)** course assignment at Politecnico di Milano.

The main goal of the project is to develop an **autonomous chatbot/agent** capable of competing in the online quiz game "**Who wants to be a PoliMillionaire?**" by interacting via text APIs. To answer multiple-choice questions as accurately as possible, the system adopts advanced *Information Retrieval* (IR) techniques interfaced with a *Retrieval-Augmented Generation* (RAG) module, leveraging knowledge bases derived from the **SimpleWiki** and **KELM** datasets.

---

## 🎯 Project Overview and Requirements

The project focuses on intelligent information retrieval and quiz resolution in strict compliance with the exam rules. What we do in detail:

1. **Strictly Open-Weights/Local Models**: As required (no calls to paid LLM APIs like OpenAI), the architecture downloads and runs open-source LLM inference directly in-memory (e.g., on local *Google Colab*).
2. **Custom RAG and Indexes Construction**: Offline development and benchmarking of *Sparse* (TF-IDF, BM25) and *Dense* (bi-encoder models + ANN vector embeddings) text indexes, which can be loaded in real-time by the retriever to compose response prompts.
3. **Agentic Math Component (SymPy-backed)**: Since LLMs are notoriously prone to geometric and algebraic hallucinations, we implemented a dedicated mathematical safety agent (`project/src/agentic_tools.py`) powered by `SymPy`. It intercepts rigid mathematical requests, parses formulas correctly, solves them analytically, and bypasses the LLM returning the 100% exact option when a match exists.
4. **PoliMillionaire Text API Client**: The `src/` module (and `api_client/` folder) contains the automation that queries the game host (handling the required rate limiting), parses the options, and submits them to the solver logics (Agent / LLM).
5. **End-to-End Experiments**: Comparisons on precision, architectures, and evaluation runs (RAG pipeline) measured against the game leaderboard positions. All logs are available in the `logs/` folder.

---

## 📁 Repository Structure

```text
📦 NLP_polimi_26
├── 📂 data/             # Raw data, pre-processed corpora, and saved indexes
│   ├── 📂 chunks/       # Fragmented documents for indexing
│   ├── 📂 indexes/      # Serialized indexes (Sparse and Dense) - tracked with Git LFS!
│   └── 📂 wiki/ & kelm/ # Raw JSONL dumps
├── 📂 docs/             # Additional documentation and theory extraction files
├── 📂 logs/             # CSV of quiz results and telemetry for various experiments
├── 📂 project/          # Project core
│   ├── 📂 notebooks/    # Exploratory notebooks for progressive testing (00 -> 07)
│   └── 📂 src/          # Source scripts: building, querying, and agents
└── 📝 README.md
```

---

## 🔍 Implemented Indexes Details

To thoroughly evaluate the retrieval effectiveness, we built several index versions for both the **SimpleWiki** corpus (about 1.6 million chunks) and the **KELM** subset (500k entries).

### 1. Sparse Indexes (Lexical Matching)
Sparse indexes rely on exact token matching and frequency statistics. They are highly memory-efficient and fast.

- **TF-IDF (`_tfidf.joblib`)**: Classic term-document matrix. Provides a quick global matching baseline and performs well when the query and document share a strong common vocabulary.
- **BM25 Base (`_bm25.joblib`)**: Standard BM25 implementation, considered the excellent baseline in text Information Retrieval for handling term frequency saturation.
- **BM25 Stop (`_stop_bm25.joblib`)**: Variant of the standard BM25 with the stop-word vocabulary removed. It eliminates the "noise" of common words (e.g., *the, a, is*), allowing BM25 to calculate more stable scores on truly important terms.
- **BM25 Title-Boosted + Stop (`_title2_stop_bm25.joblib`)**: Highly optimized version that (1) filters out stop-words and (2) assigns double weight to the text of the document **titles**. This is very useful in encyclopedic QA tasks where the answer (or the requested entity) often matches the article's title.

### 2. Dense Indexes (Semantic Matching)
Dense indexes use neural embedding vectors to map queries and documents into the same semantic space. They enable "soft-matching" and overcome the synonymy limitations of lexical approaches.

- **Dense embeddings + HNSW (`_dense_hnsw.index` & `_meta.joblib`)**: Neural vector mapping indexed with **HNSW** (Hierarchical Navigable Small World) data structures. Dense vectors allow for fast sub-linear semantic search even over hundreds of thousands of documents, albeit with higher RAM costs. The metadata file (`_meta.joblib`) maps the physical HNSW vector coordinates back to the original text document ID.

---

## 🚀 Local Installation and Setup

### Sensitive Requirements
- Modules in `requirements.txt`.
- **Git LFS** installed (crucial for downloading the heavy indexes located in `data/indexes/`). Otherwise, Git will only download a fragile pointer text file instead of the actual index, causing local runs to fail or crash.

### Setup
```bash
# 1. Clone the repository and fetch the heavy files
git clone https://github.com/TUO-USER/NLP_polimi_26.git
cd NLP_polimi_26
git lfs pull # Explicitly force the download of joblib and binary test files

# 2. Create the environment and install dependencies
pip install -r requirements.txt
```

---

## ☁️ Heavy Algorithms Execution - Google Colab Setup

Heavy computations, such as extracting neural embeddings (dense), were performed via Google Colab using the notebook `project/notebooks/07_build_dense_embeddings_colab.ipynb`.

The only specific requirement for Colab environments is that they **use a mounted Google Drive** to manage and persist the gigabytes of indexes without losing them during unexpected session interruptions. Therefore, note that the paths inside the Colab notebook differ from the local folder context; for example, they use prefixes like this:
- `PROJECT_ROOT = Path('/content/drive/MyDrive/nlp26')`
- `INDEX_DIR = PROJECT_ROOT / 'indexes'`
- `KELM_CORPUS_PATH = PROJECT_ROOT / 'kelm' / 'kelm_subset_500k.jsonl'`
