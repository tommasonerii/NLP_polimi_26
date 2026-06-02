# 🍹 PoliMillionaire NLP 2026

> **Who Wants to Be a PoliMillionaire?** — a RAG agent that plays the quiz using only open-weights models running locally.

<div align="center">

### 👥 Team **NeuroniNegroni**

| Member | GitHub |
| :-- | :-- |
| **Giulia Mengoli** | [@giulimengo](https://github.com/giulimengo) |
| **Giorgio Monaco** | [@giorgiomonaco](https://github.com/giorgiomonaco) |
| **Tommaso Neri** | [@tommasonerii](https://github.com/tommasonerii) |

*Natural Language Processing course project — A.Y. 2025/26 @ Politecnico di Milano*

</div>

---

The system plays **Who Wants to Be a PoliMillionaire?** using open-weights models running locally. The final solution combines retrieval augmented generation, neural reranking, deterministic tools for mathematics, non-generative external sources for recent questions, and a speech variant with local ASR.

## Final version

The main final notebook to deliver and present is:

```text
project/notebooks/delivery/notebook_final.ipynb
```

This notebook includes the final text pipeline and its speech extension. The text base, useful as a reference and for ablation, is:

```text
project/notebooks/development/12-v8-clean.ipynb
```

`notebook_final.ipynb` starts from the pipeline of `12-v8-clean.ipynb` and only adds the speech adapter: it downloads the WAV audio from the server, transcribes the question and options with Whisper, builds a compatible text question, and calls the same `answer_strategy`.

The `00-12_V7` notebooks remain as baselines, ablations, and experimental history.

## Assignment constraints

- No external LLM API for generating answers.
- Only open-weights models running locally.
- RAG allowed and encouraged.
- External APIs allowed only if they return raw content, not generated answers.
- Agentic tools and symbolic computation allowed.
- Timeout of about 30 seconds per question.
- Avoid requests too close together to the server.
- Compare multiple solutions, models, prompts, and architectures.

The official assignment is in [docs/assignment/GroupAssignment2026.docx](docs/assignment/GroupAssignment2026.docx).

## Final stack

| Component | Final choice |
| --- | --- |
| Main notebook | `project/notebooks/delivery/notebook_final.ipynb` |
| Text base | `project/notebooks/development/12-v8-clean.ipynb` |
| LLM answer/reasoning | `Qwen_Qwen3.5-9B-Q8_0.gguf` via `llama-cpp-python` |
| GGUF repo | `bartowski/Qwen_Qwen3.5-9B-GGUF` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` |
| Dense embedding | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Sparse retrieval | BM25/BM25S |
| Dense retrieval | HNSW |
| Local corpora | SimpleWiki, KELM, math textbooks |
| External sources | Wikipedia API, Google News RSS, Tavily |
| Maths | validated tools, SymPy/Python executor, Micro-CoT fallback |
| Speech ASR | `openai/whisper-large-v3-turbo` |
| Constrained output | GBNF single digit `0-3`, `FINAL_CHOICE` parser, option matching |

## Observed results

The logs in the repository, in text mode, show at least one run at **$1,024,000** for each category. The most important improvement is on **Maths**, where `logs/run_v8.csv` reaches $1,024,000 with 98 rows, 79 correct, 80.6% accuracy, 12.27s average latency, and 0 timeouts.

| Category | Best observed earning | Reference log |
| --- | ---: | --- |
| Entertainment | $1,024,000 | `logs/run_qwen35_gguf_all_competitions.csv`, `logs/run_v5.csv` |
| Ancient History and Politics | $1,024,000 | `logs/run_qwen35_gguf_agentic_tools_v3_all_competitions.csv`, `logs/run_qwen35_gguf_validated_tools_option_retrieval_v2.csv` |
| Science and Nature | $1,024,000 | various RAG/GGUF logs |
| Philosophy and Psychology | $1,024,000 | `logs/run_qwen35_q8_qwen3reranker06b_external_bm25s_v6.csv` |
| News | $1,024,000 | `logs/run_qwen35_gguf_validated_tools_option_retrieval_v4.csv` |
| Maths | $1,024,000 | `logs/run_12_V3_math_1M.csv`, `logs/run_v8.csv` |

The speech mode is functional but more experimental: `logs/run_v9_speech.csv` shows that the bottleneck is ASR + server timeout, not the underlying text pipeline. The notebook also saves the audio in `speech_audio_v9/` for analysis of transcription errors.

## Final Quick Start

### Kaggle

The final notebooks are designed for Kaggle with two T4 GPUs and two input datasets:

```text
/kaggle/input/datasets/giorgiomonacoo/nlp2026
/kaggle/input/datasets/tommasonerii/indexes-nlp26
```

Configure the Kaggle secrets:

```text
USERNAME
PASSWORD
HF_TOKEN
TAVILY-API-KEY
```

For the final delivery, run `project/notebooks/delivery/notebook_final.ipynb` from the start. The notebook:

1. installs minimal dependencies;
2. detects the Kaggle/Colab/local environment;
3. copies the indexes to working storage;
4. downloads and validates the Qwen3.5 Q8 GGUF;
5. loads the embedding model, BM25/HNSW, and the Qwen3 reranker;
6. loads the Maths tools, external retrieval, and routing policy;
7. runs the per-category cells and appends the text results to the logs;
8. in the final speech cells, loads Whisper and saves the speech results.

### Colab or local

The project also supports Colab/local, but Kaggle is the most stable environment for the final delivery because it clearly separates read-only datasets and output in `/kaggle/working`.

If using Colab, copy to Drive:

```text
MyDrive/nlp26/
|-- project/notebooks/delivery/notebook_final.ipynb
|-- project/notebooks/development/12-v8-clean.ipynb
|-- api_client/NLP_assignment_api_client/
|-- project/src/
|-- data/indexes/
`-- data/chunks/ or the needed corpora
```

Use Colab/Kaggle secrets, not hardcoded credentials.

## Final architecture

Summary diagram:

![Final pipeline](reports/figures/Final_Pipeline_IMG.png)

```text
Question + options
    |
    v
Environment setup
    |-- API client
    |-- local indexes
    |-- Qwen3.5 GGUF
    |-- Qwen3 reranker
    |
    v
Routing policy
    |
    |-- Maths
    |     |-- validated deterministic tools
    |     |-- Python/SymPy executor fallback
    |     `-- Micro-CoT local Qwen fallback with FINAL_CHOICE
    |
    |-- News
    |     |-- Google News RSS US/UK
    |     |-- Tavily news/raw search
    |     |-- headline-aware prompt
    |     |-- model-knowledge fallback if articles lack answer
    |     `-- constrained RAG fallback
    |
    |-- Entertainment / History
    |     |-- local retrieval
    |     |-- Wikipedia API
    |     |-- Tavily raw retrieval
    |     `-- unified rerank + answer extraction
    |
    `-- Default knowledge categories
          |-- SimpleWiki BM25 + dense HNSW
          |-- KELM BM25 + dense HNSW
          |-- optional textbook indexes
          |-- RRF fusion
          |-- Qwen3 reranking
          `-- adaptive option-wise evidence
    |
    v
Constrained option id
    |
    v
API answer + CSV logging
```

### Speech adapter

The speech adapter (integrated in `notebook_final.ipynb`, developed in `development/13_speech.ipynb`) keeps the text pipeline unchanged:

```text
Speech game WAV audio
    |
    v
Whisper large-v3-turbo
    |
    v
SpeechQuestion(question_text, option_texts)
    |
    v
V8 answer_strategy
    |
    v
API answer + speech metadata logging
```

The speech log adds the transcript, audio paths, fetch times, ASR times, and ASR device.

## Main components

### Local retrieval

- SimpleWiki: short encyclopedic knowledge.
- KELM: structured assertions useful for factual questions.
- Math textbooks: statistics, algebra, calculus, discrete math, analysis, topology.
- BM25 for fast lexical recall.
- HNSW dense search for semantic similarity.
- Reciprocal Rank Fusion to merge sparse/dense rankings.
- Qwen3-Reranker to reorder the final candidates.

### External retrieval

External sources do not generate answers: they return raw text that is then evaluated locally.

- Wikipedia API: used mostly for Entertainment and Ancient History/Politics.
- Google News RSS: used for News, with URL decoding and article body fetching.
- Tavily: used as a second raw source to increase coverage on News and knowledge.

### Maths

The Maths branch avoids depending on the model right away:

1. validated deterministic patterns and tools;
2. numeric/text matching against the options;
3. sandboxed Python executor with SymPy/math/Fraction/NormalDist;
4. local Qwen Micro-CoT with constrained output.

The LLM JSON router remains in the code as a baseline/ablation, but in routing the deterministic tools are preferred for latency and robustness.

### Output and parsing

- When supported by `llama-cpp-python`, the notebook uses GBNF `root ::= [0-3]` to force a single option id.
- The Maths fallbacks use `FINAL_CHOICE`.
- The parser also tries textual and numeric matching with the options.
- The CSV logs raw output, strategy, confidence, fallback, tool trace, and retrieved context.

## Models and evolution

### Models used

| Model / technique | Use in the project | Observed weaknesses |
| --- | --- | --- |
| First-option baseline | Minimal baseline: always picks the first option and verifies the API/logging end-to-end. | Random accuracy, no understanding, useful only to test the client and the CSVs. |
| TF-IDF | First lexical retrieval over SimpleWiki/KELM. | Very sensitive to exact words; fails on paraphrases, recent questions, and semantically close options. |
| BM25 | Sparse retrieval more robust than TF-IDF, used on SimpleWiki, KELM, and textbooks. | Still lexical: rewards shallow overlap and can retrieve out-of-context documents. |
| BM25S | Fast variant to index external documents or small corpora on the fly. | Depends on the quality of the retrieved documents; if the external source is noisy, it indexes noise. |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | First neural reranker in notebooks 05-06. | Cheap but not strong enough for complex questions and long contexts. |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Dense embedding for HNSW and semantic retrieval. | Improves semantic recall, but on its own does not decide the answer and can retrieve semantically close but non-resolving passages. |
| HNSW | Approximate nearest neighbor index for fast dense search. | Requires precomputed indexes and consistent metadata; quality depends on the embeddings. |
| `Qwen/Qwen2.5-0.5B/1.5B-Instruct` | Initial experiments with a small LLM and tool-router in Colab. | Too weak for reliable reasoning and structured output under timeout. |
| `Qwen_Qwen3.5-9B-Q6_K_L.gguf` | First strong local LLM for RAG and Maths. | Good memory/speed compromise, but less accurate than Q8 and more fragile on computations or constrained output. |
| `Qwen_Qwen3.5-9B-Q8_0.gguf` | Final LLM for reasoning, option choice, and fallback. | Heavier in VRAM/disk; requires a correct llama.cpp setup and can still be wrong if the retrieval is noisy. |
| `Qwen/Qwen3-Reranker-0.6B` | Final reranker for local candidates and external sources. | Costly compared to MiniLM; on a single T4 it competes with the GGUF for GPU/latency. |
| SymPy / Python executor | Deterministic computation for Maths: equations, probability, algebra, statistics. | Needs robust parsing of the question; fails if the text is ambiguous or if the problem requires concepts not coded. |
| Wikipedia API | Raw external source for Entertainment and History. | Homonyms and related pages can distract; needs query generation and conservative reranking. |
| Google News RSS | Primary source for recent News. | Articles not always accessible, titles more informative than the body, redirects, and noisy HTML content. |
| Tavily | Alternative raw source for News and knowledge. | Useful but not guaranteed coverage; can return related results instead of the exact news. |
| `openai/whisper-large-v3-turbo` | Final ASR for speech mode. | Adds latency and can mis-transcribe math, proper names, and short options. |

### Experimental evolution

| Step | What changes | Why it was needed | Limitation that led to the next step |
| --- | --- | --- | --- |
| `00` | API smoke test and first-option logging. | Verify login, start game, answer submit, and CSV. | Does not solve the task, serves only as a baseline. |
| `01-04` | TF-IDF/BM25 over SimpleWiki and KELM, without LLM. | Understand how far lexical retrieval gets. | Retrieves evidence but does not reason well on the options; poor semantic robustness. |
| `05` | Added MiniLM/BERT reranker. | Reorder the documents retrieved by BM25. | Improves the ranking, but without an LLM there is no real contextual decision. |
| `06` | First small Qwen LLM + Maths tool-router. | Let the model choose the option and try structured tools. | Small model unstable; fragile JSON/tool calls. |
| `07` | Building dense HNSW indexes. | Increase semantic recall beyond lexical overlap. | Dense retrieval helps but must be fused and reranked. |
| `08` | Hybrid RAG with Qwen3.5-9B GGUF. | Move to a strong local LLM with BM25+dense+RRF. | Maths and output parsing remain fragile. |
| `09-11` | Maths tools, JSON router, and hardening. | Cover recurring computations with SymPy/tools instead of only the LLM. | Router too permissive or costly; many math edge cases not covered. |
| `12` / `12_V2` | Validated tools, option-wise retrieval, GBNF. | Make output and tool calls more controllable. | News and recent questions not covered by the static corpus; Maths still incomplete. |
| `12_V3` / `12_V4` | Analysis-first Maths, Micro-CoT, News/Tavily. | Give the model a minimal reasoning space and add recent sources. | Noisy external sources and fragile News mapping. |
| `12_V5*` | Unified retrieval and answer-first micro-reasoning. | Avoid truncation: answer first, reason after. | Excellent on knowledge, but Maths still requires more targeted tools. |
| `12_V6` / `12_V7` | Qwen3-Reranker, more controlled external retrieval, anti-trap News. | Improve external sources and reduce distractions. | The V7 recommendation is superseded by the final V8 work. |
| `12-v8-maths` | Experiments dedicated to the Maths branch. | Bring Maths to a competitive level with more targeted fixes and tools/fallbacks. | Experimental notebook, not as clean as the main deliverable. |
| `12-v8-clean` | Final consolidation of the text pipeline. | Bring together the best parts in a more linear notebook usable for delivery. | Does not include speech; that is separate in V13. |
| `13_speech` | Speech adapter on top of V8 with Whisper large-v3-turbo. | Support the voice mode without changing the decision engine. | ASR and audio fetch consume time; spoken math and short options remain difficult. |

## Notebook map

The notebooks are organized into two folders:

- `project/notebooks/delivery/` — the final notebook to deliver and present;
- `project/notebooks/development/` — baselines, ablations, and experimental history.

### Delivery — `project/notebooks/delivery/`

| Notebook | Role | Status |
| --- | --- | --- |
| `notebook_final.ipynb` | V8 text pipeline + speech adapter | **main final** |
| `notebook_final_html.html` | HTML export of the final notebook | presentation snapshot |

### Development — `project/notebooks/development/`

| Notebook | Role | Status |
| --- | --- | --- |
| `00_api_smoke_test.ipynb` | API smoke test and first-option baseline | baseline |
| `01_quiz_tfidf_no_llm.ipynb` | TF-IDF SimpleWiki without LLM | baseline |
| `02_quiz_bm25_no_llm.ipynb` | BM25 SimpleWiki without LLM | baseline |
| `03_quiz_bm25_multi_index_no_llm.ipynb` | BM25 SimpleWiki + KELM | baseline |
| `04_quiz_tfidf_multi_index_no_llm.ipynb` | TF-IDF SimpleWiki + KELM | baseline |
| `05_quiz_bm25_multi_index_bert_no_llm.ipynb` (+ `_colab`) | BM25 + MiniLM/BERT reranker | reranking ablation |
| `06_quiz_bm25_bert_llm_agentic_tools_colab.ipynb` | first small LLM + tool router | agentic ablation |
| `07_build_dense_embeddings_colab.ipynb` | building dense HNSW indexes | utility |
| `08_hybrid_pipeline.ipynb` | first Hybrid RAG with Qwen3.5 GGUF | LLM baseline |
| `09_hybrid_pipeline_math_tools.ipynb` | added Maths tools and textbook indexes | Maths ablation |
| `10_agentic_math_tools_prof_style.ipynb` | Maths JSON router in tool-use style | ablation |
| `11_agentic_math_router_hardened.ipynb` | Maths parser/router hardening | ablation |
| `12_validated_tools_option_retrieval.ipynb` | validated tools + option retrieval V1 | strong baseline |
| `12_V2_validated_tools_option_retrieval.ipynb` | GBNF + adaptive retrieval + Maths fixes | ablation |
| `12_V3_validated_tools_option_retrieval.ipynb` | analysis router + Micro-CoT | ablation |
| `12_V3_math_1M.ipynb` | dedicated $1,024,000 Maths experiment | evidence |
| `12_V4_validated_tools_option_retrieval.ipynb` | News/Tavily + extended Maths | ablation |
| `12_V5-kaggle.ipynb` | unified retrieval + answer-first reasoning | Kaggle ablation |
| `12_V5_complete.ipynb` | V5 + Qwen3 reranker + News fallback + Python executor | pre-V8 baseline |
| `12_V6_validated_tools_option_retrieval.ipynb` | temporary external BM25S + Qwen3 reranker | external ablation |
| `12_V7_validated_tools_option_retrieval.ipynb` | semantic gate + anti-trap News prompt | previous baseline |
| `12-v8-maths.ipynb` | experimental Maths/V8 branch | evidence |
| `12-v8-clean.ipynb` | clean V8 text pipeline | final text base |
| `13_final.ipynb` | V8 pipeline consolidation + speech | pre-delivery |
| `13_final_comments.ipynb` | commented end-to-end version of the final pipeline | documentation |
| `13_speech.ipynb` | V8 pipeline + speech adapter | speech development |
| `ASR_speech_benchmark.ipynb` | separate live ASR benchmark | speech analysis |

> External reference: `api_client/NLP_assignment_api_client/PoliMillionaire.ipynb` is the official API tutorial.

## Repository structure

```text
NLP_polimi_26/
|-- api_client/
|   `-- NLP_assignment_api_client/
|       `-- millionaire_client/       # Python client for the PoliMillionaire API
|-- data/
|   |-- chunks/                       # chunked corpora (jsonl) for the indexes
|   |-- indexes/                      # BM25 and dense HNSW indexes, often via Git LFS
|   |-- kelm/                         # KELM subset
|   |-- maths/                        # textbook PDFs for the math indexes
|   `-- wiki/                         # SimpleWiki dump
|-- docs/
|   |-- assignment/                   # official assignment
|   |-- slides/ , tutorials/          # course material and tutorials
|   |-- retrieval_indexes.md          # commands for corpora and indexes
|   `-- kelm_limited.md               # KELM subset notes
|-- logs/                             # experiment CSVs and analysis
|-- project/
|   |-- notebooks/
|   |   |-- delivery/                 # final notebook to deliver
|   |   `-- development/              # baselines, ablations, experimental history
|   `-- src/                          # corpus, indexes, retrieval, tool scripts
|-- reports/
|   `-- figures/                      # generated figures (incl. Final_Pipeline_IMG.png)
|-- API_README.md
`-- README.md
```

## Repository setup

The indexes and some PDFs are large files and may use Git LFS:

```bash
git lfs install
git lfs pull
```

Minimal local environment for scripts and analysis:

```bash
conda create -n polimillionaire python=3.11
conda activate polimillionaire
pip install numpy pandas scikit-learn joblib bm25s pypdf requests matplotlib seaborn sympy
```

The final notebooks autonomously install the heavier runtime dependencies, including:

```text
huggingface_hub
hnswlib
bm25s
llama-cpp-python
sentence-transformers
transformers
accelerate
googlenewsdecoder
tavily-python
soundfile
scipy
```

## PoliMillionaire API

Assignment endpoint:

```text
http://131.175.15.22:51111/
```

Minimal usage:

```python
import sys
sys.path.append("api_client/NLP_assignment_api_client")

from millionaire_client import MillionaireClient

client = MillionaireClient("http://131.175.15.22:51111/")
client.login(username, password)
competitions = client.competitions.list_all()
```

Speech mode:

```python
game = client.game.start(competition_id=comp_id, mode="speech")
question_audio = game.fetch_audio_question()
option_a_audio = game.fetch_audio_option_next()
```

Full details in [API_README.md](API_README.md).

## Building corpora and indexes

The complete documentation is in [docs/retrieval_indexes.md](docs/retrieval_indexes.md).

SimpleWiki chunks example:

```bash
conda run -n polimillionaire python project/src/make_retrieval_corpus.py \
  data/wiki/simplewiki_articles.jsonl \
  --source simplewiki \
  --id-prefix swiki \
  --max-words 160 \
  --overlap-words 30 \
  --min-words 20 \
  --output data/chunks/simplewiki_160w.jsonl
```

BM25 index:

```bash
conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_160w.jsonl \
  --kind bm25 \
  --title-repeat 2 \
  --bm25-remove-stopwords \
  --output data/indexes/simplewiki_160w_title2_stop_bm25.joblib
```

Math indexes:

```powershell
.\project\src\build_all_textbook_bm25_indexes.ps1
.\project\src\build_all_textbook_dense_indexes.ps1
```

Manual query:

```bash
conda run -n polimillionaire python project/src/query_retrieval_index.py \
  data/indexes/simplewiki_160w_title2_stop_bm25.joblib \
  --query "What term describes Buster Keaton's signature facial expression? Grin Laugh Deadpan Smirk" \
  --top-k 3
```

## Logging

Relevant diagnostic fields:

- `competition_name`, `question_id`, `question_level`
- `chosen_option_id`, `correct`, `earned_amount`, `timed_out`
- `latency_seconds`
- `strategy`, `decision_source`, `confidence`
- `raw_llm_output`, `prompt_version`
- `retrieved_context`, `retrieval_sources`, retrieval scores and margins
- `option_evidence_scores_json`, `option_evidence_json`
- `tool_validated`, `validated_tool_call`, `math_tool_trace`
- `fallback_used`
- `textbook_context_*`
- speech-only: transcript, audio paths, fetch seconds, ASR seconds, ASR model/device

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `ModuleNotFoundError: millionaire_client` | Make sure `api_client/NLP_assignment_api_client` is on `sys.path`. |
| API unreachable | Avoid PoliMi Wi-Fi if it blocks the port; use mobile network/VPN. |
| Kaggle without GPU | Enable the GPU accelerator and verify with `nvidia-smi`. |
| OOM on GGUF or reranker | Reduce `RERANKER_BATCH_SIZE`, `RERANKER_MAX_LENGTH`, `LLM_CONTEXT_K`, or llama.cpp GPU layers. |
| Invalid GGUF | Verify size/`GGUF` header; clear the model cache and re-download with revision pinning. |
| `TAVILY-API-KEY` missing | Create the Kaggle secret with that exact name or adapt the cell. |
| Unparsable LLM output | Use GBNF when available; otherwise check `raw_llm_output` and fallback. |
| Noisy News | Check articles, headlines, and `retrieval_sources`; Tavily/RSS can retrieve related but non-resolving articles. |
| Maths slow or wrong | Look at `math_tool_trace`, `validated_tool_call`, `fallback_used`, and textbook context. |
| Speech timeout | ASR + audio fetch consume the budget; load and warm up Whisper before starting the speech game. |
| Speech mis-transcribes options | Use the audio saved in `speech_audio_v9/` and compare `api_options_json` with `speech_options_transcript_json`. |

## References

1. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. ICLR. [arXiv](https://arxiv.org/abs/2005.11401)
2. Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS. [arXiv](https://arxiv.org/abs/2201.11903)
3. Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR. [arXiv](https://arxiv.org/abs/2210.03629)
4. Chen et al. (2022). *Program of Thoughts Prompting*. [arXiv](https://arxiv.org/abs/2211.12588)
5. Schick et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. ICLR. [arXiv](https://arxiv.org/abs/2302.04761)
6. Asai et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. [arXiv](https://arxiv.org/abs/2310.11511)
7. Vu et al. (2024). *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation*. ACL Findings. [paper](https://aclanthology.org/2024.findings-acl.813/)
8. Lù (2024). *BM25S: Orders of magnitude faster lexical search via eager sparse scoring*. [arXiv](https://arxiv.org/abs/2407.03618)
9. Qwen Team (2025). *Qwen3 Embedding and Reranker model card*. [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

---

<div align="center">

Made with 🍹 by **NeuroniNegroni** — Tommaso · Giulia · Giorgio

</div>
