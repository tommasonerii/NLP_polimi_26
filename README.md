# PoliMillionaire NLP 2026

**Progetto NLP 2025-26 @ Politecnico di Milano**

Chatbot che gioca a **Who Wants to Be a PoliMillionaire?** usando solo modelli open-weights eseguiti localmente. Il sistema combina retrieval augmented generation (RAG), ranking lessicale/neurale, tool-augmented reasoning per matematica e fallback robusti—tutto entro il vincolo di 30 secondi per domanda.

**Stack finale (Notebook 10):**
- 📚 Retrieval: SimpleWiki + KELM (BM25 + Dense HNSW)
- 🔀 Fusione: Reciprocal Rank Fusion su 4 indici
- 🧠 Reranking: Cross-encoder BERT (CPU)
- 🧮 Math: LLM tool router + SymPy + fallback RAG
- 🤖 LLM: Qwen3.5-9B (Q6_K_L GGUF via llama-cpp-python)
- 📊 Logging: CSV con latenza, strategia, evidenza, correttezza

## Vincoli dell'assignment

La consegna richiede un notebook Colab autoesplicativo e una breve presentazione video del lavoro. I vincoli tecnici principali sono:

- i modelli devono essere eseguiti localmente, non tramite API LLM;
- sono ammessi solo modelli open-weights;
- una pipeline RAG e incoraggiata, usando indici locali oppure API che restituiscano contenuto grezzo non generato;
- eventuali API esterne non devono essere API a pagamento ne generare direttamente risposte con LLM;
- l'uso di tool agentici, per esempio calcolatrice o SymPy, e esplicitamente incoraggiato;
- il sistema deve rispettare il timeout di circa 30 secondi per domanda;
- bisogna evitare richieste consecutive troppo rapide al server del gioco;
- la valutazione deve confrontare piu soluzioni, prompt, modelli, architetture e categorie di errore.

La consegna e in [docs/assignment/GroupAssignment2026.docx](docs/assignment/GroupAssignment2026.docx). **Scadenza: 2 giugno 2026 alle 23:00 via WeBeep.**

## Quick Start (Colab)

**Per usare il notebook finale (10) in Colab:**

1. Coppia il notebook 10 su Google Drive:
   ```text
   MyDrive/nlp26/
   ├── notebooks/10_agentic_math_tools_prof_style.ipynb
   ├── api_client/NLP_assignment_api_client/
   ├── src/*.py
   └── indexes/simplewiki*.joblib, kelm*.joblib, *dense*.index, *dense*meta.joblib
   ```

2. Crea Colab Secrets (`Runtime > Manage sessions > Secrets`):
   - `HF_TOKEN`: token Hugging Face per scaricare Qwen3.5-9B
   - `USERNAME`: account PoliMillionaire
   - `PASSWORD`: password

3. Apri il notebook 10 in Colab e esegui le celle:
   - Celle 1-7: setup dipendenze, GPU, paths
   - Celle 8-13: carica indici, modelli, embedding
   - Celle 14+: esegui game loop, salva CSV in Drive

**Tempo stimato:** 5-10 min setup, poi ~10 sec/domanda durante il gioco.

## Architettura

La pipeline RAG segue il flusso:

```
Question + Opzioni
    ↓
[Maths branch? Solo per "Maths" competition]
  ├─ Deterministic tools (regex pattern matching)
  ├─ LLM tool-router (JSON struct) → SymPy/calcolo
  └─ Fallback: → RAG se nessun tool match
    ↓
[Knowledge branch] Retrieval ibrido
  ├─ SimpleWiki BM25 (sparse) + Dense HNSW (dense)
  ├─ KELM BM25 (sparse) + Dense HNSW (dense)
  ├─ Reciprocal Rank Fusion (RRF) su 4 ranking
  └─ Cross-encoder reranking (BERT, CPU)
    ↓
[Scoring & Selection]
  ├─ Top K evidenza fornita al LLM
  ├─ Qwen3.5-9B predice option_id
  └─ Fallback: prima opzione se output non parseable
    ↓
Option ID → API → Logging CSV
```

**Componenti:**
- **Indici:** SimpleWiki (434k docs, 160w), KELM (500k asserzioni corte), libri di matematica (PDF → chunks)
- **Retrieval:** BM25 (sparse, veloce) + embedding densi (HNSW, accurato)
- **Reranking:** Cross-encoder MiniLM per riordinamento top-K
- **Math tools:** Deterministic calcolator, equation solver, modular arithmetic, prime factorization, percentage
- **LLM fallback:** Qwen3.5-9B (9B params, 7.6 GiB Q6_K_L quantized) per domande generali non risolte da tool

## Struttura

```text
NLP_polimi_26/
|-- api_client/
|   `-- NLP_assignment_api_client/
|       `-- millionaire_client/       # client Python per la API del gioco
|-- data/
|   |-- indexes/                      # indici joblib/HNSW tracciati con Git LFS
|   `-- maths/                        # PDF usati per indici matematici
|-- docs/
|   |-- assignment/                   # testo ufficiale della consegna
|   |-- retrieval_indexes.md          # comandi dettagliati per indici retrieval
|   `-- kelm_limited.md               # note per costruire subset KELM
|-- logs/                             # risultati CSV degli esperimenti
|-- project/
|   |-- notebooks/                    # notebook progressivi 00-09
|   `-- src/                          # script per corpus, indici, retrieval e tool
|-- reports/
|   `-- figures/                      # grafici di accuratezza e latenza
|-- API_README.md                     # note dettagliate sul client PoliMillionaire
`-- README.md
```

## Setup

Gli indici e diversi PDF sono file grandi e sono gestiti con Git LFS. Dopo il clone:

```bash
git lfs install
git lfs pull
```

Ambiente Python consigliato:

```bash
conda create -n polimillionaire python=3.11
conda activate polimillionaire
pip install numpy pandas scikit-learn joblib bm25s pypdf requests matplotlib seaborn sympy
```

Per esperimenti con modelli locali o embeddings nei notebook Colab possono servire anche pacchetti come `transformers`, `sentence-transformers`, `accelerate`, `bitsandbytes` o `faiss`, a seconda della pipeline eseguita.

## API PoliMillionaire

Il server indicato dalla consegna e:

```text
http://131.175.15.22:51111/
```

La consegna segnala che il sito potrebbe non essere raggiungibile dalla rete Wi-Fi PoliMi per un blocco sulla porta.

Il client Python si trova in:

```text
api_client/NLP_assignment_api_client/millionaire_client/
```

Uso minimo:

```python
import sys

sys.path.append("api_client/NLP_assignment_api_client")

from millionaire_client import MillionaireClient

client = MillionaireClient("http://131.175.15.22:51111/")
client.login(username, password)
competitions = client.competitions.list_all()
```

Per dettagli su login, partite, risposta, leaderboard e logging vedere [API_README.md](API_README.md).

## Notebook — Percorso dal prototipo alla produzione

I notebook in `project/notebooks/` documentano lo sviluppo progressivo. Scegli in base al tuo scenario:

| # | Nome | Scopo | Usa se... | GPU? |
| --- | --- | --- | --- | --- |
| **00** | API smoke test | Test API PoliMillionaire | Primo test API | No |
| **01–02** | TF-IDF / BM25 baseline | Baseline retrieval senza LLM | Vuoi capire baseline | No |
| **03–04** | Multi-index BM25/TF-IDF | Fusione multi-corpus | Studi IR classico | No |
| **05** | BM25 + BERT reranking | Retrieval + neural reranking | Vuoi reranking | No |
| **06** | BM25 + BERT + LLM + tools | RAG completa, LLM 1.5B, tool regex | Test su Colab T4 | ✓ |
| **07** | Build dense embeddings | Costruisce indici HNSW | Solo per costruire indici | ✓ |
| **08** | Hybrid pipeline (GGUF) | Qwen3.5 GGUF, retrieval ibrido | Test su Colab T4 | ✓ |
| **09** | Hybrid + math tools | Come 08, math con regex | Prototipo produzione | ✓ |
| **🎯 10** | **Agentic math tools** | **Come 09, math con tool-router JSON** | **Consegna finale** | ✓ |

**Raccomandazione:** Usa il **notebook 10** per la consegna. È una refactoring di 09 con architettura tool-augmented professionale (JSON router, tool registry, fallback conservativo).

## Costruire corpus e indici

La documentazione completa dei comandi e in [docs/retrieval_indexes.md](docs/retrieval_indexes.md). Esempio SimpleWiki:

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

Indice BM25 con title boost e stopword removal:

```bash
conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_160w.jsonl \
  --kind bm25 \
  --title-repeat 2 \
  --bm25-remove-stopwords \
  --output data/indexes/simplewiki_160w_title2_stop_bm25.joblib
```

Query manuale su un indice:

```bash
conda run -n polimillionaire python project/src/query_retrieval_index.py \
  data/indexes/simplewiki_160w_title2_stop_bm25.joblib \
  --query "What term describes Buster Keaton's signature facial expression? Grin Laugh Deadpan Smirk" \
  --top-k 3
```

Per costruire tutti gli indici BM25 dei libri matematici:

```powershell
.\project\src\build_all_textbook_bm25_indexes.ps1
```

Per costruire gli indici dense HNSW degli stessi libri, dopo aver creato i chunk con lo script precedente:

```powershell
.\project\src\build_all_textbook_dense_indexes.ps1
```

Se `python` non punta all'ambiente corretto:

```powershell
.\project\src\build_all_textbook_dense_indexes.ps1 `
  -PythonExe C:\ProgramData\miniconda3\Scripts\conda.exe `
  -PythonPrefixArgs @('run', '-n', 'polimillionaire', 'python')
```

I file prodotti sono `*_200w_dense_hnsw.index` e `*_200w_dense_meta.joblib` in `data/indexes/`. Per usare il notebook 09 in Colab, copiarli anche in `/content/drive/MyDrive/nlp26/indexes`.

## State-of-the-Art: Ricerca scientifica che supporta il design

Il notebook 10 segue principi consolidati da lavori recenti in NLP e tool-augmented reasoning:

| Principio | Riferimento | Applicazione |
| --- | --- | --- |
| **RAG per domande fattive** | Lewis et al. 2020 *Retrieval-Augmented Generation* | Retrieve evidenza → LLM answer |
| **Chain-of-Thought prompting** | Wei et al. 2022 *Prompting CoT* | Decomporre matematica in step |
| **ReAct: Reasoning + Acting** | Yao et al. 2022 | Router LLM decide tool, Python esegue |
| **Structured tool calls** | Schick et al. 2023 *Toolformer* | LLM output JSON strutturato, non free-form |
| **Program of Thoughts** | Chen et al. 2022 *PoT Prompting* | Math → esecuzione deterministica |
| **Conservative fallback** | Design patterns from tool-use lit. | No tool match → fallback a RAG |

## Strategie implementate (Notebooks 0-9)

- **Baseline:** API e CSV logging end-to-end
- **Sparse IR:** TF-IDF, BM25 con varianti (stopword, title-boost)
- **Multi-index:** Reciprocal Rank Fusion su piu corpora
- **KELM retrieval:** Knowledge graph strutturato per fattualita
- **Dense retrieval:** HNSW embeddings (MiniLM-L6)
- **Neural reranking:** Cross-encoder BERT (MiniLM-L-6-v2)
- **Agentic tools (v1, Nb 09):** Regex pattern matching → SymPy
- **Agentic tools (v2, Nb 10):** LLM JSON router → tool registry → fallback RAG

## Log e analisi

Gli esperimenti salvano CSV in `logs/`. I campi principali includono:

- competizione, sessione, tentativo, livello e domanda;
- opzioni e risposta scelta;
- sorgente della decisione (`retrieval`, tool, LLM, ecc.);
- evidenza recuperata;
- correttezza, premio, timeout e latenza;
- output grezzo di eventuali tool o modelli.

Gli script `project/src/analyze_bm25_results.py` e `project/src/analyze_tfidf_results.py` producono analisi e grafici comparativi. Alcune figure gia generate sono in `reports/figures/`.

## Valutazione: Metriche e Analisi richieste

Per rispondere alla consegna, il notebook finale (10) deve mostrare:

### Metriche di base
- **Accuratezza** per ogni competizione (Entertainment, History, Science, Maths)
- **Livello medio raggiunto** per category
- **Latenza media** e **max latenza** per domanda
- **Numero timeout** (risposte oltre 30s)
- **Earned amount** medio per sessione

### Analisi comparativa
- Baseline (first option) vs retrieval-only vs RAG vs tool-augmented
- Impatto di ogni componente (sparse → dense → reranking → LLM)
- Differenza tra Simple Wiki e KELM
- Efficacia dei tool matematici su Maths category

### Analisi per categoria
- Risultati breakdown per competizione
- Corte di successo/fallimento per tipo di domanda
- Esempi concreti di errori e vincoli (timeout, parse failures)
- Impatto di RAG: domande risolte vs fallite per amount di context

### Valutazione qualitativa
- Limitazioni della soluzione (timeout, OOM, token limit)
- Possibili miglioramenti (fine-tuning, ensemble, cache)
- Trade-off latenza vs accuratezza
- Robustezza e fallback chains

**Consiglio:** Salva tutti i log in CSV, poi genera grafici con matplotlib/seaborn (già in `reports/figures/`).

## Checklist per la consegna (scadenza 2 giugno 2026, 23:00)

### Notebook Colab (main deliverable)
- [ ] Usa notebook **10** come base (or 09, ma 10 è migliore)
- [ ] Self-contained: nessuna dependency esterna fuori pip
- [ ] Colab Secrets per USERNAME/PASSWORD (NO hardcoded credentials)
- [ ] Google Drive path ben documentato
- [ ] Runnable da inizio a fine senza interruzioni
- [ ] Almeno **N=5 tentativi per competizione** per avere statistica valida
- [ ] CSV logs salvati su Drive al termine

### Analisi e risultati
- [ ] Summary cell con metriche finali (accuracy, latenza, timeout count)
- [ ] Tabella comparison: baseline vs retrieval vs RAG vs tool
- [ ] Grafici: accuracy/competizione, latenza/livello
- [ ] Almeno 3 esempi di domande risolte correttamente (con evidenza)
- [ ] Almeno 3 esempi di fallimenti (timeout, parse error, wrong answer)

### Video presentazione
- [ ] Durata 5-10 min
- [ ] Spiega il problema e vincoli
- [ ] Mostra il notebook in azione
- [ ] Commenta i risultati e metriche
- [ ] Discute limiti e possibili miglioramenti
- [ ] Upload su YouTube/Drive (link in WeBeep)

### Caricamento su WeBeep
- [ ] Notebook 10 (`.ipynb` o link Drive)
- [ ] Breve README di setup (Colab secret names, paths, cose da modificare)
- [ ] Link video presentazione
- [ ] CSV finale dei risultati (facoltativo, per referenza)

## Troubleshooting

| Problema | Soluzione |
| --- | --- |
| `ModuleNotFoundError: millionaire_client` | Assicura che `api_client/NLP_assignment_api_client` sia in `sys.path` prima di `import` |
| API unreachable (PoliMi WiFi) | Usa VPN o rete mobile; segnala al docente se persiste |
| CUDA OOM su Colab | Riduci `n_gpu_layers` nel caricamento Qwen (prova 35 invece di -1) oppure passa a Q5_K_L |
| Timeout su 30s | Aumenta retrieval latency? Riduci TOP_K_RERANK o LLM_CONTEXT_K |
| CSV parse error in logging | Assicura que `json.dumps(..., ensure_ascii=False)` per testo UTF-8 |
| Dense index missing su Drive | Esegui notebook 07 (build_dense_embeddings) o scarica da backup |
| LLM output non parseable | Handler fallback in `option_id_from_text` → prima opzione |

## Risorse utili

- **Documentazione assignment:** [GroupAssignment2026.docx](docs/assignment/GroupAssignment2026.docx)
- **API details:** [API_README.md](API_README.md)
- **Comandi corpus/indici:** [docs/retrieval_indexes.md](docs/retrieval_indexes.md)
- **KELM subset:** [docs/kelm_limited.md](docs/kelm_limited.md)

## References

1. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* ICLR. [[arxiv]](https://arxiv.org/abs/2005.11401)
2. Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in LLMs.* NeurIPS. [[arxiv]](https://arxiv.org/abs/2201.11903)
3. Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in LMs.* ICLR. [[arxiv]](https://arxiv.org/abs/2210.03629)
4. Chen et al. (2022). *Program of Thoughts Prompting.* arxiv. [[arxiv]](https://arxiv.org/abs/2211.12588)
5. Schick et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.* ICLR. [[arxiv]](https://arxiv.org/abs/2302.04761)

---

**Ultimo commit:** `9f27182` — Notebook 10 e logs  
**Team:** NeuroniNegroni (Tommaso, Giulia, Gio)
