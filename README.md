# PoliMillionaire NLP 2026

**Progetto NLP 2025-26 @ Politecnico di Milano**

Chatbot che gioca a **Who Wants to Be a PoliMillionaire?** usando solo modelli open-weights eseguiti localmente. Il sistema combina retrieval augmented generation (RAG), ranking lessicale/neurale, tool-augmented reasoning per matematica e fallback robusti—tutto entro il vincolo di 30 secondi per domanda.

**Stack consigliato (Notebook 12):**
- 📚 Retrieval: SimpleWiki + KELM (BM25 + Dense HNSW) + option-wise evidence retrieval
- 🔀 Fusione: Reciprocal Rank Fusion su 4 indici
- 🧠 Reranking: Cross-encoder BERT (CPU)
- 🧮 Math: deterministic router + validated generic tools (schema, guard, deterministic matching) + SymPy
- 🤖 LLM: Qwen3.5-9B (Q6_K_L GGUF via llama-cpp-python)
- 📊 Logging: CSV con latenza, strategia, evidenza, confidence, tool traces, rejection reasons e correttezza

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

**Per usare il notebook 12 in Colab:**

1. Copia il notebook 12 su Google Drive:
   ```text
   MyDrive/nlp26/
   ├── notebooks/12_validated_tools_option_retrieval.ipynb
   ├── api_client/NLP_assignment_api_client/
   ├── src/*.py
   └── indexes/simplewiki*.joblib, kelm*.joblib, *dense*.index, *dense*meta.joblib
   ```

2. Crea Colab Secrets (`Runtime > Manage sessions > Secrets`):
   - `HF_TOKEN`: token Hugging Face per scaricare Qwen3.5-9B
   - `USERNAME`: account PoliMillionaire
   - `PASSWORD`: password

3. Apri il notebook 12 in Colab e esegui le sezioni:
   - Sezioni 1-7: setup dipendenze, GPU, paths
   - Sezioni 8-9: carica retrieval stack, reranker e option-wise retrieval
   - Sezioni 10-13: carica tool validati, policy di routing, test e API loop

**Tempo stimato:** 5-10 min setup, poi ~10 sec/domanda durante il gioco.

## Architettura

La pipeline RAG segue il flusso:

```
Question + Opzioni
    ↓
[Maths branch? Solo per "Maths" competition]
  ├─ Deterministic router: pattern ricorrenti + rule-based shortcuts
  ├─ Validated generic tools: JSON schema → semantic guard → SymPy/Python
  ├─ Deterministic option matching: confronto numerico/testuale con le opzioni
  └─ Fallback: un solo JSON router validato, poi Qwen direct Maths in `/no_think`
    ↓
[Knowledge branch] Retrieval ibrido
  ├─ SimpleWiki BM25 (sparse) + Dense HNSW (dense)
  ├─ KELM BM25 (sparse) + Dense HNSW (dense)
  ├─ Reciprocal Rank Fusion (RRF) su 4 ranking
  ├─ Cross-encoder reranking (BERT, CPU)
  └─ Option-wise retrieval per Entertainment e domande con evidenza debole
    ↓
[Scoring & Selection]
  ├─ Top K evidenza fornita al LLM
  ├─ Score/margin retrieval + option evidence scores
  ├─ Qwen3.5-9B in `/no_think` predice option_id
  └─ Fallback tracciati: parser opzioni, risposta diretta, prima opzione solo come ultima difesa
    ↓
Option ID → API → Logging CSV
```

**Componenti:**
- **Indici:** SimpleWiki (434k docs, 160w), KELM (500k asserzioni corte), libri di matematica (PDF → chunks)
- **Retrieval:** BM25 (sparse, veloce) + embedding densi (HNSW, accurato)
- **Reranking:** Cross-encoder MiniLM per riordinamento top-K
- **Option-wise retrieval:** per ogni opzione costruisce una query dedicata, recupera evidenza e salva score/margine nei log.
- **Math tools:** calculator, equation solver, modular arithmetic, independent-trials probability, binomial/proportion tests, normal utilities, permutation max order, finite abelian group count, combinatorics, geometry e concept classifier.
- **LLM fallback:** Qwen3.5-9B (9B params, 7.6 GiB Q6_K_L quantized). La modalità `/no_think` resta attiva per RAG, tool JSON e fallback diretto Maths; l'esperimento `/think` e stato scartato per latenza e output non parsabile.

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
|   |-- notebooks/                    # notebook progressivi 00-12
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
| **10** | Agentic math tools | Come 09, math con tool-router JSON + fallback Maths breve in `/no_think` | Baseline agentica, utile per confronto | ✓ |
| **11** | Router-hardened Maths | Come 10, ma con parser JSON robusto, stop fix, regole deterministic/statistics estese e tool coverage migliore | Ablation per hardening del 10 | ✓ |
| **12** | Validated tools + option retrieval | Tool layer generico validato, semantic guards, matching deterministico e retrieval per opzione | Punto di partenza consigliato per nuove prove | ✓ |

### Problemi osservati nei notebook 10 e 11

- **Notebook 10:** il router/planner JSON delegava troppo al modello. Nei log compaiono risposte non JSON, JSON parziali o tool call semanticamente sbagliate; il fallback Maths diretto era breve e poteva troncare ragionamenti utili, mentre la coverage dei tool restava legata a casi specifici.
- **Notebook 11:** risolve parte del parsing e dei timeout, ma mantiene un planner LLM costoso e fragile. Nei trace Maths alcuni tool venivano scelti con regole troppo permissive, per esempio theorem/concept lookup usato anche quando serviva un calcolo strutturato; inoltre mancavano guardie semantiche forti e un matching deterministico uniforme verso le opzioni.
- **Retrieval in 10/11:** la query globale funziona bene quando il documento recuperato contiene già domanda e risposta, ma su Entertainment e domande fattive con evidenza debole può non recuperare prove per tutte le opzioni. Questo rende difficile distinguere opzioni simili e capire dal log perche una risposta e stata scelta.
- **Logging in 10/11:** latenza, strategia e output grezzo sono presenti, ma mancano campi diagnostici piu utili per debug: confidence, margine retrieval, score per opzione, tool accettati/scartati e motivo dello scarto.

### Notebook 12: cosa introduce

Il notebook 12 mantiene la struttura Colab self-contained e non usa LangChain. L'obiettivo e avere tool generici e verificabili, non patch specifiche per singole domande:

- **ToolSpec-like layer:** ogni tool ha schema, parser, semantic guard, esecuzione controllata e risultato strutturato.
- **Maths generico:** include power sums complessi, probabilita su prove indipendenti, aritmetica modulare, test binomiali/proporzioni, distribuzione normale, massimo ordine di permutazione, conteggio di gruppi abeliani finiti, combinatorica, geometria e classificazione concettuale.
- **Router piu conservativo:** prima regole deterministiche e tool validati, poi un singolo router JSON validato, poi fallback diretto Qwen.
- **Option-wise retrieval:** per Entertainment e casi con evidenza debole recupera contesto per ciascuna opzione e salva score/margini.
- **Log piu leggibili:** aggiunge confidence, retrieval score summary, option evidence scores, tool traces, rejection reasons e fallback indicators.

**Raccomandazione:** per nuove prove parti dal **notebook 12**; tieni **10** e **11** come baseline/ablation per mostrare il percorso di miglioramento.

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

I file prodotti sono `*_200w_dense_hnsw.index` e `*_200w_dense_meta.joblib` in `data/indexes/`. Per usare il notebook 12 in Colab, copiarli anche in `/content/drive/MyDrive/nlp26/indexes`.

## State-of-the-Art: Ricerca scientifica che supporta il design

Il notebook 12 applica principi consolidati da lavori recenti in NLP e tool-augmented reasoning:

| Principio | Riferimento | Applicazione |
| --- | --- | --- |
| **RAG per domande fattive** | Lewis et al. 2020 *Retrieval-Augmented Generation* | Retrieve evidenza → LLM answer |
| **Chain-of-Thought prompting** | Wei et al. 2022 *Prompting CoT* | Valutato sperimentalmente; non usato come default operativo per latenza su Maths |
| **ReAct: Reasoning + Acting** | Yao et al. 2022 | Il modello propone un'azione, Python valida schema/guard ed esegue |
| **Structured tool calls** | Schick et al. 2023 *Toolformer* | Tool call JSON con schema validation e rejection reasons |
| **Program of Thoughts** | Chen et al. 2022 *PoT Prompting* | Math → calcolo deterministico con Python/SymPy |
| **Conservative fallback** | Design patterns from tool-use lit. | Nessun tool valido → option-wise/global RAG o fallback diretto |

## Strategie implementate (Notebooks 0-12)

- **Baseline:** API e CSV logging end-to-end
- **Sparse IR:** TF-IDF, BM25 con varianti (stopword, title-boost)
- **Multi-index:** Reciprocal Rank Fusion su piu corpora
- **KELM retrieval:** Knowledge graph strutturato per fattualita
- **Dense retrieval:** HNSW embeddings (MiniLM-L6)
- **Neural reranking:** Cross-encoder BERT (MiniLM-L-6-v2)
- **Agentic tools (v1, Nb 09):** Regex pattern matching → SymPy
- **Agentic tools (v2, Nb 10):** LLM JSON router/planner in `/no_think` → tool registry → fallback diretto Maths breve in `/no_think`
- **Router hardening (v3, Nb 11):** parser JSON robusto, fix stop token diretto, planner più corto, theorem rules/statistics rules e tool coverage estesi
- **Validated tools + option retrieval (v4, Nb 12):** tool call con schema/guard, matching deterministico, rejection reasons e retrieval per opzione

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

Per rispondere alla consegna, il notebook scelto per il run deve mostrare:

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
- [ ] Usa notebook **12** come base per nuove prove; conserva 10 e 11 come baseline/ablation
- [ ] Self-contained: nessuna dependency esterna fuori pip
- [ ] Colab Secrets per USERNAME/PASSWORD (NO hardcoded credentials)
- [ ] Google Drive path ben documentato
- [ ] Runnable da inizio a fine senza interruzioni
- [ ] Almeno **N=5 tentativi per competizione** per avere statistica valida
- [ ] CSV logs salvati su Drive al termine

### Analisi e risultati
- [ ] Summary cell con metriche riassuntive (accuracy, latenza, timeout count)
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
- [ ] Notebook 12 (`.ipynb` o link Drive al notebook usato)
- [ ] Breve README di setup (Colab secret names, paths, cose da modificare)
- [ ] Link video presentazione
- [ ] CSV dei risultati (facoltativo, per referenza)

## Troubleshooting

| Problema | Soluzione |
| --- | --- |
| `ModuleNotFoundError: millionaire_client` | Assicura che `api_client/NLP_assignment_api_client` sia in `sys.path` prima di `import` |
| API unreachable (PoliMi WiFi) | Usa VPN o rete mobile; segnala al docente se persiste |
| Colab non assegna GPU o quota esaurita | Cambia account/runtime o attendi il reset della quota; prima di caricare Qwen esegui `!nvidia-smi` e verifica che compaia una T4 |
| CUDA OOM su Colab | Riduci `n_gpu_layers` nel caricamento Qwen (prova 35 invece di -1) oppure passa a Q5_K_L |
| `Failed to load model from file` su GGUF | Verifica size/header del file. Il notebook 12 pinna la revision Hugging Face usata dai run riusciti del notebook 11, per evitare re-upload del branch `main` non compatibili con il wheel `llama-cpp-python` installato |
| Timeout su 30s | Riduci `TOP_K_RERANK`, `LLM_CONTEXT_K` o disabilita fallback LLM non essenziali |
| CSV parse error in logging | Assicura que `json.dumps(..., ensure_ascii=False)` per testo UTF-8 |
| Dense index missing su Drive | Esegui notebook 07 (build_dense_embeddings) o scarica da backup |
| LLM output non parseable | Notebook 12 valida JSON e tool call; se il tool viene scartato, salva il motivo e passa a fallback controllati |
| Maths fallback troppo verboso o lento | Notebook 12 prova prima tool deterministici validati e usa Qwen diretto solo come fallback |

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

**Aggiornamento locale:** aggiunto Notebook 12 con validated tools e option-wise retrieval  
**Team:** NeuroniNegroni (Tommaso, Giulia, Gio)
