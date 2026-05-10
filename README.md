# PoliMillionaire NLP 2026

Progetto per il corso di Natural Language Processing 2025/26 del Politecnico di Milano.

L'obiettivo e costruire e valutare un chatbot capace di giocare a **Who wants to be a PoliMillionaire?**, usando la API testuale del gioco e rispondendo a domande multiple-choice entro il timeout previsto. La soluzione sviluppata in questo repository combina retrieval locale, ranking lessicale, indici su basi di conoscenza esterne e strumenti deterministici per domande matematiche.

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

La consegna e in [docs/assignment/GroupAssignment2026.docx](docs/assignment/GroupAssignment2026.docx). La scadenza indicata nel documento e **2 giugno 2026 alle 23:00** via WeBeep.

## Architettura del progetto

La pipeline e pensata per funzionare su Google Colab o in locale:

1. la API PoliMillionaire fornisce domanda e opzioni;
2. un router decide se usare tool deterministici, retrieval o una strategia custom;
3. gli indici locali recuperano evidenza da SimpleWiki, KELM o libri di matematica;
4. le opzioni vengono confrontate tramite scoring lessicale, BM25/TF-IDF e, nei notebook piu avanzati, componenti neurali;
5. la risposta viene inviata alla API;
6. ogni decisione viene salvata nei log CSV per analisi successiva.

Per le competizioni di matematica, `project/src/agentic_tools.py` usa regole e SymPy per intercettare casi calcolabili e scegliere direttamente l'opzione corretta quando possibile. Per le domande di conoscenza generale, `project/src/retrieval_quiz_runner.py` usa retrieval locale e scoring delle opzioni.

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

## Notebook principali

I notebook in `project/notebooks/` documentano lo sviluppo progressivo:

| Notebook | Scopo |
| --- | --- |
| `00_api_smoke_test.ipynb` | test iniziale della API |
| `01_quiz_tfidf_no_llm.ipynb` | baseline retrieval TF-IDF senza LLM |
| `02_quiz_bm25_no_llm.ipynb` | baseline BM25 senza LLM |
| `03_quiz_bm25_multi_index_no_llm.ipynb` | BM25 su piu indici |
| `04_quiz_tfidf_multi_index_no_llm.ipynb` | TF-IDF su piu indici |
| `05_quiz_bm25_multi_index_bert_no_llm*.ipynb` | retrieval con reranking BERT |
| `06_quiz_bm25_bert_llm_agentic_tools_colab.ipynb` | pipeline RAG, BERT, LLM locale e tool |
| `07_build_dense_embeddings_colab.ipynb` | costruzione embeddings densi |
| `08_hybrid_pipeline.ipynb` | pipeline ibrida completa |
| `09_hybrid_pipeline_math_tools.ipynb` | pipeline ibrida con focus su tool matematici |

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

I file prodotti sono `*_200w_dense_hnsw.index` e `*_200w_dense_meta.joblib` in `data/indexes/`. Per usare il notebook 09 in Colab, copiarli anche in `/content/drive/MyDrive/nlp26/indexes`.

## Strategie implementate

- **Baseline API e logging**: controllo end-to-end del gioco e salvataggio telemetria.
- **TF-IDF retrieval-only**: baseline lessicale veloce.
- **BM25 retrieval-only**: baseline IR principale, con varianti su stopword, bigrammi e title boost.
- **Multi-index retrieval**: fusione dei risultati da piu corpora con Reciprocal Rank Fusion.
- **KELM retrieval**: subset locale per conoscenza fattuale strutturata.
- **Dense retrieval**: embeddings e indice HNSW costruiti nei notebook Colab.
- **BERT reranking**: riordinamento dei candidati recuperati.
- **Agentic tools**: tool deterministici per matematica, algebra e calcolo simbolico.
- **Pipeline ibrida**: combinazione di retrieval, reranking, tool e modello locale.

## Log e analisi

Gli esperimenti salvano CSV in `logs/`. I campi principali includono:

- competizione, sessione, tentativo, livello e domanda;
- opzioni e risposta scelta;
- sorgente della decisione (`retrieval`, tool, LLM, ecc.);
- evidenza recuperata;
- correttezza, premio, timeout e latenza;
- output grezzo di eventuali tool o modelli.

Gli script `project/src/analyze_bm25_results.py` e `project/src/analyze_tfidf_results.py` producono analisi e grafici comparativi. Alcune figure gia generate sono in `reports/figures/`.

## Valutazione suggerita

Per rispondere bene alla consegna, il notebook finale dovrebbe mostrare:

- accuratezza e livello medio raggiunto per ogni strategia;
- latenza media e massimo tempo per domanda;
- numero di timeout;
- confronto tra baseline, retrieval-only, RAG, reranking, tool e LLM locale;
- analisi per tipo di domanda o competizione;
- esempi concreti di errori;
- impatto del contesto RAG e dei prompt;
- contributo dei tool matematici sulle domande calcolabili;
- limiti della soluzione e possibili miglioramenti.

## Note operative

- Non inserire password nel notebook: usare secret o variabili d'ambiente.
- Caricare gli indici una sola volta a inizio partita.
- Limitare il numero di snippet nel prompt per ridurre latenza e rumore.
- Validare sempre l'output del modello: se non e un `option_id` valido, usare un fallback.
- Non stressare il server con molte partite consecutive.
- Misurare sempre il tempo prima di inviare la risposta.
