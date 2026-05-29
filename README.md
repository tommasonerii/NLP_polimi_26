# PoliMillionaire NLP 2026

**Progetto NLP 2025-26 @ Politecnico di Milano**

Chatbot che gioca a **Who Wants to Be a PoliMillionaire?** usando solo modelli open-weights eseguiti localmente. Il sistema combina retrieval augmented generation (RAG), ranking lessicale/neurale, tool-augmented reasoning per matematica e fallback robusti—tutto entro il vincolo di 30 secondi per domanda.

**Stack consigliato (Notebook 12 V7):**
- 📚 Retrieval: SimpleWiki + KELM (BM25 + Dense HNSW) + option-wise evidence retrieval + indice BM25S temporaneo per fonti esterne
- 🔀 Fusione: Reciprocal Rank Fusion su 4 indici
- 🧠 Reranking: `Qwen/Qwen3-Reranker-0.6B` via `sentence-transformers` CrossEncoder, su `cuda:1` quando sono disponibili 2 GPU
- 🧮 Math: deterministic router + analysis-first JSON router + validated generic tools estesi + SymPy + Micro-CoT fallback vincolato
- 🌐 External retrieval: Wikipedia API e News/Tavily/RSS come fonti primarie per categorie dove il corpus locale e debole o obsoleto, con gate semantico sui risultati Wikipedia
- 📰 News: retrieval live multi-sorgente (Google News RSS US+UK + Tavily in parallelo), indicizzazione BM25S temporanea, prompt headline-aware e regole anti-trappola
- 🤖 LLM: Qwen3.5-9B (`Q8_0` GGUF via llama-cpp-python), con `tensor_split` quando sono visibili piu GPU
- 🎯 Output constraints: GBNF per option id finale, JSON schema constraint per router Maths e regex robusta su `FINAL_CHOICE`
- 📊 Logging: CSV con latenza, strategia, evidenza, confidence, tool traces, rejection reasons, modalita retrieval, conteggi external docs/chunks e correttezza

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

**Per usare il notebook 12 V7 in Colab:**

1. Copia il notebook 12 V7 su Google Drive:
   ```text
   MyDrive/nlp26/
   ├── notebooks/12_V7_validated_tools_option_retrieval.ipynb
   ├── api_client/NLP_assignment_api_client/
   ├── src/*.py
   └── indexes/simplewiki*.joblib, kelm*.joblib, *dense*.index, *dense*meta.joblib
   ```

2. Crea Colab Secrets (`Runtime > Manage sessions > Secrets`):
   - `HF_TOKEN`: token Hugging Face per scaricare Qwen3.5-9B
   - `USERNAME`: account PoliMillionaire
   - `PASSWORD`: password
   - `TAVILY_API_KEY`: API key Tavily (free tier) per il retrieval news

3. Apri il notebook 12 V7 in Colab e esegui le sezioni:
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
  ├─ Deterministic router: pattern ricorrenti + rule-based shortcuts dai log
  ├─ Validated generic tools estesi: JSON schema → semantic guard → SymPy/Python
  ├─ Deterministic option matching: confronto numerico/testuale con le opzioni
  └─ Fallback: router JSON analysis-first, poi Qwen Micro-CoT con `FINAL_CHOICE`
    ↓
[External-primary branch? News/Wikipedia categories]
  ├─ News: Google News RSS + Tavily fetch parallelo
  ├─ Wiki categories: Wikipedia API extracts larghi
  ├─ Chunking dei documenti esterni
  ├─ Indice BM25S temporaneo per domanda
  ├─ Query globale + query option-wise su question + option
  └─ Se l'evidenza esterna e valida: prompt solo su external chunks
    ↓
[Local Knowledge branch] Default o fallback se external vuoto
  ├─ SimpleWiki BM25 (sparse) + Dense HNSW (dense)
  ├─ KELM BM25 (sparse) + Dense HNSW (dense)
  ├─ Reciprocal Rank Fusion (RRF) su 4 ranking
  ├─ Cross-encoder reranking (BERT, CPU)
  └─ Option-wise retrieval adattivo: Entertainment/History sempre, Science solo se incerta
    ↓
[Scoring & Selection]
  ├─ Top K evidenza fornita al LLM
  ├─ Score/margin retrieval + option evidence scores
  ├─ Qwen3.5-9B in `/no_think` predice option_id vincolato a un singolo digit 0-3
  └─ Fallback tracciati: parser opzioni, risposta diretta, prima opzione solo come ultima difesa
    ↓
Option ID → API → Logging CSV
```

**Componenti:**
- **Indici:** SimpleWiki (434k docs, 160w), KELM (500k asserzioni corte), libri di matematica (PDF → chunks)
- **Retrieval:** BM25 (sparse, veloce) + embedding densi (HNSW, accurato)
- **External BM25S temporaneo:** per News/Wikipedia indicizza solo i documenti recuperati live nella domanda corrente; non salva niente su disco e non mescola di default fonti esterne con SimpleWiki/KELM.
- **Reranking:** `Qwen/Qwen3-Reranker-0.6B` riordina sia i candidati locali sia i chunk esterni prodotti dall'indice BM25S temporaneo.
- **Option-wise retrieval:** per ogni opzione costruisce una query dedicata, recupera evidenza e salva score/margine nei log.
- **Math tools:** calculator, equation solver, modular arithmetic, independent-trials probability, binomial/proportion tests, normal utilities, permutation max order, finite abelian group count, combinatorics, geometry e concept classifier.
- **LLM fallback:** Qwen3.5-9B `Q8_0` GGUF. La modalità `/no_think` resta attiva per RAG, tool JSON e fallback diretto Maths; l'esperimento `/think` e stato scartato per latenza e output non parsabile.

## News pipeline (V4)

L'analisi degli errori sui log V2 e V4 ha mostrato che i fallimenti nella categoria News non erano dovuti al reasoning del modello ma al **retrieval**: nei casi sbagliati l'articolo corretto non veniva mai recuperato oppure la risposta era contenuta solo nel titolo, mai nel corpo dell'articolo passato al LLM. Le pipeline pensate per SimpleWiki/KELM non coprono notizie recenti, quindi V4 introduce una pipeline verticale dedicata alla categoria News.

**Flusso News end-to-end:**

1. **Query generation con LLM:** il modello, dato question + opzioni, produce una query di ricerca concisa orientata a recuperare la notizia rilevante.
2. **Fetch parallelo multi-sorgente:**
   - **Google News RSS** interrogato su regioni `US` e `UK` in parallelo, con decodifica dell'URL di redirect Google News e pulizia HTML del contenuto dell'articolo.
   - **Tavily** come seconda sorgente indipendente, interrogata in parallelo per aumentare la copertura quando RSS non recupera il documento giusto.
3. **Article ranking:** gli articoli recuperati sono ordinati per keyword overlap rispetto a question + opzioni, includendo il **titolo** nel testo da cui calcolare l'overlap (così le notizie la cui risposta vive solo nel titolo restano selezionabili). Si tengono i **top 3** articoli.
4. **Prompt headline-aware:** il prompt finale espone esplicitamente `HEADLINE` + corpo per ciascun articolo, in modo che il modello possa basarsi sul titolo quando il corpo non riporta esplicitamente il fatto.
5. **Answering Chain-of-Thought:** il modello prima estrae il fatto rilevante dagli articoli e poi sceglie l'opzione. Se dichiara esplicitamente che gli articoli non contengono la risposta, si attiva il **fallback al prompt vincolato** (GBNF su option id) usato anche per le altre categorie.

**Vincoli dell'assignment:** Tavily rispetta i vincoli della consegna — è gratuito (free tier) e restituisce contenuto grezzo non generato, non risposte LLM. Il reasoning resta interamente locale su Qwen3.5-9B.

## External ephemeral retrieval (V6)

Il notebook `project/notebooks/12_V6_validated_tools_option_retrieval.ipynb` estende V4 con una modifica strutturale: quando vengono usate fonti esterne, queste non sono piu semplici documenti aggiunti al contesto locale, ma diventano una **modalita di retrieval separata**.

L'idea e che Wikipedia, News RSS e Tavily hanno una semantica diversa rispetto a SimpleWiki/KELM:

- per **News**, il corpus locale e spesso obsoleto per definizione;
- per **Entertainment** e **Ancient History and Politics**, Wikipedia puo fornire pagine piu mirate e complete;
- mischiare documenti locali ed esterni con RRF o concatenazione puo introdurre rumore, perche una fonte statica puo vincere per overlap lessicale anche quando la fonte esterna contiene l'evidenza corretta.

V6 usa quindi questa policy:

```text
News:
  fetch Google News RSS + Tavily
  se external docs validi -> BM25S temporaneo -> Qwen3 rerank -> prompt solo external
  altrimenti -> fallback local RAG

Entertainment / Ancient History and Politics:
  fetch Wikipedia API
  se external docs validi -> BM25S temporaneo -> Qwen3 rerank + option-wise evidence
  altrimenti -> fallback local RAG

Altre categorie:
  local RAG come nei notebook precedenti
```

### Come funziona l'indice temporaneo

Per ogni domanda, V6 costruisce un indice in RAM sui soli documenti esterni appena recuperati:

1. Fetch piu ampio dei dati grezzi:
   - Wikipedia extract fino a circa `14000` caratteri;
   - Google News fino a piu articoli e circa `8000` caratteri per articolo;
   - Tavily fino a piu risultati raw.
2. Chunking dei testi in finestre sovrapposte.
3. Boost del titolo nel testo indicizzato, soprattutto per News, dove la risposta e spesso nel titolo.
4. Costruzione di `ExternalEphemeralBM25S`.
5. Retrieval globale con la sola domanda.
 6. Reranking dei candidati BM25S con `Qwen/Qwen3-Reranker-0.6B`.
 7. Retrieval option-wise con query `question + option_text`, stesso indice temporaneo e reranker.
 8. Prompt finale con pochi chunk selezionati, non con tutto il testo recuperato.

Questo mantiene alto il recall dei fallback esterni senza aumentare troppo il prompt. Il costo computazionale dell'indice temporaneo e piccolo rispetto a HTTP fetch, reranking e inferenza Qwen; il vincolo principale resta evitare troppe chiamate esterne, non indicizzare troppi chunk.

### Differenza rispetto a V4

In V4 i documenti esterni venivano aggiunti ai documenti locali:

```python
docs = docs + wiki_docs
docs = news_docs + docs
```

In V6, invece:

- se l'external retrieval produce evidenza usabile, il prompt usa **solo external evidence**;
- SimpleWiki/KELM restano fallback, non competitor alla pari;
- l'indice BM25S viene costruito una volta per domanda e interrogato piu volte;
- i log distinguono `retrieval_mode = external_bm25s_ephemeral` da `local_rag`.

Nuovi campi diagnostici nel CSV V6:

- `retrieval_mode`
- `external_docs_count`
- `external_chunks_count`
- `external_sources`
- `external_index_error`
- `external_fetch_error`
- `reranker_model`
- `reranker_max_length`
- `reranker_batch_size`

Il log V6 viene salvato in:

```text
logs/run_qwen35_q8_qwen3reranker06b_external_bm25s_v6.csv
```

## Knowledge pipeline — Unified retrieval + answer-first micro-reasoning (V5-Kaggle)

Il notebook `v5-kaggle.ipynb` introduce una pipeline alternativa per le categorie knowledge (Entertainment, History, Science, Philosophy), sviluppata in parallelo al V6. Gira su Kaggle con T4 singola, usa Qwen3.5-9B Q6_K_L e il cross-encoder MiniLM originale.

L'analisi degli errori sulla categoria Entertainment ha rivelato due problemi nel design precedente:

**Problema 1: option-wise retrieval controproducente.** Il retrieval per-opzione (4 query aggiuntive per domanda) produceva margini quasi zero quando il corpus non conteneva il fatto cercato. In questi casi il segnale era rumore e peggiorava la decisione rispetto al solo giudizio del modello. V5-Kaggle elimina completamente l'option-wise retrieval per le categorie knowledge.

**Problema 2: il modello ragionava correttamente ma non riusciva a scrivere la risposta.** Un primo tentativo di aggiungere Chain-of-Thought con tag `<think>` e 512 token di budget causava troncamento sistematico: il modello scriveva ~650 token di analisi e non raggiungeva mai il tag `ANSWER:`. 9 errori su 10 erano troncamento, non ragionamento sbagliato. L'accuracy è crollata dal 80% al 44%.

**Soluzione: answer-first micro-reasoning.** Il prompt forza il modello a emettere il digit di risposta come primo token, poi una breve giustificazione. Se il modello viene troncato, la risposta è comunque disponibile. Budget: 300 token → ~10s di generazione + ~8s retrieval = ~18s totale, dentro il limite di 30s.

**Pipeline V5-Kaggle per Entertainment/History/Science/Philosophy:**

```
Question + Opzioni
    ↓
[Retrieval unificato multi-sorgente]
  ├─ Indici locali: SimpleWiki + KELM (BM25 + Dense) → RRF → top-5
  ├─ Wikipedia API: query generata dall'LLM → extract + chunk rilevanti
  └─ Tavily API: ricerca general → contenuto raw
    ↓
[Rerank unificato] Cross-encoder su TUTTI i documenti (locali + esterni)
  → filtra per soglia di qualità (reranker score > -2.0) → top-6 documenti
    ↓
[Prompt answer-first] Il modello vede l'evidenza e risponde:
  "ANSWER: X\nREASON: one sentence why"
    ↓
[Parser] Prende il primo digit 0-3 → option id
```

**Risultati:** Entertainment è passato da $4,000 a **$1,024,000**, con accuracy dell'84.7% su 59 domande e una partita perfetta 15/15. La latenza media è 16-17s.

Le pipeline Math e News restano invariate rispetto a V4.

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

Il client aggiornato supporta anche la modalita vocale del gioco:

```python
game = client.game.start(competition_id=comp_id, mode="speech")
question_audio = game.fetch_audio_question()
option_a_audio = game.fetch_audio_option_next()
```

L'audio viene scaricato come bytes WAV completi da endpoint HTTP, non come streaming MP3. Le opzioni audio vanno richieste in sequenza A-D con `fetch_audio_option_next()`; dopo la consegna possono essere riascoltate con `fetch_audio_option(index)`. Il timer della domanda speech parte dopo la richiesta dell'ultima opzione.

Per uno smoke test della modalita speech:

```powershell
C:\ProgramData\miniconda3\python.exe project/src/test_client_voice_mode.py --competition-id 0 --options 4 --test-replay --play --leaderboard
```

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
| **12** | Validated tools + option retrieval | Tool layer generico validato, semantic guards, matching deterministico e retrieval per opzione | Baseline V1 da confrontare con V2 | ✓ |
| **12 V2** | Constrained outputs + adaptive retrieval + Maths fixes | Come 12, ma con GBNF sull'option id finale, router JSON vincolato quando supportato, option-wise adattivo e fix deterministici Maths dai log | Baseline forte per confronto V4 | ✓ |
| **12 V4** | Analysis-first Maths + extended tools + Micro-CoT fallback | Come V2, ma con tool Maths generici estesi, router JSON con scratchpad non validato e fallback Maths con `FINAL_CHOICE` vincolato | Baseline forte pre-V6 | ✓ |
| **12 V6** | External BM25S + Qwen3 reranker | Come V4, ma con setup Kaggle/Colab da V5, Qwen3.5-9B Q8, `Qwen3-Reranker-0.6B`, News/Wikipedia external-primary, indice BM25S temporaneo e local RAG solo come fallback | Baseline external-primary pre-gating | ✓ |
| **12 V7** | V6 + Wiki semantic gate + News anti-trap prompt | Come V6, ma scarta Wikipedia esterna quando il reranker non trova supporto semantico e rafforza il prompt News contro keyword trap, confusione fonte/soggetto e causa/effetto | Notebook consigliato per i run finali | ✓ |
| **V5-Kaggle** | Unified retrieval + micro-reasoning | Notebook parallelo su Kaggle T4: elimina option-wise retrieval, unifica retrieval locale+Wikipedia+Tavily con rerank unico, e usa answer-first micro-reasoning. Math e News invariati da V4 | Alternativa per knowledge categories su single-GPU | ✓ |

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

### Problemi osservati nel notebook 12

Il log `logs/run_qwen35_gguf_validated_tools_option_retrieval.csv` ha evidenziato alcuni limiti pratici del notebook 12:

- **Maths non usava davvero i tool validati:** nel run analizzato tutte le domande Maths sono finite su `math_direct_llm_qwen35_gguf`; il router proponeva talvolta tool validi sintatticamente, ma non adatti al problema, oppure mancava il tool deterministico necessario.
- **Output finali troppo verbosi:** alcune risposte LLM iniziavano con l'id corretto ma continuavano con testo come `Wait`, `Explanation` o piu option id, aumentando latenza e fragilita del parsing.
- **Option-wise retrieval troppo rigido:** era attivo sempre per Entertainment e solo con soglia fissa sul top score negli altri casi; History restava spesso su retrieval globale anche quando l'evidenza era ambigua.
- **Confidence sovrastimata:** score retrieval alti potevano produrre confidence molto alta anche quando piu opzioni avevano evidenza simile o quando il documento recuperato contraddiceva l'opzione scelta.
- **Copertura Maths incompleta:** fallivano pattern concreti come durata sopra una soglia per una traiettoria quadratica, disequazioni intere con valori assoluti, trasformazioni statistiche additive e disegni fattoriali.

### Notebook 12 V2: correzioni

Il notebook `project/notebooks/12_V2_validated_tools_option_retrieval.ipynb` mantiene l'architettura del 12 ma corregge i problemi osservati nei log:

- **GBNF per risposta finale:** le chiamate finali RAG, option-wise e Maths direct usano una grammatica `root ::= [0-3]` quando supportata da `llama-cpp-python`, forzando un singolo option id.
- **Router Maths vincolato:** il router LLM usa `create_chat_completion(..., response_format=...)` con JSON schema quando disponibile; la validazione Python resta obbligatoria per schema, semantic guard e matching.
- **Adaptive option-wise retrieval:** Entertainment e History/Politics usano option-wise sempre; Science/Nature lo usa solo quando top score o margine indicano incertezza; Maths non lo usa come strategia principale.
- **Fix Maths deterministici:** aggiunti handler conservativi per traiettorie quadratiche sopra soglia, somme di soluzioni intere con valori assoluti, word problem delle pile, trasformazioni statistiche additive, experimental design fattoriale, correlazione e mutua esclusivita/indipendenza.
- **Confidence piu conservativa:** la confidence option-wise viene limitata quando il margine tra opzioni e piccolo, evitando stime troppo ottimistiche.
- **Log separato:** il V2 salva in `logs/run_qwen35_gguf_validated_tools_option_retrieval_v2.csv`, cosi il confronto con il 12 resta pulito.

### Notebook 12 V4: debolezze V2 e aggiunte

Il notebook `project/notebooks/12_V4_validated_tools_option_retrieval.ipynb` parte dal V2 e interviene sui fallimenti Maths osservati nei log V2, senza cambiare reranker o retrieval stack per non confondere gli esperimenti.

Debolezze rimaste nel V2:

- **Maths direct troppo cieco:** la grammatica `[0-3]` ha eliminato il parsing fragile, ma quando il router sceglieva `no_tool` il modello doveva calcolare internamente e rispondere con un solo token, senza spazio per un ragionamento minimo.
- **Router Maths ancora poco esplicito:** diversi problemi testuali venivano classificati come `no_tool` anche se erano riconducibili a equazioni, sistemi o probabilita gia risolvibili in Python.
- **Tool generici incompleti:** mancavano operazioni concrete come distanza da origine dopo movimenti cardinali, code/cumulative binomiali e normali, e piccoli sistemi di equazioni.
- **Diagnostica Maths migliorabile:** serviva una traccia piu chiara del motivo per cui un tool e stato accettato/scartato o di come ha calcolato il risultato.

Aggiunte del V4:

- **Tool Maths estesi:** `math_geometry` supporta `distance_from_origin`/cammini cardinali, `math_binomial_probability` supporta `at_most`, `at_least` e tail probability, `math_normal_distribution` supporta code upper/lower, e `math_solve_equation` accetta anche piccoli sistemi.
- **Router JSON analysis-first:** lo schema del router aggiunge `mathematical_analysis` prima di `tool_name` e `arguments`. Il campo serve solo come scratchpad del modello: la validazione Python continua a usare solo tool e argomenti.
- **Micro-CoT fallback per Maths:** il fallback diretto Maths non usa piu solo il digit secco; genera poche righe vincolate e termina con `FINAL_CHOICE: <0-3>`, estratto con regex robusta. Il limite e `MATH_MICRO_COT_MAX_TOKENS = 160` per ridurre il rischio di troncamento.
- **Smoke test deterministici:** una cella V4 testa i tool Python prima dell'API loop, cosi gli errori di parsing/SymPy/SciPy emergono prima di spendere chiamate di gioco.
- **Log separato:** il V4 salva in `logs/run_qwen35_gguf_validated_tools_option_retrieval_v4.csv`, mantenendo confrontabili V1, V2 e V4.

### Notebook 12 V6: external ephemeral BM25S + Qwen3 reranker

Il notebook `project/notebooks/12_V6_validated_tools_option_retrieval.ipynb` parte da V4, riprende la logica Kaggle/Colab del notebook V5 e cambia la gestione dei fallback esterni. Invece di concatenare Wikipedia/News ai risultati locali, costruisce un indice BM25S temporaneo sui documenti esterni recuperati per la domanda corrente.

Correzioni introdotte:

- **Setup Kaggle/Colab:** auto-detect dei path Kaggle, fallback Colab/local, secrets per API/HF/Tavily senza credenziali hardcoded.
- **Answer model Q8:** usa `Qwen_Qwen3.5-9B-Q8_0.gguf`; con piu GPU visibili il loader llama.cpp prova `tensor_split`.
- **Reranker forte:** sostituisce il MiniLM cross-encoder con `Qwen/Qwen3-Reranker-0.6B`, mantenendo batch e max length configurabili. Su Kaggle 2xT4 il reranker viene caricato su `cuda:1`, lasciando `cuda:0` al GGUF e alla GPU principale llama.cpp.
- **External-first per News:** Google News RSS e Tavily diventano la sorgente primaria quando producono documenti. Il local RAG viene usato solo se non ci sono risultati esterni usabili.
- **News low-evidence retry:** se il primo fetch News produce solo RSS, meno di 3 documenti o meno di 4 chunk, V6 rilancia un secondo fetch usando anche il testo delle opzioni nella query.
- **News option-wise fallback:** se il Chain-of-Thought News non restituisce una scelta affidabile, V6 interroga l'indice BM25S esterno per ogni opzione prima del fallback local/RAG.
- **Runner per categoria:** in fondo al notebook ci sono celle separate per setup client e per Ancient History and Politics, Entertainment, Maths, News, Philosophy and Psychology, Science and Nature; per smoke test su tutte usare `SELECTED_ATTEMPTS = 1`.
- **External-first per Wikipedia categories:** Entertainment e Ancient History/Politics usano Wikipedia API come sorgente primaria quando disponibile.
- **Fetch piu ampio:** il sistema recupera piu testo grezzo per pagina/articolo, poi lo filtra con BM25S invece di tagliarlo subito.
- **Indice temporaneo per domanda:** `ExternalEphemeralBM25S` indicizza chunk dei documenti esterni in RAM, senza persistenza su disco.
- **Option-wise external evidence:** per ogni opzione viene interrogato lo stesso indice temporaneo con `question + option_text`, poi i candidati vengono riordinati dal reranker.
- **Niente fusione local/external di default:** se l'external evidence e valida, il prompt usa solo quei chunk. Questo evita che SimpleWiki/KELM introducano match lessicali vecchi o generici.
- **Logging esteso:** il CSV V6 salva modalita retrieval, modello reranker, parametri reranker, numero di documenti esterni, numero di chunk, sorgenti esterne e possibili errori di fetch/index.

### Debolezze osservate nel notebook 12 V6

I log V6 mostrano che il salto a external-first migliora News e molte domande fattive, ma introduce due fragilita ricorrenti:

- **Wikipedia external troppo permissivo:** Entertainment e Ancient History/Politics usavano Wikipedia come sorgente primaria anche quando il reranker assegnava punteggi semantici molto negativi. In quei casi il BM25S trovava pagine con parole in comune, ma fuori contesto, e il local RAG non veniva piu consultato.
- **Distrazione da omonimie o pagine vicine:** quando la pagina esterna non era davvero centrata sulla domanda, il modello tendeva a seguire il testo recuperato invece della conoscenza piu semplice gia disponibile.
- **News con evidenza buona ma mapping fragile:** diversi errori non derivavano da mancanza di articoli, ma da interpretazione del ruolo richiesto dalla domanda: fonte che riporta la notizia vs soggetto della notizia, causa vs conseguenza, target/vittima/luogo, oppure fatto piu prominente dell'articolo invece del fatto chiesto.
- **Rimedi meno certi:** ripetere titoli, cambiare chunking o lanciare una seconda retrieve generalizzata puo aiutare alcuni casi, ma rischia di aumentare rumore e latenza. Nei log V6 le correzioni piu sicure sono un gate semantico conservativo per Wikipedia e istruzioni News piu esplicite su cosa non fare.

### Notebook 12 V7: correzioni conservative

Il notebook `project/notebooks/12_V7_validated_tools_option_retrieval.ipynb` parte da V6 e cambia solo i punti sopra:

- **Wiki semantic gate:** nelle categorie Wikipedia, l'evidenza esterna viene accettata solo se almeno un chunk ha supporto semantico non negativo dal reranker. Se tutti i punteggi sono negativi, il notebook scarta l'external path e torna al local RAG.
- **Prompt News anti-trappola:** il prompt News ora chiede prima il ruolo richiesto dalla domanda e aggiunge regole generiche su cosa evitare: non scegliere per keyword overlap, non confondere fonte/soggetto, causa/effetto, target/vittima/luogo, o un fatto correlato ma non richiesto.
- **Esperimento separato:** V7 usa `PROMPT_VERSION` dedicata e salva log in `logs/run_qwen35_q8_qwen3reranker06b_external_bm25s_v7.csv`, cosi il confronto con V6 resta pulito.

**Raccomandazione:** per i run finali parti dal **notebook 12 V7**; tieni **12 V6**, **12 V4**, **12 V2**, **12**, **11** e **10** come baseline/ablation per mostrare il percorso di miglioramento.

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

I file prodotti sono `*_200w_dense_hnsw.index` e `*_200w_dense_meta.joblib` in `data/indexes/`. Per usare il notebook 12 V7 in Colab, copiarli anche in `/content/drive/MyDrive/nlp26/indexes`.

## State-of-the-Art: Ricerca scientifica che supporta il design

Il notebook 12 V7 applica principi consolidati da lavori recenti in NLP e tool-augmented reasoning:

| Principio | Riferimento | Applicazione |
| --- | --- | --- |
| **RAG per domande fattive** | Lewis et al. 2020 *Retrieval-Augmented Generation* | Retrieve evidenza → LLM answer |
| **Chain-of-Thought prompting** | Wei et al. 2022 *Prompting CoT* | Usato solo nel fallback Maths Micro-CoT, con output vincolato e budget token limitato |
| **ReAct: Reasoning + Acting** | Yao et al. 2022 | Il modello propone un'azione, Python valida schema/guard ed esegue |
| **Structured tool calls** | Schick et al. 2023 *Toolformer* | Tool call JSON con schema validation e rejection reasons |
| **Program of Thoughts** | Chen et al. 2022 *PoT Prompting* | Math → calcolo deterministico con Python/SymPy |
| **Adaptive retrieval** | Self-RAG / search-augmented QA literature | News/Wikipedia usano external retrieval solo quando la categoria lo richiede; local RAG resta fallback |
| **Fast lexical retrieval** | BM25S | Indice temporaneo in RAM sui documenti esterni recuperati live |
| **Strong neural reranking** | Qwen3-Reranker | Cross-encoder 0.6B sui candidati locali e sui chunk BM25S esterni |
| **Conservative fallback** | Design patterns from tool-use lit. | Nessun tool valido → option-wise/global RAG o fallback diretto |

## Strategie implementate (Notebooks 0-12 V7)

- **Baseline:** API e CSV logging end-to-end
- **Sparse IR:** TF-IDF, BM25 con varianti (stopword, title-boost)
- **Multi-index:** Reciprocal Rank Fusion su piu corpora
- **KELM retrieval:** Knowledge graph strutturato per fattualita
- **Dense retrieval:** HNSW embeddings (MiniLM-L6)
- **Neural reranking:** da MiniLM cross-encoder a `Qwen/Qwen3-Reranker-0.6B` nel notebook V6
- **Agentic tools (v1, Nb 09):** Regex pattern matching → SymPy
- **Agentic tools (v2, Nb 10):** LLM JSON router/planner in `/no_think` → tool registry → fallback diretto Maths breve in `/no_think`
- **Router hardening (v3, Nb 11):** parser JSON robusto, fix stop token diretto, planner più corto, theorem rules/statistics rules e tool coverage estesi
- **Validated tools + option retrieval (v4, Nb 12):** tool call con schema/guard, matching deterministico, rejection reasons e retrieval per opzione
- **Constrained outputs + adaptive retrieval (v5, Nb 12 V2):** GBNF su risposta finale, router JSON vincolato quando supportato, option-wise adattivo e fix Maths dai log
- **Analysis-first Maths (v6, Nb 12 V4):** router Maths con scratchpad JSON, tool generici estesi, Micro-CoT fallback vincolato e smoke test deterministici
- **External ephemeral retrieval (Nb 12 V6):** setup Kaggle/Colab da V5, Q8 GGUF, Qwen3 reranker, Wikipedia/News external-first, BM25S temporaneo per domanda e option-wise retrieval su chunk esterni
- **Semantic gate + anti-trap News prompt (Nb 12 V7):** mantiene V6 ma blocca Wikipedia esterna semanticamente negativa e rende il prompt News piu robusto contro errori di ruolo, fonte e causalita
- **Unified retrieval + answer-first micro-reasoning (Nb V5-Kaggle):** retrieval multi-sorgente unificato (local + Wikipedia + Tavily), rerank cross-encoder su pool unico con soglia di qualità, eliminazione option-wise retrieval, answer-first micro-reasoning con budget 300 token. Entertainment raggiunge $1,024,000

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
- **Accuratezza** per ogni competizione (Entertainment, History, Science, Maths, News, Philosophy)
- **Livello medio raggiunto** per category
- **Latenza media** e **max latenza** per domanda
- **Numero timeout** (risposte oltre 30s)
- **Earned amount** medio per sessione

### Risultati per categoria (best earning sulla leaderboard)

| Categoria | Best earning |
| --- | --- |
| Entertainment | $1,024,000 |
| Philosophy | $1,024,000 |
| Science | $1,024,000 |
| History | $1,024,000 |
| News | $1,024,000 |
| Maths | $8,000 |

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
- [ ] Usa notebook **12 V7** come base per nuove prove; conserva 12 V6, 12 V4, 12 V2, 12, 11 e 10 come baseline/ablation
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
- [ ] Notebook 12 V7 (`.ipynb` o link Drive al notebook usato)
- [ ] Breve README di setup (Colab secret names, paths, cose da modificare)
- [ ] Link video presentazione
- [ ] CSV dei risultati (facoltativo, per referenza)

## Troubleshooting

| Problema | Soluzione |
| --- | --- |
| `ModuleNotFoundError: millionaire_client` | Assicura che `api_client/NLP_assignment_api_client` sia in `sys.path` prima di `import` |
| API unreachable (PoliMi WiFi) | Usa VPN o rete mobile; segnala al docente se persiste |
| Colab/Kaggle non assegna GPU o quota esaurita | Cambia account/runtime o attendi il reset della quota; prima di caricare Qwen esegui `!nvidia-smi` e verifica GPU e VRAM disponibili |
| CUDA OOM su V7 | Prima riduci `RERANKER_BATCH_SIZE`, `RERANKER_MAX_LENGTH` o `LLM_CONTEXT_K`; se non basta riduci `n_gpu_layers` nel caricamento Qwen o passa a una quantizzazione piu leggera |
| `Failed to load model from file` su GGUF | Verifica size/header del file. Il notebook 12 V7 mantiene la revision Hugging Face gia usata nei run riusciti, per evitare re-upload del branch `main` non compatibili con il wheel `llama-cpp-python` installato |
| Timeout su 30s | Riduci `TOP_K_RERANK`, `LLM_CONTEXT_K` o disabilita fallback LLM non essenziali |
| CSV parse error in logging | Assicura que `json.dumps(..., ensure_ascii=False)` per testo UTF-8 |
| Dense index missing su Drive | Esegui notebook 07 (build_dense_embeddings) o scarica da backup |
| LLM output non parseable | Notebook 12 V7 usa GBNF per l'option id finale e valida JSON/tool call; se un vincolo runtime non e supportato, passa a fallback controllati e tracciati |
| Maths fallback troppo verboso o lento | Notebook 12 V7 prova prima tool deterministici validati, usa router JSON analysis-first quando disponibile e limita il fallback diretto con Micro-CoT vincolato |
| External retrieval vuoto o rumoroso | Controlla `external_docs_count`, `external_chunks_count`, `external_sources`, `external_fetch_error` e `external_index_error` nel CSV V7; se external non e usabile, il notebook torna a `local_rag` |

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
6. Asai et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* [[arxiv]](https://arxiv.org/abs/2310.11511)
7. Vu et al. (2024). *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.* ACL Findings. [[paper]](https://aclanthology.org/2024.findings-acl.813/)
8. Lù (2024). *BM25S: Orders of magnitude faster lexical search via eager sparse scoring.* [[arxiv]](https://arxiv.org/abs/2407.03618)
9. Qwen Team (2025). *Qwen3 Embedding and Reranker model card.* [[huggingface]](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

---

**Aggiornamento locale:** aggiunto Notebook V5-Kaggle con unified retrieval multi-sorgente, answer-first micro-reasoning e Entertainment a $1,024,000. Notebook 12 V6 aggiornato con Qwen3-Reranker-0.6B, Q8_0 GGUF, retry News su evidenza povera, fallback option-wise esterno per News e `PROMPT_VERSION` V6 coerente. Notebook 12 V7 aggiunto con Wiki semantic gate, prompt News anti-trappola e log separato V7.
**Team:** NeuroniNegroni (Tommaso, Giulia, Gio)
