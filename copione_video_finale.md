# Copione video finale - PoliMillionaire NLP 2026

Durata target: **5 minuti massimo**.

Formato richiesto dal docente: **screen capture del notebook, senza slide, senza video accelerato**.

Notebook da mostrare come principale:

```text
project/notebooks/13_speech.ipynb
```

Messaggio chiave:

> Notebook 13 e la versione finale: prende la pipeline testuale V8, gia basata su RAG, Qwen locale, reranking e tool matematici, e aggiunge un adapter speech con Whisper. La modalita speech cambia solo il modo in cui la domanda entra nel sistema; il motore decisionale resta lo stesso.

## Prima di registrare

- Aprire `13_speech.ipynb`.
- Aggiungere o tenere visibili all'inizio le celle markdown con:
  - nomi e email dei membri del gruppo;
  - link al video;
  - statement su eventuali coding assistants usati;
  - sintesi architettura;
  - risultati principali;
  - limiti.
- Tenere aperto anche `README.md` o una cella del notebook con il diagramma della pipeline.
- Non eseguire live tutte le celle pesanti: mostrare il notebook, la pipeline e i log/output gia prodotti.
- Evitare di fare chiamate API live durante il video, a meno che sia una demo breve e gia testata.

## Mappa di scroll su notebook 13

Usare questa mappa mentre si registra. Il video deve sembrare un commento guidato al notebook, non una presentazione separata.

| Tempo | Sezione del notebook 13 da mostrare | Perche mostrarla |
| --- | --- | --- |
| 0:00-0:25 | Titolo iniziale e celle introduttive | Task, vincoli della consegna, notebook finale. |
| 0:25-0:55 | Sezioni 1-6: dependencies, Drive paths, GGUF model, LLM wrapper | Fondazione locale: modello open-weight, inferenza controllata, niente LLM API esterne. |
| 0:55-1:35 | Sezioni 7-9: embedding, BM25, HNSW, reranker, prompting, parsing | RAG statico e risposta vincolata a option id. |
| 1:35-2:15 | Sezione 10: Maths tools, shortcut deterministici, Python/SymPy fallback | Failure mode della matematica e soluzione tool-based. |
| 2:15-2:50 | Wikipedia/Tavily fallback e pipeline per Entertainment, History, News | Failure mode di domande ambigue o fresche: serve evidenza grezza esterna, ma ragionamento locale. |
| 2:50-3:25 | Sezione 11: routing policy, `answer_strategy`, categoria Maths | La parte R&D piu importante: strategia diversa per categoria, non un unico prompt. |
| 3:25-4:00 | Sezioni 13-14: API loop, smoke tests, run per categoria | Valutazione reale tramite API e log, non solo esempi scelti. |
| 4:00-4:35 | Sezione "V9 speech mode": Whisper adapter e run speech per categoria | Notebook 13 aggiunge speech sopra V8 senza cambiare il motore testuale. |
| 4:35-5:00 | Diagramma finale, risultati o limiti nel notebook/README | Chiusura: punti forti, trade-off, limiti. |

Frase guida da tenere a mente:

> The final notebook is modular because each module corresponds to a type of mistake we actually observed.

## Timeline video

Questa e' la versione da leggere. Target realistico: circa 5 minuti con ritmo naturale.

### 0:00 - 0:25 | Introduzione e vincoli

**Cosa mostrare nel notebook:** titolo, membri del gruppo, task summary.

**Da dire:**

> This is our final notebook for the NLP 2026 group assignment. The goal is to build a chatbot that plays Who Wants to Be a PoliMillionaire through the provided API.
>
> The important constraints are that the answer model must run locally, use open weights, avoid LLM APIs, and answer within the game timeout. Our final submission is Notebook 13.

### 0:25 - 0:55 | Notebook 13

**Cosa mostrare:** prime celle del notebook 13 e setup del modello.

**Da dire:**

> Notebook 13 is built on top of our best text-only pipeline, V8. The extra part is speech: Whisper transcribes the WAV question and options, then the same text pipeline answers the question.
>
> So speech changes only the input format. The reasoning engine, routing logic, retrieval and final answer selection stay the same.

### 0:55 - 1:45 | Architettura

**Cosa mostrare:** diagramma pipeline, poi sezioni retrieval e maths.

**Da dire:**

> The final system is modular because each module corresponds to a type of mistake we actually observed.
>
> For general knowledge, the local model alone was not robust enough. It sometimes confused similar entities or answered from incomplete context, so we added RAG. BM25 handles exact wording, dense retrieval handles semantic matches, and reranking filters noisy passages before the final model call.
>
> For Maths, retrieval was not enough. The model could often describe the right method but still make arithmetic or option-matching mistakes. So maths questions first use validated deterministic tools and Python or SymPy execution, with the local model only as a fallback.
>
> For News and fresh questions, external services are used only to retrieve raw evidence. They do not generate the answer.

### 1:45 - 2:25 | Evoluzione dei modelli

**Cosa mostrare:** celle di caricamento Qwen GGUF, embedding e reranker.

**Da dire:**

> The model stack also evolved. The first notebooks had no generative LLM: only TF-IDF, BM25, and then a MiniLM cross-encoder reranker.
>
> The first generative model was a small Qwen2.5 Instruct model, used mainly for tool routing and compact RAG fallback. Later we moved to a local Qwen3.5 9B GGUF backend, first with Q6 quantization and finally Q8 in V8 and Notebook 13.
>
> The final stack uses Qwen3.5 9B Q8 for local answer generation, MiniLM embeddings with HNSW for dense retrieval, BM25 for sparse retrieval, and Qwen3-Reranker for evidence selection.

### 2:25 - 3:15 | Routing e R&D

**Cosa mostrare:** `answer_strategy`, categoria Maths, API loop.

**Da dire:**

> The central function is `answer_strategy`. It chooses the strategy from the competition category.
>
> The main lesson from development was that one universal prompt was not enough. Different categories failed for different reasons: missing evidence, noisy evidence, arithmetic errors, final-format errors, or speech transcription errors.
>
> This is why the final pipeline uses routing. Maths goes through tools first. News uses fresh evidence. Static knowledge uses local RAG. The final answer is always constrained to one of the four valid option ids.

### 3:15 - 4:05 | Evaluation

**Cosa mostrare:** run per categoria, log summary o risultati README.

**Da dire:**

> We evaluated the system through API runs, not only with hand-picked examples. The logs store the selected strategy, latency, correctness, retrieved evidence and tool traces.
>
> We kept the previous notebooks as ablations, because the assignment evaluates both leaderboard performance and the investigation process. Across the saved logs, we reached at least one 1,024,000-dollar run for each category.
>
> The most important improvement was on Maths, where V8 combines validated tools, option matching and local fallback reasoning while keeping the run within the timeout.

### 4:05 - 4:45 | Speech e limiti

**Cosa mostrare:** sezione V9 speech mode e run speech.

**Da dire:**

> Speech was added only after the text pipeline was stable, so that ASR errors could be isolated from reasoning errors. The `SpeechGameAdapter` fetches audio, transcribes it with Whisper large-v3-turbo, rebuilds a text-compatible question, and calls the same competition runner.
>
> The main limitation of speech mode is latency and transcription quality, especially for mathematical notation and short option texts. In text mode, the main trade-off is between retrieval depth, reranking quality and generation time.

### 4:45 - 5:00 | Chiusura

**Cosa mostrare:** diagramma finale o titolo notebook.

**Da dire:**

> In summary, Notebook 13 is a local, tool-augmented RAG pipeline with a modular speech adapter. The final design is the result of the errors observed across the previous versions, rather than a single prompt around one model.

## Versione italiana alternativa

Se preferiamo registrare in italiano, usare questa traccia abbreviata.

### Apertura

> Questo e il nostro notebook finale per il progetto NLP. L'obiettivo e giocare a Who Wants to Be a PoliMillionaire rispettando i vincoli della consegna: modelli locali, open-weights, niente API LLM esterne, risposte entro circa 30 secondi e confronto tra piu soluzioni.

### Architettura

> La soluzione finale e il Notebook 13. Parte dalla pipeline testuale V8 e aggiunge la modalita speech. In speech mode, il sistema scarica gli audio WAV, trascrive domanda e opzioni con Whisper large-v3-turbo, e poi passa il testo alla stessa `answer_strategy` usata in modalita testuale.
>
> La pipeline ha tre rami principali: Maths, News e Knowledge. Maths usa tool validati, Python e SymPy, poi un fallback Qwen locale. News usa Google News RSS e Tavily come fonti grezze, non come generatori di risposte. Knowledge usa RAG locale su SimpleWiki, KELM e textbook, con BM25, dense HNSW, RRF e Qwen3-Reranker.

### Modelli

> Anche lo stack dei modelli e evoluto. All'inizio non usavamo LLM generativi: solo TF-IDF, BM25 e poi un reranker MiniLM cross-encoder. Il primo LLM generativo e stato Qwen2.5 Instruct piccolo, usato come router per tool e fallback RAG compatto.
>
> Nelle versioni successive siamo passati a Qwen3.5 9B GGUF locale, prima Q6 e poi Q8 nella versione finale V8/Notebook 13. Per gli embedding dense usiamo MiniLM con indici HNSW; per il reranking finale usiamo Qwen3-Reranker 0.6B al posto del MiniLM cross-encoder. Per speech usiamo Whisper large-v3-turbo, sempre localmente.

### Evoluzione

> Siamo partiti da baseline semplici: first option, TF-IDF e BM25 senza LLM. Poi abbiamo aggiunto reranking, retrieval denso e Qwen GGUF.
>
> Lo sviluppo e stato guidato soprattutto dagli errori nei log. Quando l'errore veniva da evidenza mancante o rumorosa, abbiamo lavorato sul retrieval. Quando veniva da calcoli o matching numerico, abbiamo smesso di chiedere al modello di calcolare e abbiamo aggiunto tool validati e SymPy. Quando veniva dal formato finale, abbiamo vincolato l'output a un option id valido.
>
> Per questo il notebook finale e modulare: ogni modulo corrisponde a un tipo di errore che abbiamo osservato davvero. Il V8 consolida la pipeline testuale, e il 13 aggiunge speech solo come adapter di input.

### Risultati e limiti

> Nei log salvati abbiamo almeno una run da 1,024,000 dollari per ogni categoria. Il miglioramento piu importante e su Maths: `run_v8.csv` arriva a 1,024,000, con 98 domande loggate, 79 corrette, accuracy circa 80.6 percento e zero timeout.
>
> La modalita speech e piu sperimentale: funziona, ma aggiunge latenza e puo introdurre errori di trascrizione, soprattutto su matematica e opzioni brevi.

### Chiusura

> In conclusione, il sistema finale e un agente RAG locale, con tool deterministici e un adapter speech modulare. Rispetta i vincoli della consegna e mostra chiaramente l'evoluzione dai baseline alla soluzione finale.

## Cose da non dire nel video

- Non dire che usiamo API LLM esterne.
- Non dire che Tavily genera risposte: dire sempre che restituisce contenuto grezzo.
- Non promettere che speech sia robusto quanto text mode.
- Non passare troppo tempo sui dettagli dei log: il video deve mostrare il notebook e l'architettura.
- Non fare live demo lunga con API/server durante il video.

## Domande probabili del docente

### Perche Qwen3.5 GGUF?

Perche dopo i baseline senza LLM e il primo Qwen2.5 piccolo, serviva un backend locale piu forte per ragionamento e RAG. Qwen3.5 9B GGUF resta open-weight ed eseguibile localmente; Q6 era un buon compromesso iniziale, mentre Q8 e diventato il backend finale per maggiore affidabilita.

### Perche non solo RAG?

Perche gli errori non erano tutti dello stesso tipo. Maths spesso richiede calcolo, non solo recupero. Inoltre alcune domande fallivano per evidenza rumorosa, altre per formato finale non valido. Per questo usiamo RAG dove serve evidenza, tool validati dove serve calcolo, e parsing vincolato per la scelta finale.

### Perche non fine-tuning?

Perche la maggior parte degli errori osservati non indicava chiaramente un problema dei pesi del modello. Spesso mancava il contesto giusto, il contesto era rumoroso, il calcolo era fragile, o la risposta finale non era nel formato corretto. Migliorare la pipeline era quindi piu controllabile, piu veloce da testare e piu coerente con i vincoli della consegna.

### Le API esterne violano la consegna?

No: Google News RSS, Wikipedia e Tavily vengono usati solo per recuperare testo grezzo. La risposta viene generata localmente dal nostro modello.

### Cosa cambia tra V8 e Notebook 13?

V8 e il motore testuale finale. Notebook 13 aggiunge speech mode con Whisper e costruisce una domanda testuale equivalente. La decisione finale usa la stessa pipeline.

### Qual e il collo di bottiglia principale?

In text mode, il collo di bottiglia e il trade-off tra retrieval/reranking e tempo di generazione. In speech mode, si aggiungono fetch audio, ASR e possibili errori di trascrizione.

### Cosa migliorereste?

Miglioreremmo il parsing speech per matematica, una cache per retrieval esterno, un gate piu robusto per fonti rumorose e una maggiore copertura dei tool matematici.
