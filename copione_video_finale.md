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

### 0:00 - 0:25 | Introduzione e vincoli

**Cosa mostrare nel notebook:** titolo, membri del gruppo, task summary, eventualmente prima cella markdown aggiunta all'inizio.

**Da dire:**

> This is our final notebook for the NLP 2026 group assignment. The goal is to build a chatbot that plays Who Wants to Be a PoliMillionaire through the provided API.
>
> The main constraints are: models must run locally, we cannot use LLM APIs, the models must be open-weights, and each answer has to be produced within about 30 seconds. We also evaluate different model and retrieval strategies, as requested by the assignment.

### 0:25 - 0:55 | Soluzione finale

**Cosa mostrare:** sezioni 1-6: dipendenze, path, download/caricamento Qwen GGUF, wrapper LLM.

**Da dire:**

> Our final solution is Notebook 13. It extends our best text pipeline, called V8, with a speech adapter.
>
> In text mode, the system receives a question and four options. In speech mode, it first downloads the WAV audio for the question and the options, transcribes them with Whisper large-v3-turbo, and then creates the same text object used by the V8 decision engine.
>
> So the speech component is modular: it changes the input representation, not the reasoning pipeline.
>
> The GGUF wrapper is still useful here because the rest of the notebook needs a stable local inference interface: fixed generation parameters, controlled output, answer parsing and integration with retrieval, tools and constrained decoding.
>
> We considered changing the model itself, but most of the errors we saw were not clearly caused by the model weights. They came from missing context, noisy context, numerical precision, or invalid final formatting. For that reason, improving the pipeline around the model was more direct and easier to test.

### 0:55 - 1:35 | Architettura generale

**Cosa mostrare:** diagramma pipeline, poi sezioni 7-9 su retrieval, RRF, reranker e answer parsing.

**Da dire:**

> The core architecture has three main branches.
>
> First, Maths questions use deterministic and validated tools, a Python and SymPy executor, and only then a local Qwen fallback with constrained final choice parsing.
>
> Second, News questions use fresh raw evidence from Google News RSS and Tavily. These services are not used as LLMs: they only return raw documents or article text, which is allowed by the assignment.
>
> Third, general knowledge questions use local RAG over SimpleWiki, KELM and mathematical textbooks. We combine sparse BM25 retrieval, dense HNSW retrieval, reciprocal rank fusion and Qwen3 reranking.
>
> We added retrieval because the local model alone was not robust enough on knowledge questions. Many wrong answers were not pure reasoning errors: the model remembered a similar fact, confused two entities, or answered from incomplete context. Retrieval makes the model look at explicit evidence before choosing.
>
> BM25 helped when the wording matched the question, while dense retrieval helped when the same fact was expressed differently. Both could still return noisy candidates, so reranking became necessary to decide which passages were actually useful for the final answer.
>
> The final answer is always converted to a multiple-choice option id.

### 1:35 - 2:15 | Modelli usati

**Cosa mostrare:** celle di caricamento Qwen GGUF, embedding/reranker, poi solo indicare che Whisper appare nella sezione speech finale.

**Da dire:**

> The model stack also evolved during the project. The first notebooks did not use a generative LLM: they used TF-IDF, BM25 and then a MiniLM cross-encoder reranker.
>
> The first generative LLM we added was a small Qwen2.5 Instruct model, mainly as a tool router and compact RAG fallback. Later, the main reasoning backend moved to a local Qwen3.5 9B GGUF model, first with Q6 quantization and then with Q8 in the final V8 and Notebook 13 versions.
>
> The final answer model is therefore Qwen3.5 9B in Q8 GGUF format, executed locally with llama-cpp-python. We use Q8 because it was more reliable than lighter variants while still fitting in the available GPU setup.
>
> Dense retrieval uses a MiniLM sentence-transformer with HNSW indexes, while sparse retrieval uses BM25 or BM25S. For reranking, the final notebooks use Qwen3-Reranker 0.6B, replacing the earlier MiniLM cross-encoder because it gave cleaner evidence to the LLM.
>
> For speech mode, we use openai/whisper-large-v3-turbo locally. The ASR output is then passed to exactly the same reasoning code as text mode.

### 2:15 - 3:00 | Evoluzione sperimentale

**Cosa mostrare:** sezione Maths tools, Python/SymPy executor, Wikipedia/Tavily fallback.

**Da dire:**

> We started with very simple baselines: first-option, TF-IDF and BM25 retrieval without an LLM. These were useful to validate the API and logging, but they were not enough for robust answering.
>
> We then added a MiniLM reranker, dense HNSW retrieval, a small Qwen2.5 Instruct model for routing and fallback, and eventually the larger local Qwen3.5 GGUF backend. This gave us a real RAG pipeline while keeping the answer generation local.
>
> The development was mostly driven by the mistakes we saw in the logs. When a wrong answer came from missing evidence, we worked on retrieval. When it came from arithmetic, we stopped asking the model to calculate and added tools. When it came from answer formatting, we constrained the output.
>
> The main remaining weaknesses were Maths, recent News and ambiguous entity questions. For Maths, the model could often describe the right reasoning, but small arithmetic or option-matching errors were enough to lose the game. So we progressively added validated tools, SymPy execution, deterministic option matching and a Micro-CoT fallback.
>
> For News and entity-sensitive categories, we added fresh retrieval and fallback evidence from RSS, Tavily and Wikipedia. The goal was to avoid common traps like confusing the source of a news item with the subject of the question, or retrieving a related article that does not actually answer the question.
>
> The final V8 notebook consolidates these components, and Notebook 13 adds the speech interface.

### 3:00 - 3:45 | Routing policy e codice

**Cosa mostrare:** sezione 11 `answer_strategy`, categoria Maths, poi sezioni 13-14 API loop e run per categoria.

**Da dire:**

> The central function is `answer_strategy`. It selects the strategy based on the competition category.
>
> If the category is Maths, it first tries validated deterministic tools. If no tool gives a reliable answer, it tries a Python executor, and then a local Qwen reasoning fallback.
>
> If the category is News, it retrieves recent articles and uses a headline-aware prompt.
>
> For Entertainment and History, the notebook can use local evidence plus Wikipedia and Tavily. For other categories, it mainly relies on the local RAG stack.
>
> This is the R&D conclusion of the project: we did not find one universal prompt that solved everything. Different categories failed for different reasons, so the final pipeline chooses the strategy based on the error pattern we observed during development.

### 3:45 - 4:25 | Risultati ed evaluation

**Cosa mostrare:** log summary, risultati README, celle finali con CSV o tabelle, poi inizio sezione V9 speech mode.

**Da dire:**

> The assignment evaluation considers both leaderboard performance and the quality of the investigation. We therefore kept the previous notebooks as ablations and logged strategy, latency, correctness, retrieved evidence and tool traces.
>
> Across the saved logs, we observed at least one run reaching 1,024,000 dollars for each category. The most important final improvement is on Maths: in `run_v8.csv`, the V8 Maths run reaches 1,024,000, with 98 logged questions, 79 correct answers, about 80.6 percent accuracy and no timeouts.
>
> Notebook 13 then tests the speech extension. We added speech only after the text pipeline was stable, because we did not want to debug speech and reasoning at the same time. The `SpeechGameAdapter` wraps the original game object. It fetches audio, transcribes it, builds a text-compatible question, and then calls the same competition runner.

### 4:25 - 4:55 | Limiti e trade-off

**Cosa mostrare:** V9 speech run cells e limitations markdown cell.

**Da dire:**

> The main limitations are retrieval noise, especially for fresh or ambiguous questions, and the difficulty of parsing mathematical language from speech.
>
> The larger Qwen Q8 model improves reliability, but it costs more memory and time. Qwen3-Reranker is also stronger than MiniLM, but it increases GPU load.
>
> Speech mode is more experimental. The main bottleneck is no longer the reasoning engine, but the extra ASR and audio-fetch latency, plus transcription errors on mathematical notation and short option texts.
>
> We avoided external LLM APIs entirely. External services are used only for raw evidence, which keeps the reasoning local and compliant with the assignment.

### 4:55 - 5:00 | Chiusura

**Cosa mostrare:** final cell / notebook title.

**Da dire:**

> In summary, our final system is a local, tool-augmented RAG agent with a modular speech adapter. It satisfies the assignment constraints and shows the full evolution from simple retrieval baselines to the final multimodal notebook.

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
