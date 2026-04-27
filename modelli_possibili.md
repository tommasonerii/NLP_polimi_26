# Possibili modelli per PoliMillionaire

Questo file riassume strategie compatibili con gli argomenti del corso e con un vincolo pratico forte: al massimo abbiamo una singola GPU T4 gratuita su Colab o Kaggle. Quindi niente modelli grossi, niente ensemble pesanti e niente pipeline lente. Le probabilita sono stime iniziali da verificare nel notebook, non risultati sperimentali.

Assunzioni realistiche:

- runtime gratuito con al massimo una GPU T4;
- circa 15-16 GB di VRAM, ma non sempre tutti disponibili;
- modelli piccoli o quantizzati, idealmente caricati una sola volta a inizio notebook;
- timeout del gioco di circa 30 secondi per domanda;
- niente API di LLM per generare risposte, per vincolo della consegna;
- obiettivo: battere baseline random, rispondere entro tempo e produrre una buona analisi.

## Roadmap ordinata

L'ordine sotto e quello consigliato per lavorare: prima cose semplici, veloci e misurabili; poi miglioramenti progressivi; infine esperimenti piu complessi solo se resta tempo.

| Ordine | Soluzione | Complessita | Tempo risposta | Probabilita di funzionamento | Quando farla | Quando passare oltre |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | Baseline random | Molto bassa | Istantaneo | 20% | Subito | Appena il loop API e il logging funzionano |
| 2 | Baseline "prima opzione" | Molto bassa | Istantaneo | 20% | Subito | Dopo aver misurato qualche partita |
| 3 | Regole semplici / keyword matching | Bassa | Istantaneo | 25% | Dopo le baseline | Se copre pochi casi o resta fragile |
| 4 | TF-IDF + cosine similarity | Bassa | Molto veloce | 35% | Prima soluzione IR seria | Se BM25 e disponibile |
| 5 | BM25 su documenti locali | Bassa-media | Molto veloce | 40% | Primo retrieval consigliato | Quando abbiamo log e knowledge base minima |
| 6 | BM25 che sceglie direttamente l'opzione | Media | Molto veloce | 38% | Prima soluzione "intelligente" senza LLM | Se serve ragionamento o conoscenza implicita |
| 7 | Agentic router + tool matematici SymPy | Media | Molto veloce | 45-55% su Maths | Prima dei LLM, per domande calcolabili | Quando i tool non coprono piu casi |
| 8 | SymPy/statistics/algebra tools | Media | Molto veloce | 45-60% su Maths | Aggiungerli in modo modulare | Dopo aver rilevato pattern numerici |
| 9 | LLM piccolo 1B-2B zero-shot | Media | Veloce | 35-42% | Primo test generativo su T4 | Se output e parsabile e sta nei tempi |
| 10 | LLM piccolo 1B-2B few-shot | Media | Veloce | 38-45% | Stabilizzare formato e scelta opzione | Se il prompt lungo non rallenta troppo |
| 11 | BM25 + LLM 1B-2B | Media | Veloce | 45-50% | Primo RAG completo e leggero | Se migliora rispetto a LLM puro |
| 12 | LLM 3B-4B quantizzato | Media-alta | Medio | 42-50% | Modello principale se la T4 regge | Se latenza e VRAM sono accettabili |
| 13 | BM25 + LLM 3B-4B quantizzato | Media-alta | Medio | 50-55% | Soluzione target realistica | Quando il notebook e gia solido |
| 14 | Sentence embeddings piccoli + LLM piccolo | Media-alta | Medio | 48-52% | Da provare contro BM25 | Solo se BM25 perde su parafrasi |
| 15 | Cross-encoder mini per reranking | Alta | Medio-lento | 35-45% | Esperimento offline | Se aggiunge troppo poco, scartarlo |
| 16 | LLM 7B-8B in 4-bit | Alta | Medio-lento | 35-45% | Esperimento di qualita | Solo se non va in OOM e non sfora 30s |
| 17 | Ensemble di prompt | Alta | Lento | 35% | Solo offline o su poche domande | Se raddoppia la latenza, non usarlo in partita |
| 18 | Fine-tuning / LoRA | Molto alta | Variabile | 30% | Solo se avete dati e tempo | Non farlo prima di RAG + logging |
| 19 | Audio ASR/TTS o browser automation | Molto alta | Lento/fragile | 25% | Solo come extra creativo | Non deve bloccare la soluzione principale |

## Priorita effettiva

La sequenza minima che conviene davvero implementare e:

1. API + logging.
2. Random baseline.
3. Prima opzione baseline.
4. Agentic router con tool matematici/statistici semplici.
5. BM25 su documenti locali o Wikipedia/cache.
6. BM25 diretto sulle opzioni.
7. LLM piccolo 1B-2B zero-shot.
8. LLM piccolo 1B-2B few-shot.
9. BM25 + LLM piccolo.
10. Se la T4 regge: LLM 3B-4B quantizzato.
11. Se migliora davvero: BM25 + LLM 3B-4B.

Tutto quello dopo e sperimentale. Va bene provarlo, ma non deve diventare dipendenza critica del progetto.

## Modelli realistici con una T4

Le famiglie sotto sono intese come classi di modelli, non come obbligo di usare proprio quel nome. Con una T4 conviene misurare subito: tempo di caricamento, VRAM occupata, tempo medio di generazione con prompt reale e numero di timeout.

| Famiglia | Uso consigliato | Probabilita stimata | Note pratiche |
| --- | --- | ---: | --- |
| LLM istruito 1B-2B | Baseline LLM veloce | 35-42% | Consigliato per primo test end-to-end; dovrebbe stare comodo su T4 |
| LLM istruito 3B-4B quantizzato | Modello principale leggero | 42-50% | Miglior compromesso per una T4; usare output breve e max token basso |
| LLM istruito 7B-8B in 4-bit | Esperimento di qualita | 35-45% | Possibile ma non da dare per scontato: testare OOM e latenza prima di usarlo nel gioco |
| Modello "thinking" piccolo | Domande logiche | 38-48% | Limitare molto il ragionamento, altrimenti rischia timeout |
| Sentence-transformer mini | Retrieval semantico | 40-52% come componente | Utile per RAG, molto piu leggero di usare LLM per cercare |
| BM25 puro | Retrieval lessicale | 35-45% come componente | Ottima baseline di retrieval, quasi gratis computazionalmente |
| Cross-encoder mini | Reranking | 35-45% come componente | Valutarlo offline: in partita puo essere troppo lento |

## Cosa eviterei

- Modelli 13B+ o non quantizzati: troppo rischio di OOM e timeout.
- Ensemble di 2+ LLM durante la partita: poco pratico su una sola T4.
- Usare un 7B come unica soluzione senza fallback: se il runtime e lento o va in OOM si blocca tutto.
- Prompt lunghi con troppi documenti RAG: aumentano latenza e confusione.
- Fine-tuning come prima cosa: prima serve un loop stabile e dati di errore.
- Browser automation: piu fragile della API testuale fornita.

## Configurazione consigliata per una T4

Configurazione primaria:

- router: decide tra tool, retrieval e LLM;
- tool: SymPy, pattern statistici, controlli sulle opzioni, algebra semplice;
- retrieval: BM25 su CPU;
- modello generativo: prima LLM 1B-2B, poi eventualmente LLM 3B-4B quantizzato;
- prompt: massimo 3 snippet, nessuna spiegazione lunga;
- output: solo id opzione;
- `max_new_tokens`: molto basso, idealmente 4-8 token;
- fallback: se parsing fallisce, scegliere con scoring lessicale o prima opzione;
- logging: sempre attivo.

Configurazione alternativa piu leggera:

- retrieval: BM25 o TF-IDF;
- nessun modello generativo, oppure LLM 1B-2B;
- utile come fallback se il 3B-4B e troppo lento nel runtime disponibile.

Configurazione sperimentale:

- LLM 7B-8B in 4-bit;
- provarlo solo offline o su poche domande;
- tenerlo solo se risponde stabilmente entro pochi secondi e non causa OOM.

## Architettura consigliata

La soluzione piu sensata e una pipeline leggera:

1. Ricevi domanda e opzioni dalla API.
2. Applica normalizzazione minima del testo.
3. Router agentico:
   - se e matematica/statistica/algebra, prova un tool deterministico;
   - se e conoscenza generale, prova retrieval locale/live;
   - se nessun tool basta, usa LLM piccolo.
4. Recupera 3 snippet al massimo con BM25 o embeddings piccoli.
5. Passa a un LLM piccolo un prompt molto corto.
6. Forza output solo come `option_id`.
7. Se l'output non e valido, fallback alla risposta con scoring lessicale o prima opzione.
8. Logga domanda, risposta, tool usato, latenza, modello, strategia e correttezza.

## Prompt consigliato

Usare prompt corti. Esempio:

```text
You must answer a multiple-choice quiz.
Return only the option id.

Question: {question}
Options:
{id1}. {text1}
{id2}. {text2}
{id3}. {text3}
{id4}. {text4}

Context:
{retrieved_context}

Answer:
```

Evitare prompt che chiedono spiegazioni lunghe durante la partita. Le spiegazioni si possono generare dopo, offline, per analisi.

## Esperimenti minimi da fare in ordine

1. Random baseline.
2. Prima opzione baseline.
3. Regole/keyword semplici.
4. TF-IDF o BM25 senza LLM.
5. BM25 che sceglie l'opzione direttamente.
6. Calculator tool sulle domande numeriche.
7. LLM piccolo zero-shot.
8. LLM piccolo few-shot.
9. BM25 + LLM piccolo.
10. LLM 3B-4B quantizzato, se la T4 regge.
11. BM25 + LLM 3B-4B, se migliora davvero.
12. Embeddings piccoli + LLM piccolo, se BM25 non basta.

## Metriche da misurare

- Accuratezza totale.
- Livello medio raggiunto.
- Premio medio ottenuto.
- Tempo medio per domanda.
- Numero di timeout.
- Numero di errori di parsing dell'output LLM.
- Accuratezza per categoria di domanda.
- Confronto tra baseline, LLM puro, RAG e RAG + tool.
- Error analysis con esempi concreti.

## Raccomandazione pratica

Partirei cosi:

1. API + logging.
2. Baseline random e prima opzione.
3. Regole semplici e BM25 locale.
4. BM25 diretto sulle opzioni.
5. LLM 1B-2B con prompt corto.
6. BM25 + LLM 1B-2B.
7. Solo dopo provare un 3B-4B quantizzato.
8. Solo alla fine provare un 7B quantizzato e confrontare se il miglioramento vale latenza, VRAM e rischio timeout.

La tesi del progetto puo essere: con una sola T4, una pipeline RAG leggera e ben valutata puo battere un LLM piccolo usato da solo, anche se non raggiunge prestazioni da modelli grandi.
