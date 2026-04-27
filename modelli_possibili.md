# Possibili modelli per PoliMillionaire

Questo file riassume alcune strategie compatibili con gli argomenti visti nel corso di NLP e con i vincoli del progetto. Le probabilita sono stime iniziali, non risultati misurati: servono per decidere da dove partire. Nel notebook vanno poi sostituite o affiancate con metriche reali.

Assunzione usata per la stima: "funzionare" significa rispondere entro 30 secondi, superare una baseline random e produrre risultati spiegabili nel report.

| Approccio | Argomenti del corso collegati | Probabilita di funzionamento | Pro | Contro | Priorita |
| --- | --- | ---: | --- | --- | --- |
| Baseline random | Valutazione sperimentale | 20% | Facilissima, utile come confronto minimo | Prestazioni basse, nessuna intelligenza | Obbligatoria come baseline |
| Keyword matching / regole | Text preprocessing, classificazione semplice | 25% | Veloce, controllabile, utile per domande ricorrenti | Fragile, poco generale | Bassa |
| TF-IDF + cosine similarity su knowledge base locale | Text search, vector space model | 40% | Molto veloce, semplice da spiegare, niente GPU | Dipende dalla qualita dei documenti indicizzati | Media |
| BM25 / search lessicale su documenti locali | Text search e information retrieval | 45% | Spesso meglio di TF-IDF su query brevi, veloce | Non capisce bene sinonimi/parafrasi | Media |
| Word embeddings statici | Word embeddings | 35% | Buon ponte tra matching lessicale e semantico | Modelli statici meno forti su domande complesse | Bassa-media |
| Sentence embeddings + nearest neighbors | Embeddings contestuali, semantic search | 55% | Buono per recuperare contesto semanticamente vicino | Da solo non sempre sceglie bene tra 4 opzioni | Alta come componente RAG |
| Classificatore supervisionato leggero | Text classification | 30% | Facile da valutare se si hanno dati etichettati | Mancano dati di training specifici del quiz | Bassa, salvo raccolta dati |
| Transformer encoder fine-tuned | Transformers, text classification | 45% | Puo imparare pattern se si crea dataset di domande | Serve dataset, rischio overfitting, training piu costoso | Media-bassa |
| LLM open-weight zero-shot | Generative models, LLMs | 60% | Implementazione rapida, copre molte categorie | Puo essere lento o allucinare, dipende dal modello locale | Alta |
| LLM open-weight few-shot | Prompting, LLMs | 65% | Spesso migliora formato e affidabilita | Prompt piu lungo, piu latenza | Alta |
| RAG lessicale: BM25 + LLM locale | Search, RAG, LLMs | 70% | Buon equilibrio tra conoscenza esterna e ragionamento | Retrieval sbagliato puo confondere il modello | Molto alta |
| RAG semantico: embeddings + LLM locale | Semantic search, RAG, transformers | 75% | Migliore su parafrasi e domande fattuali | Richiede indicizzazione embeddings e gestione contesto | Molto alta |
| LLM + tool calculator | Agentic AI, tool calling | 65% | Utile per matematica, date, conversioni, logica semplice | Serve rilevare quando usare il tool | Alta come modulo |
| Ensemble di prompt sullo stesso LLM | LLMs, valutazione | 68% | Riduce errori casuali con voto di maggioranza | Moltiplica il tempo di inferenza | Media-alta |
| Ensemble di piu modelli locali | LLMs, model comparison | 72% | Puo aumentare robustezza e offre analisi interessante | Pesante in RAM/GPU e tempo | Alta se l'hardware regge |
| Fine-tuning di LLM piccolo | Fine-tuning, LLMs | 55% | Interessante da discutere, puo migliorare formato/strategie | Dati difficili da ottenere, rischio alto rispetto al tempo | Media-bassa |
| Interfaccia audio con ASR/TTS | Sequence models, transformers, speech | 40% | Originale se l'audio viene rilasciato | Rischio timeout e problemi di trascrizione | Opzionale |
| Browser automation invece della API testuale | Deployment/interazione web | 35% | Mostra capacita di integrazione | Piu fragile e probabilmente non necessario | Bassa |

## Modelli open-weight candidati

La consegna vieta l'uso di API di LLM per generare le risposte. Quindi i modelli vanno eseguiti localmente, per esempio su Colab o macchina propria.

| Famiglia modello | Uso consigliato | Probabilita stimata | Note operative |
| --- | --- | ---: | --- |
| Piccolo LLM istruito, 1B-3B parametri | Baseline LLM veloce | 50-60% | Buono per test rapidi, meno forte su conoscenza specifica |
| LLM istruito 7B-8B quantizzato | Modello principale | 60-70% | Buon compromesso qualita/tempo se gira entro 30 secondi |
| LLM "thinking" piccolo/medio | Domande logiche o complesse | 60-72% | Da verificare: il ragionamento lungo puo causare timeout |
| Sentence-transformer compatto | Retrieval semantico | 55-70% come componente | Non risponde da solo, ma migliora il contesto dato al LLM |
| Cross-encoder per reranking | Reranking documenti/opzioni | 50-65% come componente | Migliora il retrieval ma aggiunge latenza |
| Modello ASR open-source | Trascrizione audio, se disponibile | 35-50% | Utile solo se viene rilasciata l'interfaccia audio |

## Architettura consigliata

La soluzione piu promettente e una pipeline RAG + LLM locale:

1. Ricevi domanda e opzioni dalla API.
2. Classifica rapidamente il tipo di domanda: fattuale, matematica/logica, definizione, data/persona/luogo, altro.
3. Se e matematica/logica, usa un tool calculator o codice Python.
4. Se e fattuale, recupera documenti con BM25 o embeddings.
5. Costruisci un prompt compatto con domanda, opzioni e massimo 3-5 snippet.
6. Chiedi al modello di restituire solo l'id dell'opzione, piu una confidenza opzionale.
7. Se la confidenza e bassa, prova un secondo prompt o un secondo modello, ma solo se resta tempo.
8. Salva tutto in un log per analisi finale.

## Metriche da misurare nel notebook

- Accuratezza totale.
- Livello medio raggiunto.
- Premio medio ottenuto.
- Tempo medio per domanda.
- Numero di timeout.
- Accuratezza per categoria di domanda.
- Confronto tra baseline, LLM puro, RAG, RAG + tool, ensemble.
- Error analysis: esempi di domande sbagliate e probabile causa dell'errore.

## Raccomandazione pratica

Ordine di implementazione suggerito:

1. Random baseline.
2. LLM locale zero-shot.
3. LLM locale few-shot con output vincolato.
4. RAG semantico o BM25 + LLM.
5. Calculator tool.
6. Ensemble solo se c'e tempo e la latenza resta sotto 30 secondi.

