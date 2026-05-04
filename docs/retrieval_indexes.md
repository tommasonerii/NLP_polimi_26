# Indici retrieval locali

Questi comandi trasformano i dataset locali in indici interrogabili per RAG o
scoring diretto delle opzioni.

## SimpleWiki

Chunking degli articoli:

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

Indice TF-IDF:

```bash
conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_160w.jsonl \
  --kind tfidf \
  --output data/indexes/simplewiki_160w_tfidf.joblib \
  --max-features 300000 \
  --min-df 2 \
  --ngram-max 2
```

Indice BM25:

```bash
conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_160w.jsonl \
  --kind bm25 \
  --output data/indexes/simplewiki_160w_bm25.joblib
```

## Esperimenti solo base dati

Gli indici sotto cambiano solo corpus/tokenizzazione/campi indicizzati. Non
richiedono LLM diversi.

### BM25 con title boost

Il titolo viene ripetuto prima del testo del chunk solo nel testo indicizzato,
mentre lo snippet salvato resta il chunk originale. E utile per domande factual
in cui l'entita principale e nel titolo della pagina.

```bash
conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_160w.jsonl \
  --kind bm25 \
  --title-repeat 2 \
  --output data/indexes/simplewiki_160w_title2_bm25.joblib
```

### BM25 title boost + stopword removal

Riduce rumore e memoria dell'indice BM25. Da confrontare con la versione senza
stopword removal: su alcune query brevi le stopword possono ancora portare poco
segnale sintattico.

```bash
conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_160w.jsonl \
  --kind bm25 \
  --title-repeat 2 \
  --bm25-remove-stopwords \
  --output data/indexes/simplewiki_160w_title2_stop_bm25.joblib
```

### BM25 title boost + bigrammi

Indicizza anche bigrammi tokenizzati con `_`, ad esempio `johnny_cash` o
`united_states`. Aumenta dimensione indice, quindi va provato prima su KELM o su
un subset.

```bash
conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_160w.jsonl \
  --kind bm25 \
  --title-repeat 2 \
  --bm25-remove-stopwords \
  --bm25-ngram-max 2 \
  --output data/indexes/simplewiki_160w_title2_stop_bigram_bm25.joblib
```

### Chunk size alternative

Per domande quiz factual conviene confrontare chunk piu piccoli degli attuali
160 token.

```bash
conda run -n polimillionaire python project/src/make_retrieval_corpus.py \
  data/wiki/simplewiki_articles.jsonl \
  --source simplewiki \
  --id-prefix swiki80 \
  --max-words 80 \
  --overlap-words 20 \
  --min-words 20 \
  --output data/chunks/simplewiki_80w.jsonl

conda run -n polimillionaire python project/src/build_retrieval_index.py \
  data/chunks/simplewiki_80w.jsonl \
  --kind bm25 \
  --title-repeat 2 \
  --bm25-remove-stopwords \
  --output data/indexes/simplewiki_80w_title2_stop_bm25.joblib
```

## Test rapido

```bash
conda run -n polimillionaire python project/src/query_retrieval_index.py \
  data/indexes/simplewiki_160w_tfidf.joblib \
  --logs logs/first_option_comp_0.csv \
  --limit 5 \
  --top-k 3
```

Oppure query manuale:

```bash
conda run -n polimillionaire python project/src/query_retrieval_index.py \
  data/indexes/simplewiki_160w_tfidf.joblib \
  --query "What term describes Buster Keaton's signature facial expression? Grin Laugh Deadpan Smirk" \
  --top-k 3
```

## Nota pratica

Caricare l'indice da disco e fare query sono due cose diverse. Nel gioco
l'indice va caricato una sola volta prima di iniziare la partita; durante le
domande si chiama solo la funzione di retrieval.

TF-IDF e piu pratico come primo indice runtime. BM25 recupera spesso snippet
molto buoni, ma l'indice SimpleWiki completo puo richiedere piu tempo per essere
caricato da zero.
