# KELM limitato

Per il progetto conviene partire con un subset piccolo di KELM e usarlo solo per
retrieval locale, non per fine-tuning.

## Ambiente Conda

```bash
conda activate polimillionaire
conda install -c conda-forge datasets
```

## Subset consigliato iniziale

```bash
conda run -n polimillionaire python project/src/extract_kelm_subset.py \
  --limit 100000 \
  --output data/kelm/kelm_subset_100k.jsonl
```

Il comando usa streaming dal JSONL ufficiale Google, quindi evita di
materializzare tutto il dataset prima di scrivere le prime righe.

## Subset piu grande

Se il retrieval migliora rispetto a Wikipedia/SimpleWiki, aumentare a 500k:

```bash
conda run -n polimillionaire python project/src/extract_kelm_subset.py \
  --limit 500000 \
  --output data/kelm/kelm_subset_500k.jsonl
```

## Variante ancora piu leggera

Per usare il subset single-triple non ufficiale:

```bash
conda run -n polimillionaire python project/src/extract_kelm_subset.py \
  --url "" \
  --dataset visoc/KELM \
  --limit 100000 \
  --output data/kelm/kelm_single_triple_100k.jsonl
```

## Formato output

Ogni riga e un documento JSONL:

```json
{"id":"kelm_0","text":"...","sentence":"...","triple":"...","source":"google-research-datasets/kelm"}
```

Indicizzare principalmente `text`, tenendo `sentence`, `triple` e `source` per
debug e citazione nel report.
