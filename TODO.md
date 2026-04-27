# PoliMillionaire - Project TODO List

Questi task sono pensati per rispettare i vincoli del progetto (modelli locali, NO API a pagamento) e sfruttare al meglio la singola GPU T4 (16GB di VRAM) a disposizione su Google Colab.

- [ ] **1. Indicizzare KELM con BM25**
  - [ ] Creare un indice separato per la knowledge base KELM.
  - [ ] Verificare se l'aggiunta di KELM migliora i risultati del Retrieval rispetto alla baseline di SimpleWiki, specialmente per domande fattuali.

- [ ] **2. Impostare LLM quantizzato (4-bit) su Colab**
  - [ ] Configurare `transformers` con `bitsandbytes` (oppure usare `vLLM` / `Ollama`).
  - [ ] Caricare un modello da ~8 miliardi di parametri (es. Llama-3-8B-Instruct o Mistral-7B). La quantizzazione a 4-bit occuperà solo ~5-6 GB di VRAM, lasciando ampio spazio per il contesto.

- [ ] **3. Aggiungere Reranking con BERT (Cross-Encoder)**
  - [ ] Utilizzare un modello BERT pre-addestrato per sentence similarity (es. `cross-encoder/ms-marco-TinyBERT-L-2-v2`).
  - [ ] Passare i top-K documenti (es. top 20) restituiti da BM25 a BERT per un riordinamento basato sulla semantica.
  - [ ] Selezionare solo i veri top-3 o top-5 da passare come contesto all'LLM.

- [ ] **4. Creare pipeline RAG (BM25 + BERT + LLM)**
  - [ ] Implementare una logica che catturi la query del quiz.
  - [ ] Ricercare (top-k ampi) sia su SimpleWiki che su KELM con BM25.
  - [ ] Rerankare i risultati con BERT per scegliere i migliori.
  - [ ] Iniettare i risultati estratti nel prompt (few-shot o system prompt) fornito al LLM per generare/scegliere la corretta opzione.

- [ ] **5. Integrare Tool Calling (Calcolatrice/SymPy)**
  - [ ] Adattare le funzioni in `agentic_tools.py` in in modo che possano essere invocate dall'LLM locale.
  - [ ] Utilizzare l'esecuzione esatta tramite codice (Calcolatrice) per rispondere alle domande della categoria "Maths", aiutando l'LLM a non allucinare sui calcoli.

- [ ] **5. Test e confronto modelli (Standard vs Thinking)**
  - [ ] Valutare un modello convenzionale (es. Llama-3-8B-Instruct).
  - [ ] Paragonarlo ad un modello "Reasoning" leggero (es. DeepSeek-R1-Distill-Llama-8B o Qwen-Math).
  - [ ] Misurare la latenza (< 30 secondi richiesti) e calcolare l'accuratezza finale, rispondendo agli *Hints* dei professori nel report/video finale.