# Video Script - PoliMillionaire Pipeline Development

Target duration: **4:30-5:00 minutes**.

Required format: **notebook screen capture, no slides, maximum 5 minutes**.

Notebook to show: `project/notebooks/13_speech.ipynb`.

## Scroll Map

| Time | Notebook section | Main point |
| --- | --- | --- |
| 0:00-0:45 | Retrieval setup | From lexical indexes to hybrid retrieval. |
| 0:45-1:25 | Model loading | Evolution toward local Qwen3.5 GGUF. |
| 1:25-2:35 | Prompting and `answer_strategy` | Modular prompts and routing by category. |
| 2:35-3:35 | Maths, News, external evidence | Fixes driven by error patterns in logs. |
| 3:35-4:20 | Speech mode | Whisper adapter over the V8 text pipeline. |
| 4:20-5:00 | Logs and limits | Results, trade-offs, remaining weaknesses. |

## Script

We started with simple baselines, because we first needed to understand the API, timing constraints, and logging format. The first notebook checked the client and played by always selecting the first option. After that, we moved to retrieval-only systems: first TF-IDF, then BM25, more precisely Okapi BM25, as a stronger sparse retrieval baseline.

Those early versions showed the first limitation. Sparse retrieval is fast when the question shares words with the right document, but fragile with paraphrases, ambiguous entities, and semantically close options. So the next step was not a bigger model immediately, but better evidence.

We added multiple sources: SimpleWiki for encyclopedic knowledge, KELM for short factual assertions, and later textbook indexes for maths context. Since scores from different indexes are not directly comparable, we used Reciprocal Rank Fusion to merge rankings. Then we added dense retrieval with MiniLM embeddings and HNSW indexes, so the system could retrieve semantically similar passages.

After improving recall, the next issue was precision. BM25 and dense search retrieved useful candidates, but also noisy passages. This is why we introduced reranking: first a MiniLM/BERT cross-encoder, later Qwen3-Reranker. In the final pipeline, reranking is used for both local evidence and raw external sources.

The generative model also evolved. At first there was no LLM. Then we tested smaller Qwen instruct models for compact reasoning and tool routing, but they were unstable with hard questions, JSON output, and final answer formatting. The final notebook uses a local Qwen3.5 9B GGUF model through `llama-cpp-python`, so answer generation stays local.

The main lesson from the logs was that a single universal prompt was not enough. Different categories failed in different ways. Some questions failed because the evidence was missing, some because it was noisy, some because the model made arithmetic mistakes, and some because the final output was not a valid option. For this reason, the final system is organized around `answer_strategy`, which routes each question to a category-specific path.

For general knowledge, we use local RAG: sparse retrieval, dense retrieval, fusion, reranking, and then a constrained local model call. For Entertainment and History, we also use Wikipedia and Tavily as raw evidence sources. For News, static corpora are not enough, so we use Google News RSS and Tavily to retrieve recent raw text. These services do not generate answers; they only provide evidence.

Maths was the category where the pipeline changed the most, because not all maths questions had the same nature. Some were computational: solve an equation, evaluate a probability, simplify an expression, or compare numeric options. For these, retrieval and prompting were the wrong abstraction. The model often recognized the right method, but then made a small arithmetic mistake or mapped the computed value to the wrong option. This is where we introduced agentic tools: the model can decide that calculation is needed and request a structured tool call, while Python validates the request, runs the computation with controlled functions such as SymPy or numeric solvers, and accepts the result only if it matches one of the four options. Other maths questions were instead knowledge questions, for example about definitions, theorems, or terminology. Those are treated more like general knowledge: we retrieve evidence from the textbook indexes and let the local model reason over that context, without forcing a calculator-style tool when no explicit computation is needed.

Prompting was refined through trial and error. We separated prompts for knowledge, News, and Maths, added different behavior when evidence is weak, and constrained the final answer to a valid option id. The CSV logs store strategy, latency, raw output, retrieved context, tool traces, and correctness, so each improvement was linked to observed failure patterns.

Notebook 13 adds speech mode on top of the stable V8 text pipeline. Whisper large-v3-turbo transcribes the audio question and options, rebuilds a text-compatible question object, and then calls the same `answer_strategy`. This makes speech errors easier to isolate from reasoning errors.

Across the saved logs, we reached at least one 1,024,000-dollar run for every category. The strongest improvement was in Maths, where V8 reached 1,024,000 with about 80.6% accuracy and no timeouts in the main log. The remaining limits are retrieval latency, noisy News evidence, and speech transcription errors, especially for formulas and short options.

## Short Backup Lines

- Each module corresponds to an error pattern we observed in the logs.
- External services retrieve raw text only; they do not generate answers.
- Speech changes the input format, not the decision engine.
- BM25 and dense retrieval solve different recall problems.
- Maths became more reliable when tools handled the computation.
