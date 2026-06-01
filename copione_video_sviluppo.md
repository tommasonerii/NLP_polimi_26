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

This video shows how our final pipeline evolved across experiments.

Thomas

We began with simple baselines to understand the API, timing constraints, and logging format. The first notebook checked the client by selecting the first option. We then moved to retrieval-only systems: TF-IDF, followed by BM25 as a stronger sparse baseline.

These versions exposed a limitation: sparse retrieval is fast when a question shares words with the right document, but fragile with paraphrases, ambiguous entities, and semantically close options. The next step was therefore not a larger model, but better evidence.

We added multiple sources: SimpleWiki for encyclopedic knowledge, KELM for short factual assertions, and textbook indexes for maths context. 

Because index scores are not directly comparable, we used Reciprocal Rank Fusion to merge rankings. Dense retrieval with MiniLM embeddings and HNSW indexes then enabled retrieval of semantically similar passages.

After improving recall, the next issue was precision. BM25 and dense search retrieved useful candidates, but also noisy passages. This is why we introduced reranking: first a MiniLM/BERT cross-encoder, later Qwen3-Reranker. In the final pipeline, reranking is used for both local evidence and raw external sources.

The generative model also evolved. At first there was no LLM. Then we tested smaller Qwen instruct models for compact reasoning and tool routing, but they were unstable with hard questions, JSON output, and final answer formatting. The final notebook uses a local Qwen3.5 9B quantized model through `llama-cpp-python`, so answer generation stays local.

Gio

The main lesson from the logs was that a single universal prompt was not enough. Different categories failed in different ways. Some questions failed because the evidence was missing, some because it was noisy, some because the model made arithmetic mistakes. For this reason, the final system is organized around `answer_strategy`, which routes each question to a category-specific path.

For general knowledge, we use local RAG: sparse  and dense retrieval, fusion, reranking and then a constrained local model call. For Entertainment and History, we also use Wikipedia and Tavily as raw evidence sources. For News, static corpora are not enough, so we use Google News RSS and Tavily to retrieve recent raw text.

Maths was the category where the pipeline changed the most, because not all maths questions had the same nature. Some were computational, like solving an equation or simplify an expression, for these, retrieval and prompting were the wrong abstraction. The model often recognized the right method, but then made a small arithmetic mistake or mapped the computed value to the wrong option. 

This is where we introduced agentic tools: the model can decide that calculation is needed and request a structured tool call, while Python validates the request and runs the computation with deterministic functions. Other maths questions were instead knowledge questions, for example about definitions or theorems. Those are treated more like general knowledge: we retrieve evidence from the textbooks and let the local model reason over that context, without forcing a tool when no explicit computation is needed.

Giuli

Prompting was refined through trial and error. We separated prompts for knowledge, News, and Maths, and added different behavior when evidence was weak. The CSV logs store strategy, latency, raw output, retrieved context, tool traces and correctness, so each improvement was linked to observed failure patterns.

Last notebook adds speech mode on top of the stable text pipeline. After a multimodel benchmark, we selected Whisper to transcribe the audio question and options. the pipeline then rebuilds a text-compatible question object, and then call the same `answer_strategy`. This makes speech errors easier to isolate from reasoning errors.

Across the saved logs, we reached at least 1,024,000-dollar run for each category. The most meaningful improvement was in Maths, because the first baselines were extremely weak there: simple retrieval could not solve calculations, and pure prompting was too unstable. Reaching 1,024,000 with about 80.6% accuracy required the largest pipeline change, combining routing, agentic tools, validation, and textbook-based knowledge retrieval. 

The remaining weaknesses are that some theoretical or edge-case maths questions still depend on selecting the right tool or textbook evidence, while in speech mode the main issue becomes transcription, especially for formulas, proper names and short options such as News answers.

## Short Backup Lines

- Each module corresponds to an error pattern we observed in the logs.
- External services retrieve raw text only; they do not generate answers.
- Speech changes the input format, not the decision engine.
- BM25 and dense retrieval solve different recall problems.
- Maths became more reliable when tools handled the computation.
