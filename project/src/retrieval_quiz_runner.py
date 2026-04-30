"""Run PoliMillionaire games with a local retrieval-only strategy.

The strategy does not call any LLM. For each answer option it retrieves evidence
with the query "question + option" and picks the option with the best retrieval
score. This is intentionally simple so TF-IDF and BM25 can be compared fairly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any

import joblib
import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def load_retrieval_index(index_path: str | Path) -> dict[str, Any]:
    return joblib.load(index_path)


def retrieve(index: dict[str, Any] | list[dict[str, Any]], query: str, top_k: int = 3) -> list[tuple[float, dict[str, Any]]]:
    if isinstance(index, list):
        # Handle multiple indexes using Reciprocal Rank Fusion (RRF)
        # to combine uncalibrated BM25 scores from different corpora
        rrf_k = 60
        all_results = {}
        for idx in index:
            results = retrieve(idx, query, top_k=top_k * 2) # Get slightly more for better fusion
            for rank, (score, doc) in enumerate(results):
                source = str(doc.get("source") or idx.get("corpus_path") or "")
                doc_id = doc.get("doc_id") or doc.get("id")
                chunk_id = doc.get("chunk_id")
                doc_id_key = (source, str(doc_id), str(chunk_id or ""))
                if doc_id_key not in all_results:
                    all_results[doc_id_key] = {"doc": doc, "rrf_score": 0.0}
                all_results[doc_id_key]["rrf_score"] += 1.0 / (rrf_k + rank + 1)
        
        # Sort by RRF score
        fused = sorted(all_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        return [(item["rrf_score"], item["doc"]) for item in fused[:top_k]]

    kind = index.get("kind")
    if not kind:
        raise ValueError("Provided index does not have a 'kind' key.")
        
    docs = index["docs"]

    if kind == "tfidf":
        # Rimuoviamo le stopwords prima di vettorizzare per accelerare la moltiplicazione sparsa
        clean_query = " ".join([t for t in tokenize(query) if t not in STOPWORDS]) or query
        query_vec = index["vectorizer"].transform([clean_query])
        scores = (index["matrix"] @ query_vec.T).toarray().ravel()
    elif kind == "bm25":
        # use bm25s get_scores
        # Rimuoviamo le stopwords: evita a BM25 di caricare le liste posizionali gigantesche di "the", "and", ecc.
        q_tokens = [t for t in tokenize(query) if t not in STOPWORDS]
        if not q_tokens:
            q_tokens = tokenize(query) # Fallback

        try:
            scores = np.asarray(index["bm25"].get_scores(q_tokens, show_progress=False), dtype=np.float32).ravel()
        except TypeError:
            scores = np.asarray(index["bm25"].get_scores(q_tokens), dtype=np.float32).ravel()
    else:
        raise ValueError(f"Unsupported retrieval index kind: {kind}")

    if len(scores) == 0:
        return []

    top_k = min(top_k, len(scores))
    top = np.argpartition(-scores, kth=top_k - 1)[:top_k]
    top = top[np.argsort(-scores[top])]
    return [(float(scores[i]), docs[int(i)]) for i in top if scores[i] > 0]


def compact_snippet(doc: dict[str, Any], max_chars: int = 350) -> dict[str, Any]:
    text = str(doc.get("text", ""))
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return {
        "id": doc.get("id"),
        "title": doc.get("title", ""),
        "source": doc.get("source", ""),
        "url": doc.get("url", ""),
        "text": text,
    }


def option_evidence_score(option_text: str, evidence_docs: list[tuple[float, dict[str, Any]]]) -> float:
    option_low = str(option_text).lower().strip()
    option_terms = [term for term in tokenize(option_text) if term not in STOPWORDS and len(term) > 1]
    if not option_low and not option_terms:
        return 0.0

    score = 0.0
    for rank, (_, doc) in enumerate(evidence_docs):
        rank_weight = 1.0 / (rank + 1)
        haystack = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
        if option_low and len(option_low) > 2 and option_low in haystack:
            score += 3.0 * rank_weight
        for term in option_terms:
            if term in haystack:
                score += rank_weight / max(1.0, len(option_terms) ** 0.5)
    return score


from agentic_tools import choose_with_agentic_tools

@dataclass
class RetrievalDecision:
    option_id: int
    option_text: str
    confidence: float
    option_scores: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    decision_source: str = ""
    tool_name: str = ""
    raw_tool_call: str = ""
    tool_result: str = ""
    llm_raw_answer: str = ""
    bert_top_option: str = ""


def choose_with_retrieval(question, index: dict[str, Any], option_top_k: int = 5, evidence_top_k: int = 5) -> RetrievalDecision:
    global_query = " ".join([str(question.text), *[str(opt.text) for opt in question.options]])
    global_results = retrieve(index, global_query, top_k=evidence_top_k)

    option_scores: list[dict[str, Any]] = []
    for opt in question.options:
        option_query = f"{question.text} {opt.text}"
        results = retrieve(index, option_query, top_k=option_top_k)
        scores = [score for score, _ in results]
        best_score = scores[0] if scores else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0
        retrieval_score = best_score + 0.20 * mean_score
        evidence_score = option_evidence_score(str(opt.text), global_results)
        option_scores.append(
            {
                "option_id": opt.id,
                "option_text": str(opt.text),
                "score": retrieval_score,
                "evidence_score": evidence_score,
                "best_score": best_score,
                "mean_top_score": mean_score,
                "top_evidence": compact_snippet(results[0][1]) if results else None,
            }
        )

    max_evidence_score = max(row["evidence_score"] for row in option_scores)
    if max_evidence_score > 0:
        max_retrieval_score = max(row["score"] for row in option_scores) or 1.0
        for row in option_scores:
            normalized_retrieval = row["score"] / max_retrieval_score
            row["score"] = row["evidence_score"] + 0.05 * normalized_retrieval

    option_scores.sort(key=lambda row: row["score"], reverse=True)
    best = option_scores[0]
    second_score = option_scores[1]["score"] if len(option_scores) > 1 else 0.0
    confidence = best["score"] - second_score

    return RetrievalDecision(
        option_id=int(best["option_id"]),
        option_text=str(best["option_text"]),
        confidence=float(confidence),
        option_scores=option_scores,
        evidence=[
            {
                "score": score,
                **compact_snippet(doc),
            }
            for score, doc in global_results
        ],
        decision_source="retrieval",
    )


LOG_FIELDS = [
    "timestamp",
    "strategy",
    "session_id",
    "competition_id",
    "competition_name",
    "attempt",
    "question_number",
    "question_id",
    "level",
    "question_text",
    "options_json",
    "chosen_option_id",
    "chosen_option_text",
    "confidence",
    "option_scores_json",
    "evidence_json",
    "decision_source",
    "tool_name",
    "raw_tool_call",
    "tool_result",
    "llm_raw_answer",
    "bert_top_option",
    "time_remaining_before",
    "decision_latency_seconds",
    "submit_latency_seconds",
    "total_latency_seconds",
    "earned_before",
    "correct",
    "timed_out",
    "game_over",
    "earned_after",
    "result_status",
    "current_level_after",
    "reached_level",
    "error_message",
]


def play_logged_game(client, competition_id: int, attempt: int, index: Any, strategy_name: str, choose_fn=None) -> list[dict[str, Any]]:
    game = client.game.start(competition_id=competition_id)
    competition_name = game.state.competition.name
    rows: list[dict[str, Any]] = []
    question_number = 0

    while game.in_progress:
        question = game.current_question
        if question is None:
            break
        try:
            setattr(question, "competition_id", competition_id)
            setattr(question, "competition_name", competition_name)
        except Exception:
            pass

        question_number += 1
        time_remaining_before = game.time_remaining
        earned_before = game.earned_amount
        options = [{"id": opt.id, "text": opt.text} for opt in question.options]
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "strategy": strategy_name,
            "session_id": game.session_id,
            "competition_id": competition_id,
            "competition_name": competition_name,
            "attempt": attempt,
            "question_number": question_number,
            "question_id": question.id,
            "level": question.level,
            "question_text": question.text,
            "options_json": json.dumps(options, ensure_ascii=False),
            "time_remaining_before": time_remaining_before,
            "earned_before": earned_before,
            "error_message": "",
        }

        start = time.perf_counter()
        try:
            if choose_fn is not None:
                decision = choose_fn(question, index)
            elif competition_name.lower() == "maths":
                # Try the agentic tools first
                agentic_decision = choose_with_agentic_tools(question, fallback=lambda q: None)
                if agentic_decision is not None:
                    # Fabricate a compatibility object for the logs
                    decision = RetrievalDecision(
                        option_id=agentic_decision.option_id,
                        option_text=next(opt.text for opt in question.options if opt.id == agentic_decision.option_id),
                        confidence=agentic_decision.confidence,
                        option_scores=[],
                        evidence=[{"text": agentic_decision.explanation}],
                        decision_source="tool_regex",
                        tool_name=agentic_decision.strategy,
                        tool_result=agentic_decision.explanation,
                    )
                else:
                    decision = choose_with_retrieval(question, index)
            else:
                decision = choose_with_retrieval(question, index)
            decision_latency = time.perf_counter() - start

            submit_start = time.perf_counter()
            result = game.answer(decision.option_id)
            submit_latency = time.perf_counter() - submit_start

            row.update(
                {
                    "chosen_option_id": decision.option_id,
                    "chosen_option_text": decision.option_text,
                    "confidence": decision.confidence,
                    "option_scores_json": json.dumps(decision.option_scores, ensure_ascii=False),
                    "evidence_json": json.dumps(decision.evidence, ensure_ascii=False),
                    "decision_source": getattr(decision, "decision_source", ""),
                    "tool_name": getattr(decision, "tool_name", ""),
                    "raw_tool_call": getattr(decision, "raw_tool_call", ""),
                    "tool_result": getattr(decision, "tool_result", ""),
                    "llm_raw_answer": getattr(decision, "llm_raw_answer", ""),
                    "bert_top_option": getattr(decision, "bert_top_option", ""),
                    "decision_latency_seconds": decision_latency,
                    "submit_latency_seconds": submit_latency,
                    "total_latency_seconds": decision_latency + submit_latency,
                    "correct": result.correct,
                    "timed_out": result.timed_out,
                    "game_over": result.game_over,
                    "earned_after": result.earned_amount,
                    "result_status": result.status,
                    "current_level_after": result.current_level,
                    "reached_level": result.reached_level,
                }
            )
            rows.append(row)
            print_attempt_row(row)

            if result.game_over or result.timed_out:
                break

        except Exception as exc:
            row.update(
                {
                    "chosen_option_id": "",
                    "chosen_option_text": "",
                    "confidence": "",
                    "option_scores_json": "",
                    "evidence_json": "",
                    "decision_source": "",
                    "tool_name": "",
                    "raw_tool_call": "",
                    "tool_result": "",
                    "llm_raw_answer": "",
                    "bert_top_option": "",
                    "decision_latency_seconds": time.perf_counter() - start,
                    "submit_latency_seconds": "",
                    "total_latency_seconds": time.perf_counter() - start,
                    "correct": "",
                    "timed_out": "",
                    "game_over": True,
                    "earned_after": game.earned_amount,
                    "result_status": "error",
                    "current_level_after": game.current_level,
                    "reached_level": "",
                    "error_message": repr(exc),
                }
            )
            rows.append(row)
            print_attempt_row(row)
            break

    return rows


def write_logs(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def print_attempt_row(row: dict[str, Any]) -> None:
    decision_source = row.get("decision_source") or ""
    tool_name = row.get("tool_name") or ""
    source_text = f" source={decision_source}" if decision_source else ""
    tool_text = f" tool={tool_name}" if tool_name else ""
    print(
        f"[{row['strategy']}] comp={row['competition_id']} {row['competition_name']} "
        f"attempt={row['attempt']} q={row['question_number']} level={row['level']} "
        f"chosen={row.get('chosen_option_id')} correct={row.get('correct')} "
        f"earned={row.get('earned_after')} latency={row.get('total_latency_seconds')}"
        f"{source_text}{tool_text}"
    )
    print(f"Q: {row['question_text']}")
    print(f"A: {row.get('chosen_option_text')}")
    if row.get("raw_tool_call"):
        print(f"Tool call: {row.get('raw_tool_call')}")
    if row.get("tool_result"):
        print(f"Tool result: {row.get('tool_result')}")
    evidence = row.get("evidence_json")
    if evidence:
        try:
            top = json.loads(evidence)[0]
            print(f"Top evidence: {top.get('title') or top.get('id')} :: {top.get('text')}")
        except Exception:
            pass
    if row.get("error_message"):
        print(f"ERROR: {row['error_message']}")
    print()


def run_all_competitions(client, index: Any, strategy_name: str, attempts_per_competition: int = 5, choose_fn=None) -> list[dict[str, Any]]:
    competitions = client.competitions.list_all()
    rows: list[dict[str, Any]] = []
    for comp in competitions:
        for attempt in range(1, attempts_per_competition + 1):
            rows.extend(
                play_logged_game(
                    client=client,
                    competition_id=comp.id,
                    attempt=attempt,
                    index=index,
                    strategy_name=strategy_name,
                    choose_fn=choose_fn,
                )
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows to summarize.")
        return

    by_comp: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["competition_id"]), str(row["competition_name"]))
        by_comp.setdefault(key, []).append(row)

    print("\nSummary")
    for (competition_id, name), comp_rows in sorted(by_comp.items()):
        correct = sum(1 for row in comp_rows if row.get("correct") is True)
        total = len(comp_rows)
        sessions = {(row["session_id"], row["attempt"]) for row in comp_rows}
        max_level = max(int(row.get("level") or 0) for row in comp_rows)
        avg_latency = np.mean([float(row.get("total_latency_seconds") or 0.0) for row in comp_rows])
        print(
            f"{competition_id} {name}: rows={total}, sessions={len(sessions)}, "
            f"correct={correct}, row_acc={correct / total:.3f}, "
            f"max_seen_level={max_level}, avg_latency={avg_latency:.3f}s"
        )
