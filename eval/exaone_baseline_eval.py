# -*- coding: utf-8 -*-
"""EXAONE 로컬 경로 'before' 베이스라인: 골든셋에서 30건 샘플링 → 실제 답변 생성 →
OpenAI 판사(gpt-4o-mini)로 Faithfulness/Answer Relevancy 채점. 일회성 진단 스크립트.

EXAONE는 로컬(비-OpenAI) 모델이라 OpenAI 판사와 벤더가 달라 자기편향 문제 없음
(README '자기편향 주의' 섹션과 동일 판단 — 독립 평가로 신뢰 가능).

결과는 매 건 즉시 JSONL에 append(중단돼도 부분 결과 보존).
"""
import json
import random
import time
from pathlib import Path

from rag import common as C
from rag.eval.judge import Judge
from rag.graph import build_graph
from rag.search import Index

ROOT = C.ROOT
OUT = Path("/private/tmp/claude-501/-Users-gyuyeong-projects/ec6664da-8b58-4864-8794-7da1b8965774/scratchpad/exaone_baseline_results.jsonl")
N_SAMPLE = 30


def _env_openai():
    p = ROOT / ".env"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "OPENAI_API_KEY":
            return v.strip().strip('"').strip("'")
    return None


def sample_questions(n=N_SAMPLE, seed=42):
    """게시판별 비례 샘플링(전체 분포: 016005 489·016003 437·016002 142·016001 112·016006 15)."""
    by_board = {}
    with (ROOT / "eval" / "goldenset.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_board.setdefault(d.get("board", "?"), []).append(d)
    total = sum(len(v) for v in by_board.values())
    rng = random.Random(seed)
    picked = []
    for board, items in sorted(by_board.items()):
        k = max(1, round(n * len(items) / total))
        picked.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(picked)
    return picked[:n]


def main():
    t_start = time.time()
    openai_key = _env_openai()
    assert openai_key, "OPENAI_API_KEY 필요(.env) — 판사용"

    print("[1/3] 골든셋에서 질문 샘플링...", flush=True)
    questions = sample_questions()
    print(f"  {len(questions)}건 샘플링 완료 (게시판 분포: "
          f"{sorted(set(q.get('board') for q in questions))})", flush=True)

    print("[2/3] Index 로드 (BGE-M3 임베더 + 리랭커)...", flush=True)
    t0 = time.time()
    index = Index()
    print(f"  {time.time() - t0:.1f}s", flush=True)

    judge = Judge("OpenAI", openai_key)  # EXAONE는 로컬 모델이라 OpenAI 판사와 자기편향 없음
    print(f"[3/3] {len(questions)}건 EXAONE 답변 생성 + 채점 시작...", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(questions):
            q = item["question"]
            t0 = time.time()
            g = build_graph(index, local=True)
            st = g.invoke({"question": q},
                          {"configurable": {"thread_id": f"baseline_{i}"}})
            ans_obj = st.get("answer", {})
            answer = ans_obj.get("answer", "")
            used_refs = ans_obj.get("used_refs", [])
            has_grounds = ans_obj.get("has_grounds", False)
            retrieved = st.get("retrieved", [])
            grounding = "\n\n".join(h.get("text", "") for h in retrieved)
            record = {
                "idx": i, "id": item.get("id"), "board": item.get("board"),
                "question": q, "expected_ref_keys": item.get("expected_ref_keys", []),
                "retrieved_refs": [(h["ref_key"] or h["doc_no"]) for h in retrieved],
                "answer": answer, "used_refs": used_refs, "has_grounds": has_grounds,
                "gen_latency_s": round(time.time() - t0, 1),
            }
            if has_grounds:
                ev = judge.evaluate(q, answer, grounding)
                if ev:
                    record.update({
                        "faithfulness": ev["faithfulness"],
                        "unsupported": ev["unsupported"],
                        "answer_relevancy": ev["answer_relevancy"],
                        "relevancy_reason": ev["relevancy_reason"],
                    })
                else:
                    record["judge_failed"] = True
            else:
                record["refusal"] = True
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.time() - t_start
            f_str = record.get("faithfulness", "-")
            r_str = record.get("answer_relevancy", "-")
            print(f"  [{i+1}/{len(questions)}] {record['gen_latency_s']}s · "
                  f"F={f_str} R={r_str} refusal={not has_grounds} · "
                  f"누적 {elapsed:.0f}s", flush=True)

    print(f"\n총 소요 {time.time() - t_start:.0f}s", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
