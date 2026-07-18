# -*- coding: utf-8 -*-
"""재학습된 exaone3.5-lora:v3(completion-only loss, max_length 자동확장, epoch 자동선택,
route/rewrite는 항상 기본모델 -- production 코드 fix 그대로 적용됨)를 held-out 30문항으로 평가.
v1(거부율 66.7%, Faithfulness 0.800)과 베이스라인(거부율 33%, Faithfulness 0.900)과 비교.
exaone_after_final_eval_v2.py와 동일 패턴/샘플링(seed=42)으로 공정 비교.
"""
import json
import random
import time
from pathlib import Path

import httpx
from openai import OpenAI as _OpenAI
_orig_openai_init = _OpenAI.__init__
def _patched_openai_init(self, *args, **kwargs):
    kwargs.setdefault("timeout", httpx.Timeout(1800.0, connect=10.0))
    return _orig_openai_init(self, *args, **kwargs)
_OpenAI.__init__ = _patched_openai_init

from rag import common as C
from rag.eval.judge import Judge
from rag.graph import Pipeline, _after_retrieve_edge
from rag.search import Index


def _merge(state, update):
    for k, v in update.items():
        if k == "trace" and "trace" in state:
            state["trace"] = state["trace"] + v
        else:
            state[k] = v
    return state


def run_pipeline_once(p, question):
    state = {"question": question, "history": [], "trace": []}
    _merge(state, p.route(state))
    _merge(state, p.retrieve(state))
    if _after_retrieve_edge(state) == "rewrite":
        _merge(state, p.rewrite(state))
        _merge(state, p.retrieve(state))
    _merge(state, p.answer(state))
    _merge(state, p.verify(state))
    return state


ROOT = C.ROOT
OUT = Path("/private/tmp/claude-501/-Users-gyuyeong-projects/ec6664da-8b58-4864-8794-7da1b8965774/scratchpad/exaone_v3_final_results.jsonl")
N_SAMPLE = 30
START_FROM = 0


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
    assert openai_key, "OPENAI_API_KEY 필요(.env) -- 판사용"

    print("[1/3] 골든셋에서 질문 샘플링...", flush=True)
    questions = sample_questions()
    print(f"  {len(questions)}건 샘플링 완료 (v1과 동일 seed=42, idx {START_FROM}부터)", flush=True)

    print("[2/3] Index 로드...", flush=True)
    t0 = time.time()
    index = Index()
    print(f"  {time.time() - t0:.1f}s", flush=True)

    judge = Judge("OpenAI", openai_key)
    pipeline = Pipeline(index, local=True, local_model="exaone3.5-lora:v3")
    remaining = questions[START_FROM:]
    print(f"[3/3] {len(remaining)}건 EXAONE v3(재학습) 답변 생성 + 채점...", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fout:
        for offset, item in enumerate(remaining):
            i = START_FROM + offset
            q = item["question"]
            t0 = time.time()
            st = run_pipeline_once(pipeline, q)
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
            f_str = record.get("faithfulness", "-")
            r_str = record.get("answer_relevancy", "-")
            print(f"  [idx={i}, {i+1}/{N_SAMPLE}] {record['gen_latency_s']}s · "
                  f"F={f_str} R={r_str} refusal={not has_grounds} · "
                  f"누적 {time.time()-t_start:.0f}s", flush=True)

    print(f"\n총 소요 {time.time() - t_start:.0f}s", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
