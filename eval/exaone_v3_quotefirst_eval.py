# -*- coding: utf-8 -*-
"""파인튜닝된 exaone3.5-lora:v3 위에, "결론 전에 근거 원문을 먼저 그대로 인용하라"는
지시 1줄을 런타임에 추가해 재평가 -- 파인튜닝 + 프롬프트 트릭을 합치면 v3 단독(Faithfulness
0.794)보다 더 나아지는지, 그리고 오늘 측정한 베이스+프롬프트(0.858)보다도 나아지는지 확인.

- rag/graph.py 파일은 전혀 수정하지 않음 (exaone_baseline_quotefirst_eval.py와 동일 패턴,
  Pipeline._answer_system_prompt를 프로세스 안에서만 monkey-patch).
- local_model="exaone3.5-lora:v3"만 다르고 나머지(30문항 seed=42, 검색 파이프라인, 판사)는
  베이스+프롬프트 실험·기존 v1/v2/v3/베이스라인 평가와 완전히 동일 -> 직접 비교 가능.
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

QUOTE_FIRST_ADDITION = (
    " 핵심 결론을 말하기 전에, 그 결론의 근거가 되는 근거 원문 문장을 [식별자] 인용과 "
    "함께 그대로 한 번 옮겨 적은 뒤 결론을 서술하라(원문을 바꿔 쓰지 말 것)."
)

_orig_answer_system_prompt = Pipeline._answer_system_prompt


def _patched_answer_system_prompt(self):
    return _orig_answer_system_prompt(self) + QUOTE_FIRST_ADDITION


Pipeline._answer_system_prompt = _patched_answer_system_prompt


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
OUT = Path("/private/tmp/claude-501/-Users-gyuyeong-projects/ec6664da-8b58-4864-8794-7da1b8965774/scratchpad/exaone_v3_quotefirst_results.jsonl")
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
    """베이스라인/v1/v2/v3/베이스+프롬프트와 완전히 동일한 샘플링(같은 seed)."""
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
    print(f"  {len(questions)}건 샘플링 완료 (기존 4개 평가와 동일 seed=42)", flush=True)

    print("[2/3] Index 로드...", flush=True)
    t0 = time.time()
    index = Index()
    print(f"  {time.time() - t0:.1f}s", flush=True)

    judge = Judge("OpenAI", openai_key)
    pipeline = Pipeline(index, local=True, local_model="exaone3.5-lora:v3")
    print("[실제 시스템 프롬프트]\n" + pipeline._answer_system_prompt() + "\n", flush=True)

    done_idx = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_idx.add(json.loads(line)["idx"])
    print(f"[3/3] {len(questions)}건 중 이미 완료 {len(done_idx)}건 — 이어서 진행 "
          f"(v3 LoRA + 프롬프트 추가)...", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fout:
        for i, item in enumerate(questions):
            if i in done_idx:
                continue
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
