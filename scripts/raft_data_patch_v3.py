# -*- coding: utf-8 -*-
"""raft_train_data_full.jsonl(v1/v2가 쓴 700건)을 v3용으로 "패치"한다 -- GPT 재호출 없음.

바꾸는 것: 근거자료 개수만 production(rag/graph.py retrieve() k=8)과 동일하게 8개로 보정.
  - grounded: 기존 golden(정답 근거)은 그대로 두고, distractor(가짜 근거)만 로컬 재검색으로
    8개 총량이 되도록 추가.
  - refusal: distractor를 5개 -> 8개로 늘림(golden은 원래도 안 보여줬음, 그대로 유지).
안 바꾸는 것: 답변(assistant) 텍스트는 100% 그대로 유지 -- 이미 golden 근거에 맞춰 검증된
  문장이라 근거 문서 "개수"만 늘리는 이번 패치로는 재작성할 필요가 없음(golden 인용은
  그대로 유효). 인용 형식(원문 직접인용 유도) 실험은 이번엔 하지 않음 -- 별도 변수로
  나중에 독립적으로 테스트하기 위해 보류(raft_data_gen_v3.py에 이미 구현해둠).

출력: raft_train_data_v3.jsonl (build_v3_notebook.py가 참조하는 파일명과 동일).
"""
import json
import time
from pathlib import Path

from rag import common as C
from rag.graph import Pipeline
from rag.search import Index

ROOT = C.ROOT
SCRATCH = Path("/private/tmp/claude-501/-Users-gyuyeong-projects/ec6664da-8b58-4864-8794-7da1b8965774/scratchpad")
SRC = SCRATCH / "raft_train_data_full.jsonl"
OUT = SCRATCH / "raft_train_data_v3.jsonl"
PRODUCTION_K = 8


def _fmt_ctx(evidence):
    return "\n\n".join(f"[{e['ref_key']}] ({e['collection']}) {e['text'][:700]}" for e in evidence)


def load_goldenset_by_id():
    by_id = {}
    with (ROOT / "eval" / "goldenset.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_id[d["id"]] = d
    return by_id


def main():
    t0 = time.time()
    records = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[1/3] 원본 로드: {len(records)}건", flush=True)

    goldenset = load_goldenset_by_id()
    print(f"[2/3] 골든셋 {len(goldenset)}건 로드 + Index 로드...", flush=True)
    index = Index()
    p = Pipeline(index=index, local=True)

    print(f"[3/3] 근거자료 개수 패치 시작 (목표: 전부 {PRODUCTION_K}개로 통일)...", flush=True)
    n_patched, n_skipped, n_grounded_bumped, n_refusal_bumped = 0, 0, 0, 0
    out_records = []
    for i, rec in enumerate(records):
        item = goldenset.get(rec["source_id"])
        if item is None:
            n_skipped += 1
            out_records.append(rec)
            continue

        colls = [c for c in item.get("expected_collections", []) if c in index.colls] or list(index.colls)
        hits = index.retrieve_routed(item["question"], colls, k=8, min_standards=1, per_coll=12)

        if rec["category"] == "grounded":
            golden = []
            for ref in item.get("expected_ref_keys", []):
                r = p._lookup(ref)
                if r:
                    golden.append({"ref_key": ref, "collection": r["collection"], "text": r["text"]})
            if not golden:
                n_skipped += 1
                out_records.append(rec)
                continue
            golden_keys = {e["ref_key"] for e in golden}
            n_distractor = max(0, PRODUCTION_K - len(golden))
            distractors = [h for h in hits if (h["ref_key"] or h["doc_no"]) not in golden_keys][:n_distractor]
            all_ev = golden + [{"ref_key": h["ref_key"] or h["doc_no"], "collection": h["collection"],
                                 "text": h["text"]} for h in distractors]
            new_ctx = _fmt_ctx(all_ev)
            n_before = rec["messages"][1]["content"].count("\n\n[")  # 대략치, 로그용
            rec["messages"][1]["content"] = f"질문: {item['question']}\n\n근거:\n{new_ctx}"
            if len(all_ev) > n_before:
                n_grounded_bumped += 1

        elif rec["category"] == "refusal":
            golden_keys = set(item.get("expected_ref_keys", []))
            distractors_only = [h for h in hits
                                if (h["ref_key"] or h["doc_no"]) not in golden_keys][:PRODUCTION_K]
            if len(distractors_only) < 2:
                n_skipped += 1
                out_records.append(rec)
                continue
            new_ctx = _fmt_ctx([{"ref_key": h["ref_key"] or h["doc_no"], "collection": h["collection"],
                                  "text": h["text"]} for h in distractors_only])
            rec["messages"][1]["content"] = f"질문: {item['question']}\n\n근거:\n{new_ctx}"
            n_refusal_bumped += 1

        # messages[0](system)/messages[-1](assistant, 정답 텍스트)는 절대 건드리지 않음
        out_records.append(rec)
        n_patched += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(records)} 처리, 누적 {time.time()-t0:.0f}s", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n패치 완료: 총 {len(out_records)}건 (근거보정 {n_patched}건, "
          f"건너뜀 {n_skipped}건 -- 원본 그대로 유지)", flush=True)
    print(f"  grounded 근거보강: {n_grounded_bumped}건 / refusal 근거보강: {n_refusal_bumped}건", flush=True)
    print(f"저장: {OUT}", flush=True)
    print(f"총 소요 {time.time() - t0:.0f}s", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
