# -*- coding: utf-8 -*-
"""LoRA 학습 데이터 생성(RAFT 스타일) v3 — raft_data_gen_full.py(v1/v2가 쓴 데이터) 대비 변경점:

1. [핵심 수정] train/inference 근거 개수 불일치 해소.
   production answer 노드는 항상 retrieve()가 반환한 근거 8개(k=8, rag/graph.py:275)를 그대로
   본다. 그런데 기존 데이터는 grounded 예시에 골든 1~n개 + distractor 최대 3개(총 4개 안팎),
   refusal 예시는 distractor 최대 5개만 넣었다 — 즉 모델이 "학습 때 본 근거 개수"와 "실전에서
   보는 근거 개수"가 달랐다(RAFT 원 논문 Zhang et al. 2024가 지적하는 정확히 그 문제,
   arXiv:2403.10131). v3는 두 카테고리 모두 근거를 8개로 채운다(가능한 만큼).
2. [경량 추가] 답변 포맷에 "근거 문장을 직접 인용한 뒤 결론"을 유도하는 지시 1줄 추가
   (RAFT의 CoT+원문축어인용 포맷이 HotpotQA +9.66%p/HuggingFace API +14.93%p — 근거:
   arXiv:2403.10131 Table 4). 단, rag/graph.py의 _answer_system_prompt() 주석에 이미 기록된
   교훈("프롬프트에 부정 지시를 추가하면 모델이 과잉 방어해 refusal이 급증했다", v1 25%→58%,
   v2 33%→43%/Faithfulness 0.900→0.794)을 반영해 **부정 지시가 아닌 긍정 지시 1개만** 추가하고,
   이 프롬프트는 데이터 생성 전용(good_sys)이지 실제 서비스 시스템 프롬프트(_answer_system_prompt)
   가 아니다 — 그건 절대 건드리지 않는다.
3. refusal 라벨 생성 로직(골든을 일부러 제외한 컨텍스트로 REFUSAL 학습)은 그대로 유지 —
   이미 R-Tuning류 "이 근거로는 실제로 답이 안 나옴" 원칙에 부합하는 설계였음.

출력: raft_train_data_v3.jsonl (v1/v2 데이터 파일은 건드리지 않음, 완전히 새 파일).
"""
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rag import common as C
from rag import llm as L
from rag.graph import Pipeline, _find_invalid_citations
from rag.search import Index

ROOT = C.ROOT
SCRATCH = Path("/private/tmp/claude-501/-Users-gyuyeong-projects/ec6664da-8b58-4864-8794-7da1b8965774/scratchpad")
OUT = SCRATCH / "raft_train_data_v3.jsonl"

MODEL = "gpt-5.4-mini"
N_POSITIVE_TARGET = 600
N_REFUSAL_TARGET = 100
POSITIVE_CANDIDATES = 780
MAX_RETRY = 1
MAX_WORKERS = 6
PRODUCTION_K = 8   # rag/graph.py retrieve()의 k=8과 반드시 일치시킬 것 (train/inference 정합성)

good_sys = (
    "너는 한국 회계기준 학습데이터 포맷터다. 아래 '공식 정답 요지'와 '근거'만 사용해 "
    "자연스러운 한국어 답변을 재구성한다. 절대 새로운 사실·결론을 추가하지 말고, "
    "근거에 없는 내용은 쓰지 마라. 인용은 근거의 대괄호 식별자를 그대로 [식별자] 형태로 "
    "문장에 넣어라. 근거 중 이 질문과 무관한 것은 인용하지 마라. "
    "질문 자체의 문서번호(질의회신 ID, 예: 2025-I-KQA006 형식)는 근거가 아니므로 "
    "절대 인용 대괄호에 넣지 마라 — 오직 위에서 준 '근거' 목록의 식별자만 인용하라. "
    "핵심 결론을 말하기 전에, 그 결론의 근거가 되는 원문 문장을 [식별자] 인용과 함께 "
    "그대로 한 번 옮겨 적은 뒤 결론을 서술하라(원문을 바꿔 쓰지 말 것)."
)


def _fmt_ctx(evidence):
    return "\n\n".join(f"[{e['ref_key']}] ({e['collection']}) {e['text'][:700]}" for e in evidence)


def load_excluded_ids():
    p = SCRATCH / "exaone_baseline_results.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def sample_pool(exclude_ids, seed=123):
    by_board = {}
    with (ROOT / "eval" / "goldenset.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("id") in exclude_ids or not d.get("expected_ref_keys"):
                continue
            by_board.setdefault(d.get("board", "?"), []).append(d)
    rng = random.Random(seed)
    pool = []
    for board in sorted(by_board):
        items = by_board[board][:]
        rng.shuffle(items)
        pool.extend(items)
    rng.shuffle(pool)
    return pool


write_lock = threading.Lock()
progress = {"pos_done": 0, "pos_rejected": 0, "ref_done": 0}
t_start = None


def process_positive(item, index, p, llm):
    golden = []
    for ref in item["expected_ref_keys"]:
        rec = p._lookup(ref)
        if rec:
            golden.append({"ref_key": ref, "collection": rec["collection"], "text": rec["text"]})
    if not golden:
        return None
    colls = [c for c in item["expected_collections"] if c in index.colls] or list(index.colls)
    hits = index.retrieve_routed(item["question"], colls, k=8, min_standards=1, per_coll=12)
    golden_keys = {e["ref_key"] for e in golden}
    # v3: distractor를 3개로 고정하지 않고 production과 동일한 8개 총량까지 채운다
    # (골든이 많으면 distractor는 그만큼 줄어듦 -- 항상 골든 우선 보존, 총량만 8 맞춤)
    n_distractor = max(0, PRODUCTION_K - len(golden))
    distractors = [h for h in hits if (h["ref_key"] or h["doc_no"]) not in golden_keys][:n_distractor]
    all_ev = golden + [{"ref_key": h["ref_key"] or h["doc_no"], "collection": h["collection"],
                         "text": h["text"]} for h in distractors]
    ctx = _fmt_ctx(all_ev)
    user_prompt = f"질문: {item['question']}\n\n공식 정답 요지: {item['expected_gist']}\n\n근거:\n{ctx}"

    answer, invalid = None, None
    for attempt in range(MAX_RETRY + 1):
        try:
            cand = llm.complete(good_sys, user_prompt if attempt == 0 else
                                 user_prompt + f"\n\n[검증 실패] 다음은 근거에 없는 식별자다: "
                                               f"{invalid}. 제거하고 다시 작성하라.",
                                 temperature=1)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "source_id": item["id"]}
        cand_invalid = _find_invalid_citations(cand, golden_keys)
        answer, invalid = cand, cand_invalid
        if not cand_invalid:
            break

    if invalid:
        return {"source_id": item["id"], "rejected": True, "invalid": invalid}
    return {
        "source_id": item["id"], "category": "grounded",
        "user_content": f"질문: {item['question']}\n\n근거:\n{ctx}",
        "answer": answer,
        "n_evidence": len(all_ev),
    }


def main():
    global t_start
    t_start = time.time()
    excluded = load_excluded_ids()
    pool = sample_pool(excluded)
    print(f"[준비] held-out {len(excluded)}건 제외, 후보 풀 {len(pool)}건", flush=True)

    done_ids = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["source_id"])
    print(f"[준비] 이미 생성됨: {len(done_ids)}건 — 건너뛰고 이어서", flush=True)

    print("[Index 로드]", flush=True)
    index = Index()
    p = Pipeline(index=index, local=True)
    sys_prompt = p._answer_system_prompt()   # 실제 서비스 프롬프트 그대로 사용 (건드리지 않음)
    L.configure_langsmith()
    llm = L.LLM("openai", MODEL, node="raft_gen_v3")

    pool = [x for x in pool if x["id"] not in done_ids]
    pos_candidates = pool[:POSITIVE_CANDIDATES]
    ref_candidates = pool[POSITIVE_CANDIDATES:POSITIVE_CANDIDATES + N_REFUSAL_TARGET + 50]

    n_pos_done = sum(1 for l in (OUT.read_text(encoding="utf-8").splitlines() if OUT.exists() else [])
                      if l.strip() and json.loads(l)["category"] == "grounded")
    n_ref_done = sum(1 for l in (OUT.read_text(encoding="utf-8").splitlines() if OUT.exists() else [])
                      if l.strip() and json.loads(l)["category"] == "refusal")
    print(f"[준비] 기존 채택 grounded={n_pos_done} refusal={n_ref_done}", flush=True)

    fout = OUT.open("a", encoding="utf-8")

    print(f"\n[1/2] 근거있음 병렬 생성 시작 (후보 {len(pos_candidates)}건, "
          f"동시 {MAX_WORKERS}개, 목표 {N_POSITIVE_TARGET}건, 근거 총량={PRODUCTION_K}개로 통일)",
          flush=True)
    pos_written = n_pos_done
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_positive, item, index, p, llm): item for item in pos_candidates}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            if res.get("error"):
                print(f"  [오류] id={res['source_id']} {res['error']}", flush=True)
                continue
            if res.get("rejected"):
                with write_lock:
                    progress["pos_rejected"] += 1
                print(f"  [거부] id={res['source_id']} 무효인용={res['invalid']} "
                      f"(누적거부={progress['pos_rejected']})", flush=True)
                continue
            record = {
                "source_id": res["source_id"], "category": "grounded",
                "messages": [{"role": "system", "content": sys_prompt},
                             {"role": "user", "content": res["user_content"]},
                             {"role": "assistant", "content": res["answer"]}],
            }
            with write_lock:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                pos_written += 1
                done_now = pos_written
            elapsed = time.time() - t_start
            print(f"  [채택 {done_now}/{N_POSITIVE_TARGET}] id={res['source_id']} "
                  f"근거{res['n_evidence']}개 누적 {elapsed:.0f}s", flush=True)
            if done_now >= N_POSITIVE_TARGET:
                break

    print(f"\n[2/2] refusal 순차 생성 시작 (목표 {N_REFUSAL_TARGET}건, GPT 호출 없음, "
          f"근거 총량={PRODUCTION_K}개로 통일)", flush=True)
    ref_written = n_ref_done
    for item in ref_candidates:
        if ref_written >= N_REFUSAL_TARGET:
            break
        colls = [c for c in item["expected_collections"] if c in index.colls] or list(index.colls)
        hits = index.retrieve_routed(item["question"], colls, k=8, min_standards=1, per_coll=12)
        golden_keys = set(item["expected_ref_keys"])
        # v3: 5개 고정 대신 production과 동일하게 최대 8개까지 채움 (골든은 전부 제외 유지)
        distractors_only = [h for h in hits if (h["ref_key"] or h["doc_no"]) not in golden_keys][:PRODUCTION_K]
        if len(distractors_only) < 2:
            continue
        ctx = _fmt_ctx([{"ref_key": h["ref_key"] or h["doc_no"], "collection": h["collection"],
                          "text": h["text"]} for h in distractors_only])
        record = {
            "source_id": item["id"], "category": "refusal",
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": f"질문: {item['question']}\n\n근거:\n{ctx}"},
                         {"role": "assistant", "content": Pipeline.REFUSAL}],
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        ref_written += 1
        print(f"  [refusal 채택 {ref_written}/{N_REFUSAL_TARGET}] id={item['id']} "
              f"근거{len(distractors_only)}개", flush=True)

    fout.close()
    print(f"\n총 소요 {time.time() - t_start:.0f}s (grounded={pos_written} refusal={ref_written} "
          f"거부={progress['pos_rejected']})", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
