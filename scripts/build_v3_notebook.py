# -*- coding: utf-8 -*-
"""v2 노트북(kasb_exaone_qlora_colab_v2.ipynb, 40셀) -> v3 노트북 빌드.

v2 대비 변경점(리서치 근거: fable5 조사, arXiv:2403.10131 RAFT / arXiv:2305.11206 LIMA):
1. 학습 데이터 파일을 raft_train_data_v3.jsonl로 교체(distractor를 production k=8에 맞춤,
   CoT+원문인용 포맷 추가 -- raft_data_gen_v3.py로 생성).
2. 체크포인트 선택: eval_loss 기반 load_best_model_at_end 제거 -> 매 epoch 체크포인트를
   실제로 생성시켜 GPT-4o-mini 판사로 채점하고 그 점수로 최적 epoch을 고르는 방식으로 교체.
   (LIMA 논문이 지적한 "perplexity와 생성 품질 무상관"이 v2 실패의 유력 원인이었음)
3. 배포/최종평가 안내에 v2 실측 결과(거부율 46.7%, Faithfulness 0.750)를 비교 기준으로 추가.

셀 0-26(GPU 확인 ~ LoRA 설정)은 v2와 동일 -- 그대로 재사용.
"""
import json
from pathlib import Path

SRC = Path("/Users/gyuyeong/Downloads/kasb_exaone_qlora_colab_v2.ipynb")
DST = Path("/Users/gyuyeong/Downloads/kasb_exaone_qlora_colab_v3.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def keep(cells, i):
    c = cells[i]
    return {"cell_type": c["cell_type"], "metadata": c.get("metadata", {}), "source": c["source"],
            **({"execution_count": None, "outputs": []} if c["cell_type"] == "code" else {})}


CELL_UPLOAD_MD = md("## 3. 학습 데이터 업로드\n\n"
                     "**v3: `raft_train_data_v3.jsonl`을 업로드하세요** "
                     "(`raft_data_gen_v3.py`로 생성 -- v1/v2가 쓴 `raft_train_data_full.jsonl`과는 "
                     "다른 파일입니다. v3는 grounded/refusal 예시 모두 근거 문서 수를 production의 "
                     "k=8과 동일하게 맞췄고, 답변에 원문 직접인용을 유도하는 지시가 추가됐습니다).")

CELL_UPLOAD_CODE = code('from google.colab import files\n'
                         'uploaded = files.upload()  # raft_train_data_v3.jsonl 선택\n'
                         'DATA_PATH = "raft_train_data_v3.jsonl"\n')

CELL_SFTCONFIG_CODE = code(
'''from trl import SFTConfig

OUTPUT_DIR = "/content/drive/MyDrive/kasb_lora/checkpoints" if 'WORK_DIR' in dir() else "./checkpoints"

if MAX_LENGTH > 8192:
    BS, GA = 1, 16
elif MAX_LENGTH > 4096:
    BS, GA = 2, 8
else:
    BS, GA = 4, 4
print(f"MAX_LENGTH={MAX_LENGTH} -> batch={BS}, grad_accum={GA} (실효 배치 {BS*GA})")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GA,
    learning_rate=2e-4,                  # [논문 확정값] QLoRA Table 9 (7B)
    lr_scheduler_type="constant",        # [논문 확정값] QLoRA Appendix B.2
    optim="paged_adamw_32bit",           # [논문 확정값] QLoRA paged optimizer
    adam_beta2=0.999,                    # [논문 확정값] QLoRA Appendix B.2
    max_grad_norm=0.3,                   # [논문 확정값] QLoRA Appendix B.2
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",               # eval_loss는 계속 로깅(참고/진단용) -- 선택 기준으로는 안 씀
    save_strategy="epoch",
    save_total_limit=3,                  # epoch 3개 어댑터 체크포인트 전부 Drive 보존 (각 수백MB)
    # v3: load_best_model_at_end / metric_for_best_model="eval_loss" 제거.
    # v2가 이 기준으로 골랐다가 Faithfulness가 v1보다도 나빠졌음(0.750) -- LIMA 논문
    # (Zhou et al. 2023, arXiv:2305.11206)이 "perplexity는 생성 품질과 무상관"이라 경고한
    # 그대로였음. 대신 아래 11번 셀에서 매 epoch을 실제로 생성시켜 판사로 채점해 고른다.
    max_length=MAX_LENGTH,               # 감사 셀 실측 -- 잘림 0건 보장
    packing=False,
    completion_only_loss=True,           # prompt(근거문서+질문) loss 제외, completion(정답)에만 학습
    report_to="none",
)
''')

CELL_31_MD = md("학습 로그의 `loss`(train)와 `eval_loss`(내부 검증)를 epoch마다 비교해서 눈으로만 "
                 "참고하세요(과적합 조짐 파악용). **최종 체크포인트 선택은 이 값으로 하지 않습니다** "
                 "-- 바로 다음 셀에서 매 epoch을 실제로 생성시켜 판사로 채점하는 방식으로 자동 선택합니다 "
                 "(LIMA 논문이 지적한, perplexity와 실제 생성 품질의 무상관 문제를 피하기 위함).")

CELL_11_MD = md("## 11. 체크포인트별 생성 + 판사 채점 -> 최적 epoch 자동 선택\n\n"
                 "v2는 `eval_loss`가 가장 낮은 체크포인트를 자동 선택했는데, 그 체크포인트가 실제로는 "
                 "가장 나빴습니다(Faithfulness 0.750, v1의 0.800보다도 낮음). LIMA 논문이 경고한 "
                 "\"perplexity는 생성 품질과 상관없다\"가 그대로 재현된 것으로 보입니다.\n\n"
                 "그래서 v3는 매 epoch 체크포인트마다 **internal-val 40개에 대해 실제로 답변을 생성**시키고, "
                 "**GPT-4o-mini 판사**(지금까지 베이스라인/v1/v2 채점과 동일 모델 -- 비교 일관성 유지)로 "
                 "채점해서, 그 점수가 가장 높은 체크포인트를 고릅니다. 채점 기준: "
                 "grounded 문항은 근거에 실제로 부합하게 답했으면 1점(지어내면 0점, 부당 거부하면 0점), "
                 "refusal 문항은 실제로 거부했으면 1점(근거 없이 지어내 답하면 0점).\n\n"
                 "OpenAI API 키가 필요합니다(판사 호출용, Colab 세션에만 보관되고 저장되지 않습니다).")

CELL_JUDGE_SELECT_CODE = code(
'''import gc, glob, json, os, re
from getpass import getpass

import httpx
import torch
from openai import OpenAI as _OpenAI

OPENAI_API_KEY = getpass("OpenAI API 키 입력 (판사용, 세션에만 보관, 저장 안 됨): ").strip()
_judge_client = _OpenAI(api_key=OPENAI_API_KEY, timeout=httpx.Timeout(120.0, connect=10.0))
JUDGE_MODEL = "gpt-4o-mini"   # 베이스라인/v1/v2 채점과 동일 모델 -- 비교 일관성 유지

# 키 사전 검증 -- 틀린 키로 40건x3체크포인트를 다 돌리고 나서야 알아채는 것을 방지
try:
    _judge_client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, max_tokens=5,
        messages=[{"role": "user", "content": "ping"}])
    print("OpenAI API 키 확인 완료 -- 판사 호출 가능")
except Exception as e:  # noqa: BLE001
    raise RuntimeError(
        f"OpenAI API 키 검증 실패: {e}\\n"
        "-- 키를 다시 복사(앞뒤 공백/줄바꿈 없이)해서 이 셀을 재실행하세요. "
        "https://platform.openai.com/account/api-keys 에서 키가 살아있는지도 확인하세요.") from e

JUDGE_SYS = (
    "너는 한국 회계기준 RAG 답변 채점자다. 질문/근거/모델답변/정답범주를 보고 JSON으로만 "
    '답한다: {"score": 0~1 실수, "reason": "한 문장"}. 채점 기준: '
    "정답범주가 'grounded'면 -- 모델이 실제로 답변했고(거부하지 않았고) 그 내용이 근거에 "
    "있는 사실과 일치하면 1.0, 근거에 없는 내용을 지어냈으면 0.0, 일부만 지어냈으면 0.5, "
    "정당한 근거가 있는데도 거부했으면(부당 거부) 0.0. "
    "정답범주가 'refusal'이면 -- 모델이 실제로 거부했으면(짧게 모른다고 답했으면) 1.0, "
    "근거에 없는 내용을 지어내 답했으면 0.0."
)


def judge_one(question, grounding, category, model_answer):
    user = (f"정답범주: {category}\\n\\n질문: {question}\\n\\n근거:\\n{grounding[:3000]}\\n\\n"
            f"모델 답변: {model_answer}")
    try:
        r = _judge_client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}])
        return float(json.loads(r.choices[0].message.content).get("score", 0.0))
    except Exception as e:  # noqa: BLE001
        print("  [판사 오류]", e)
        return None


def generate_answer(gen_model, prompt_text, max_new_tokens=512):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.15, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# 학습이 끝났으니 옵티마이저 상태 등 학습 전용 메모리를 먼저 비운다 (생성 메모리 확보)
del trainer
gc.collect()
torch.cuda.empty_cache()

ckpt_dirs = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"),
                   key=lambda p: int(re.search(r"checkpoint-(\\d+)", p).group(1)))
assert ckpt_dirs, f"{OUTPUT_DIR}에 checkpoint-* 없음 -- save_strategy 설정을 확인하세요"
print(f"평가할 체크포인트: {[os.path.basename(c) for c in ckpt_dirs]} ({len(val_records)}건씩 채점)")

model.eval()
results = {}
for ckpt in ckpt_dirs:
    name = os.path.basename(ckpt)
    print(f"\\n=== {name} 로드 + 생성 + 채점 ===")
    model.load_adapter(ckpt, adapter_name=name)   # 기존 PeftModel에 이 체크포인트를 추가 로드
    model.set_adapter(name)                       # 생성 시 이 어댑터를 활성화

    scores = []
    for i, rec in enumerate(val_records):
        msgs = rec["messages"]
        prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        user_content = msgs[1]["content"]   # "질문: ...\\n\\n근거:\\n..."
        question = user_content.split("근거:")[0].replace("질문:", "").strip()
        grounding = user_content.split("근거:", 1)[1] if "근거:" in user_content else user_content
        ans = generate_answer(model, prompt_text)
        sc = judge_one(question, grounding, rec["category"], ans)
        if sc is not None:
            scores.append(sc)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(val_records)} 진행, 현재 평균 {sum(scores)/len(scores):.3f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    results[ckpt] = avg
    print(f"{name} 최종 평균 점수: {avg:.3f} ({len(scores)}/{len(val_records)}건 채점됨)")

best_ckpt = max(results, key=results.get)
print("\\n=== 체크포인트별 점수 (생성+판사 채점 기준) ===")
for c, s in sorted(results.items(), key=lambda x: -x[1]):
    marker = " <- 선택" if c == best_ckpt else ""
    print(f"  {os.path.basename(c)}: {s:.3f}{marker}")
print(f"\\n최종 선택: {best_ckpt}")
''')

CELL_SAVE_BEST_CODE = code(
'''import shutil

ADAPTER_DIR = f"{OUTPUT_DIR}/best_adapter"
if os.path.exists(ADAPTER_DIR):
    shutil.rmtree(ADAPTER_DIR)
shutil.copytree(best_ckpt, ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("저장 완료 (생성+판사 채점 기준 최적 체크포인트):", ADAPTER_DIR, "<-", os.path.basename(best_ckpt))
print("나머지 epoch 체크포인트도 Drive에 보존됨:", f"{OUTPUT_DIR}/checkpoint-*")
''')

CELL_DEPLOY_MD = md(
'''## 14. (로컬 맥에서 실행) Ollama에 새 태그로 배포

**아래는 Colab이 아니라 로컬 맥 터미널에서 실행하는 명령입니다.** 기존 태그들은 롤백/비교용으로
그대로 남겨두고 새 태그로 추가합니다.
- `exaone3.5:7.8b`(베이스, 거부율 33%/Faithfulness 0.900 -- 지금까지 최고 성능)
- `exaone3.5-lora:v1`(1차 재학습 실패, 거부율 66.7%/Faithfulness 0.800)
- `exaone3.5-lora:v2`(2차 재학습 실패, 거부율 46.7%/Faithfulness 0.750 -- eval_loss 기준 체크포인트 선택의 부작용으로 추정)

```bash
# 1) 기존 태그의 template/parameter를 그대로 재사용하기 위해 먼저 확인
ollama show exaone3.5:7.8b --modelfile > /tmp/exaone_base.modelfile
cat /tmp/exaone_base.modelfile

# 2) 새 Modelfile 작성 -- FROM 줄만 새 gguf로 바꾸고 TEMPLATE/SYSTEM은 위에서 확인한 걸 그대로 복사.
#    PARAMETER 블록에는 repeat_penalty/num_predict를 반드시 추가하세요 -- v1이 route() 노드에서
#    반복생성 무한루프에 빠졌던 문제의 재발 방지 안전장치입니다(재학습 여부와 무관하게 항상 넣습니다).
cat > /tmp/exaone_lora_v3.modelfile <<'EOF'
FROM /path/to/exaone_best_q4_k_m.gguf
# (여기에 exaone_base.modelfile의 TEMPLATE 블록을 그대로 붙여넣기)
PARAMETER repeat_penalty 1.15
PARAMETER num_predict 2048
EOF

# 3) 새 태그로 생성 (기존 태그들은 그대로 보존됨)
ollama create exaone3.5-lora:v3 -f /tmp/exaone_lora_v3.modelfile
```
''')

CELL_EVAL_MD = md(
'''## 15. 필수 -- held-out 30문항 재평가

**이 단계 없이는 재학습이 실제로 도움이 됐는지 알 수 없습니다.** `KASB_LOCAL_MODEL=exaone3.5-lora:v3`
환경변수로 held-out 30문항 평가 스크립트를 다시 돌려서, 지금까지의 세 결과와 모두 비교하세요.

| | 거부율 | Faithfulness |
|---|---|---|
| 베이스라인(파인튜닝 안 함) | 33.0% | **0.900** (지금까지 최고) |
| v1(표준 SFT, eval_loss 선택) | 66.7% | 0.800 |
| v2(completion-only loss, eval_loss 선택) | 46.7% | 0.750 (v1보다도 낮음) |
| v3(위 두 개 + 생성 기반 체크포인트 선택) | ? | ? |

- Faithfulness가 0.900을 넘고 거부율이 과도하게(예: 40%+) 튀지 않으면 -> 성공, 새 태그를 실제로 스왑
- 이번에도 베이스라인을 못 넘으면 -> Drive에 보존된 다른 epoch 체크포인트(`checkpoint-<step>`)를
  `convert_adapter_to_gguf()`로 재변환해 시도해보되, 그래도 안 되면 파인튜닝 자체를 재고할 근거로
  기록하세요(리서치에서 확인된 "According to..." 프롬프팅 등 무학습 대안이 이미 후보로 있습니다).

```bash
KASB_LOCAL_MODEL=exaone3.5-lora:v3 python exaone_v3_final_eval.py
```
''')


def main():
    nb = json.load(SRC.open(encoding="utf-8"))
    cells = nb["cells"]

    new_cells = []
    new_cells += [keep(cells, i) for i in range(0, 9)]     # 0~8: GPU~데이터 업로드 헤더 직전까지
    new_cells += [CELL_UPLOAD_MD, CELL_UPLOAD_CODE]         # 9~10 교체: v3 데이터 파일명
    new_cells += [keep(cells, i) for i in range(11, 27)]    # 11~26: Drive마운트~LoRA설정 그대로
    new_cells += [keep(cells, 27)]                          # 27: "## 9. 학습 설정" 헤더
    new_cells += [CELL_SFTCONFIG_CODE]                      # 28 교체: eval_loss 선택 제거
    new_cells += [keep(cells, 29)]                          # 29: "## 10. 학습 실행" 헤더
    new_cells += [keep(cells, 30)]                          # 30: SFTTrainer + train (그대로)
    new_cells += [CELL_31_MD]                                # 31 교체: 안내 문구
    new_cells += [CELL_11_MD, CELL_JUDGE_SELECT_CODE, CELL_SAVE_BEST_CODE]  # 새 체크포인트 선택 단계
    new_cells += [keep(cells, i) for i in range(34, 38)]    # 34~37: 병합/GGUF 변환 그대로
    new_cells += [CELL_DEPLOY_MD, CELL_EVAL_MD]              # 38~39 교체: v3 태그/비교표

    nb["cells"] = new_cells
    nb["metadata"].pop("widgets", None)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 완료: {DST} ({len(new_cells)}셀)")


if __name__ == "__main__":
    main()
