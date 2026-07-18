# -*- coding: utf-8 -*-
"""원본 kasb_exaone_qlora_colab.ipynb을 읽어, 진단된 4가지 구조적 문제를 고친
kasb_exaone_qlora_colab_v2.ipynb를 생성한다. 원본 셀 순서/메타데이터는 최대한 보존하고
문제되는 셀만 교체/삽입/삭제한다.
"""
import json

SRC = "/Users/gyuyeong/Downloads/kasb_exaone_qlora_colab.ipynb"
DST = "/Users/gyuyeong/Downloads/kasb_exaone_qlora_colab_v2.ipynb"

with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
assert len(cells) == 55, f"예상과 다른 셀 수: {len(cells)}"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def keep(i):
    c = cells[i]
    return {
        "cell_type": c["cell_type"],
        "metadata": c.get("metadata", {}),
        "source": c["source"],
        **({"execution_count": None, "outputs": []} if c["cell_type"] == "code" else {}),
    }


# 원본 셀20(import 있음)과 셀22(import 없음)는 완전한 중복이 아니었다 --
# 22를 살리고 20을 지우면서 import가 빠짐(첫 배포 시 실수). 22에 import를 보강해 되돌린다.
CELL_22_FIXED_CODE = code('''import time
from transformers import AutoModelForCausalLM

MAX_RETRIES = 30
model = None
for attempt in range(1, MAX_RETRIES + 1):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"성공! (시도 {attempt}회 만에)")
        break
    except OSError as e:
        print(f"[시도 {attempt}/{MAX_RETRIES}] 실패 -- 30초 후 재시도 ({type(e).__name__})")
        time.sleep(30)

if model is None:
    raise RuntimeError("30회 재시도해도 실패 -- HuggingFace 서버 문제가 아직 안 풀린 것 같습니다.")
model.config.use_cache = False
''')


# ---------------------------------------------------------------- A. 데이터 길이 감사 (신규, 원본 셀14 뒤)
CELL_A_MD = md("## 4-1. 학습 데이터 길이 감사 (v1 실패 원인 1: max_length 4096이 최장 예시보다 짧아 정답이 잘렸음)\n"
               "700건 전체를 실제 채팅 템플릿으로 렌더링해 토큰 길이를 재고, 잘림 0건을 보장하는 `MAX_LENGTH`를 정합니다.\n")

CELL_A_CODE = code('''import math, statistics
from transformers import AutoTokenizer

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

lengths = [
    len(tokenizer.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False))
    for r in records
]
ls = sorted(lengths)
pct = lambda p: ls[min(len(ls) - 1, int(len(ls) * p / 100))]
max_len = ls[-1]

print(f"n={len(ls)}  max={max_len}  mean={statistics.mean(ls):.0f}  "
      f"p50={pct(50)}  p95={pct(95)}  p99={pct(99)}")
for th in (4096, 8192, 12288):
    n_over = sum(l > th for l in ls)
    print(f"  {th} 초과: {n_over}건 ({n_over / len(ls) * 100:.1f}%)")

_CANDIDATES = [4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576]
MAX_LENGTH = next((c for c in _CANDIDATES if c >= max_len + 16),
                  math.ceil((max_len + 16) / 1024) * 1024)
assert MAX_LENGTH >= max_len, "MAX_LENGTH < 최장 예시 -- v1 실패 원인(정답 잘림) 재발"
print(f"\\n선택된 MAX_LENGTH = {MAX_LENGTH} (최장 {max_len} 토큰을 잘림 없이 포함, 잘리는 예시 0건)")
''')

# ---------------------------------------------------------------- B. 원본 셀16 교체 (completion-only 분리)
CELL_16_CODE = code('''from transformers import AutoTokenizer
from trl import SFTConfig

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

assert "completion_only_loss" in SFTConfig.__dataclass_fields__, \\
    "이 trl 버전에는 completion_only_loss가 없음 -- !pip install -U trl 후 런타임 재시작"

_msgs = train_ds[0]["messages"]
assert _msgs[-1]["role"] == "assistant"
_full   = tokenizer.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=False)
_ctx    = tokenizer.apply_chat_template(_msgs[:-1], tokenize=False, add_generation_prompt=False)
_prompt = tokenizer.apply_chat_template(_msgs[:-1], tokenize=False, add_generation_prompt=True)
assert _full.startswith(_prompt), "generation_prompt 렌더링이 전체 렌더링의 접두사가 아님 -- 템플릿 확인 필요"

ASSISTANT_MARKER = _prompt[len(_ctx):]
assert ASSISTANT_MARKER.strip(), "assistant 마커가 비어 있음"
print(f"assistant 턴 시작 마커(실측): {ASSISTANT_MARKER!r}")
print(f"eos_token: {tokenizer.eos_token!r}")
print(f"템플릿 {{% generation %}} 지원: {'{% generation %}' in (tokenizer.chat_template or '')}")

def to_prompt_completion(example):
    msgs = example["messages"]
    assert msgs[-1]["role"] == "assistant", "마지막 메시지가 assistant가 아닌 예시 존재"
    full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    assert full.startswith(prompt)
    return {"prompt": prompt, "completion": full[len(prompt):]}

train_ds = train_ds.map(to_prompt_completion, remove_columns=train_ds.column_names)
val_ds = val_ds.map(to_prompt_completion, remove_columns=val_ds.column_names)

# 토큰 경계 검증: 분리 토큰화(prompt+completion)가 전체 토큰화와 완전히 일치해야
# 마스킹 경계가 오염되지 않음 (collator류의 흔한 함정)
for i in range(min(20, len(train_ds))):
    p, c = train_ds[i]["prompt"], train_ds[i]["completion"]
    sep = tokenizer(p, add_special_tokens=False).input_ids + tokenizer(c, add_special_tokens=False).input_ids
    joint = tokenizer(p + c, add_special_tokens=False).input_ids
    assert sep == joint, f"예시 {i}: prompt/completion 경계가 토큰화 시 어긋남 -- 마스킹 경계 오염 위험"
print("경계 검증 통과: 분리 토큰화 == 전체 토큰화 (20/20)")

print("--- completion 예시 (loss가 걸릴 유일한 부분) ---")
print(train_ds[0]["completion"][:400])
print("--- completion 끝부분 (종료 토큰 확인) ---")
print(repr(train_ds[0]["completion"][-40:]))
''')

# ---------------------------------------------------------------- B-2. 원본 셀19 교체 (create_causal_mask 패치)
# 원본은 모델을 최소 1번 로드해야 생기는 캐시 파일을 모델 로드 "전"에 찾으려 해서 새 런타임에서
# assert로 죽었음(2026-07-16 실측). 가중치는 안 받고 코드 파일만 먼저 캐시하도록 보강 + 못 찾아도
# 죽지 않고 경고만 하도록 완화.
CELL_19_FIXED_CODE = code('''import glob, re, inspect
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import create_causal_mask

sig_params = list(inspect.signature(create_causal_mask).parameters.keys())
has_cache_position = "cache_position" in sig_params
print("현재 create_causal_mask 인자:", sig_params)

# 캐시 파일이 아직 없으면(=모델을 한 번도 안 불러온 새 런타임) 가중치는 받지 않고
# 커스텀 모델링 코드 파일만 먼저 받아서 캐시를 채운다.
cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
model_class_ref = getattr(cfg, "auto_map", {}).get("AutoModelForCausalLM")
if model_class_ref:
    get_class_from_dynamic_module(model_class_ref, BASE_MODEL)
    print("모델링 코드 캐시 확보:", model_class_ref)

candidates = glob.glob("/root/.cache/huggingface/modules/transformers_modules/**/modeling_exaone.py", recursive=True)
print("찾은 파일:", candidates)

if not candidates:
    print("경고: modeling_exaone.py를 여전히 못 찾음 -- 이 버전은 패치가 불필요하거나 파일명이 다를 수 있음. "
          "패치 없이 다음 셀(모델 로드)로 진행하고, 거기서 에러가 나면 그때 다시 확인하세요.")
else:
    for path in candidates:
        src = open(path, encoding="utf-8").read()
        changed = False

        if "input_embeds=inputs_embeds" in src:
            src = src.replace("input_embeds=inputs_embeds", "inputs_embeds=inputs_embeds")
            changed = True
            print(f"[{path}] input_embeds -> inputs_embeds 치환")

        if not has_cache_position:
            pattern = re.compile(
                r"(attention_mask=attention_mask,\\s*\\n)(\\s*)cache_position=cache_position,\\s*\\n(\\s*)(past_key_values=past_key_values,)"
            )
            new_src, n = pattern.subn(r"\\1\\3\\4", src)
            if n:
                src = new_src
                changed = True
                print(f"[{path}] cache_position 인자 {n}곳 제거(현재 버전 미지원)")

        if changed:
            open(path, "w", encoding="utf-8").write(src)
            print(f"저장 완료: {path}")
        else:
            print(f"[{path}] 변경 없음")
''')

# ---------------------------------------------------------------- C. 원본 셀27 교체 (SFTConfig)
CELL_27_CODE = code('''from trl import SFTConfig

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
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,                  # epoch 3개 어댑터 체크포인트 전부 Drive 보존 (각 수백MB)
    load_best_model_at_end=True,         # 학습 종료 시 eval_loss 최소 체크포인트를 자동 로드
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_length=MAX_LENGTH,               # 감사 셀 실측 -- 잘림 0건 보장
    packing=False,
    completion_only_loss=True,           # prompt(근거문서+질문) loss 제외, completion(정답)에만 학습
    report_to="none",
)
''')

# ---------------------------------------------------------------- D. 원본 셀29 교체 (trainer + 마스킹 검증 + 학습)
CELL_29_CODE = code('''from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

# --- 학습 전 마스킹 검증: 실제 collate 결과에서 loss 대상 토큰 확인 ---
_rows = [trainer.train_dataset[i] for i in range(2)]
_batch = trainer.data_collator(_rows)
_labels = _batch["labels"]
_norm = lambda s: "".join(s.split())
for _i in range(_labels.shape[0]):
    lab = _labels[_i]
    n_total = int(_batch["attention_mask"][_i].sum()) if "attention_mask" in _batch else lab.numel()
    n_train = int((lab != -100).sum())
    assert n_train > 0, f"예시 {_i}: loss 대상 토큰 0개 -- 전체가 마스킹됨(마스킹 실패)"
    assert n_train < n_total, f"예시 {_i}: 전체 토큰에 loss -- 마스킹 미적용"
    print(f"예시 {_i}: 전체 {n_total} 토큰 중 loss 대상 {n_train}개 ({n_train/n_total:.1%})")
    if n_train / n_total > 0.7:
        print(f"  경고: loss 비율이 비정상적으로 높음 -- 아래 디코드 출력을 눈으로 확인할 것")
    trained_text = tokenizer.decode([t for t in lab.tolist() if t != -100])
    expected = train_ds[_i]["completion"]
    assert _norm(expected)[:30] in _norm(trained_text), \\
        f"예시 {_i}: loss 대상 토큰이 completion과 불일치 -- 마스킹 경계 오류"
    assert _norm(train_ds[_i]["prompt"])[len(_norm(train_ds[_i]["prompt"]))//2:][:30] not in _norm(trained_text), \\
        f"예시 {_i}: prompt(근거문서) 내용에 loss가 걸려 있음"

_recon = tokenizer.decode(_batch["input_ids"][0][_batch["attention_mask"][0].bool()])
_orig = train_ds[0]["prompt"] + train_ds[0]["completion"]
assert _norm(_orig)[:100] in _norm(_recon), "collate된 input_ids가 원본 렌더링과 불일치"
print("입력 끝부분(종료 토큰 중복 여부 확인):", repr(_recon[-80:]))
print("--- loss 대상 디코드(앞 200자) -- 정답 답변이어야 함 ---")
print(trained_text[:200])
print("\\n마스킹 검증 통과 -- 학습 시작")

trainer.train()
''')

# ---------------------------------------------------------------- E. 원본 셀32 교체 (best 어댑터 저장)
CELL_32_CODE = code('''print("best checkpoint:", trainer.state.best_model_checkpoint)
print("best eval_loss:", trainer.state.best_metric)

ADAPTER_DIR = f"{OUTPUT_DIR}/best_adapter"
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("저장 완료:", ADAPTER_DIR)
print("나머지 epoch 체크포인트도 Drive에 보존됨:", f"{OUTPUT_DIR}/checkpoint-*")
''')

# ---------------------------------------------------------------- F. 원본 셀34~52 -> D-1 / D-2
CELL_D1_CODE = code('''import os
!pip uninstall -y torchao hf_xet -q
os.environ["HF_HUB_DISABLE_XET"] = "1"

!git clone --depth 1 https://github.com/ggml-org/llama.cpp
!pip install -q -r llama.cpp/requirements.txt
!cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
!cmake --build llama.cpp/build --target llama-quantize -j 4
assert os.path.exists("llama.cpp/build/bin/llama-quantize"), "llama-quantize 빌드 실패"
''')

CELL_D2_CODE = code('''import os, gc, sys, types, shutil, subprocess
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def _require_free(gb, step, path="/content"):
    free = shutil.disk_usage(path).free / 1024**3
    print(f"[{step}] 디스크 여유 {free:.1f}GB (필요 최소 {gb}GB)")
    if free < gb:
        raise RuntimeError(
            f"[{step}] 여유 {free:.1f}GB < {gb}GB -- 중단. /content의 불필요 파일을 지우고 "
            f"재실행하세요. 단, HF 캐시(/root/.cache/huggingface)는 지우지 말 것(재사용됨).")

def _rm(path):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif path and os.path.exists(path):
        os.remove(path)

def convert_adapter_to_gguf(adapter_dir, tag, base_model=BASE_MODEL, work="/content"):
    """어댑터 1개 -> 병합(bf16) -> GGUF f16 -> Q4_K_M. 다른 epoch을 시도하려면
    adapter_dir와 tag만 바꿔 재호출. 완성된 중간산출물이 있으면 그 단계는 건너뜀.

    디스크 예산(60GB) 피크: HF캐시 16 + merged 16 + f16 16 ~= 47GB (2단계 변환 구간이 최대).
    참고: convert_hf_to_gguf.py --outtype은 f32/f16/bf16/q8_0/tq1_0/tq2_0만 지원 --
    Q8_0 배포라면 --outtype q8_0으로 f16 중간 단계를 생략할 수 있으나,
    Q4_K_M(K-quant)은 llama-quantize 전용이므로 f16 경유가 필수
    (q8_0에서 재양자화는 --allow-requantize 필요 + 품질 손실).
    """
    merged_dir = f"{work}/merged_{tag}"
    f16_path   = f"{work}/exaone_{tag}_f16.gguf"
    fixed_path = f"{work}/exaone_{tag}_f16_fixed.gguf"
    q4_path    = f"{work}/exaone_{tag}_q4_k_m.gguf"
    in_progress = None
    try:
        # 1/4 병합 -- 피크: 캐시16 + merged16 ~= 32GB
        if not (os.path.exists(merged_dir) or os.path.exists(f16_path) or os.path.exists(fixed_path)):
            _require_free(18, "1/4 병합")
            tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            _emb = next((m for m in base.modules() if isinstance(m, nn.Embedding)), None)
            assert _emb is not None, "EXAONE 임베딩 레이어를 못 찾음"
            base.get_input_embeddings = types.MethodType(lambda self: _emb, base)
            merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
            in_progress = merged_dir
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tok.save_pretrained(merged_dir)
            in_progress = None
            del base, merged
            gc.collect(); torch.cuda.empty_cache()

        # 2/4 GGUF f16 변환 -- 피크: 캐시16 + merged16 + f16 16 ~= 47GB (전체 최대 피크)
        if not (os.path.exists(f16_path) or os.path.exists(fixed_path)):
            _require_free(18, "2/4 GGUF f16 변환")
            in_progress = f16_path
            subprocess.run(["python", "llama.cpp/convert_hf_to_gguf.py", merged_dir,
                            "--outfile", f16_path, "--outtype", "f16"], check=True)
            assert os.path.getsize(f16_path) > 10 * 1024**3, "f16 gguf가 비정상적으로 작음(불완전)"
            in_progress = None
            _rm(merged_dir)  # 성공 즉시 병합본 삭제 -> +16GB 회수

        # 3/4 EXAONE 메타데이터 보정 -- rms_epsilon 키가 없을 때만 실행
        #     (최신 llama.cpp가 exaone 메타데이터를 제대로 쓰면 자동 스킵됨)
        src_gguf = fixed_path if os.path.exists(fixed_path) else f16_path
        if src_gguf == f16_path:
            sys.path.insert(0, "llama.cpp/gguf-py")
            import gguf
            reader = gguf.GGUFReader(f16_path, "r")
            arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
            has_rms = reader.get_field(f"{arch}.attention.layer_norm_rms_epsilon") is not None
            eps_field = reader.get_field(f"{arch}.attention.layer_norm_epsilon")
            if not has_rms and eps_field is not None:
                _require_free(18, "3/4 메타데이터 보정")  # f16 두 벌 공존 구간: 캐시16 + 16*2 ~= 47GB
                for _p in ("llama.cpp/gguf-py/gguf/scripts", "llama.cpp/gguf-py/scripts"):
                    if os.path.isdir(_p):
                        sys.path.insert(0, _p)
                try:
                    from gguf_new_metadata import copy_with_new_metadata, MetadataDetails
                except ImportError as e:
                    raise RuntimeError("gguf_new_metadata.py를 찾지 못함 -- llama.cpp 버전 확인 필요") from e
                new_md = {f"{arch}.attention.layer_norm_rms_epsilon":
                          MetadataDetails(gguf.GGUFValueType.FLOAT32, eps_field.contents())}
                in_progress = fixed_path
                writer = gguf.GGUFWriter(fixed_path, arch=arch, endianess=reader.endianess)
                copy_with_new_metadata(reader, writer, new_md, remove_metadata=[])
                in_progress = None
                del reader; gc.collect()
                os.remove(f16_path)
                src_gguf = fixed_path
            else:
                del reader; gc.collect()

        # 4/4 Q4_K_M 양자화 -- 피크: 캐시16 + f16 16 + q4 5 ~= 37GB
        _require_free(7, "4/4 Q4_K_M 양자화")
        in_progress = q4_path
        subprocess.run(["./llama.cpp/build/bin/llama-quantize", src_gguf, q4_path, "Q4_K_M"],
                       check=True)
        assert os.path.getsize(q4_path) > 3 * 1024**3, "q4 gguf가 비정상적으로 작음(불완전)"
        in_progress = None
        os.remove(src_gguf)  # 성공 즉시 f16 삭제 -> +16GB 회수

        print(f"완료: {q4_path} ({os.path.getsize(q4_path)/1024**3:.2f} GB)")
        return q4_path
    except BaseException:
        _rm(in_progress)  # 만들다 만 파일만 삭제 -- 완성된 중간산출물은 보존(재실행 시 이어감)
        raise

# ---- 실행: best 체크포인트 1개만 변환 (3회 반복 안 함 -- 디스크 문제의 핵심 해결) ----
for _v in ("model", "trainer", "base", "merged"):
    if _v in globals():
        del globals()[_v]
gc.collect(); torch.cuda.empty_cache()

ADAPTER_DIR = f"{OUTPUT_DIR}/best_adapter"
q4 = convert_adapter_to_gguf(ADAPTER_DIR, tag="best")
dst = os.path.join(OUTPUT_DIR, os.path.basename(q4))
shutil.copy(q4, dst)
print("Drive 저장 완료:", dst)

# held-out 30문항 평가에서 best가 실패하면, 다른 epoch을 어댑터 경로만 바꿔 재시도:
# convert_adapter_to_gguf(f"{OUTPUT_DIR}/checkpoint-<step>", tag="ep2")
''')

CELL_36_NOTE_MD = md("## 13. GGUF 변환 + 양자화 (Ollama 배포용)\n\n"
                     "`mlx_lm`의 GGUF 변환은 EXAONE 아키텍처를 지원하지 않아 `llama.cpp`의 "
                     "`convert_hf_to_gguf.py`를 씁니다. 원본 노트북의 셀 34~52는 디스크 부족으로 "
                     "여러 번 재시도한 흔적(중복·수동 임기응변 셀)이라 아래 2개 셀로 통합했습니다: "
                     "**성공한 단계 직후 이전 산출물을 즉시 삭제**하고, HF 캐시(베이스 모델 원본)는 "
                     "재사용을 위해 보존하며, `load_best_model_at_end`로 뽑힌 best 체크포인트 "
                     "1개만 변환합니다(60GB 예산, 3회 반복 안 함).\n")

# ---------------------------------------------------------------- G. 원본 셀53/54 텍스트 갱신
CELL_53_MD = md('''## 14. (로컬 맥에서 실행) Ollama에 새 태그로 배포

**아래는 Colab이 아니라 로컬 맥 터미널에서 실행하는 명령입니다.** 기존 `exaone3.5:7.8b`(베이스)와
`exaone3.5-lora:v1`(실패한 첫 어댑터, 거부율 66.7%/Faithfulness 0.800로 확인됨)은 롤백/비교용으로
그대로 남겨두고, 새 태그로 추가합니다.

```bash
# 1) 기존 태그의 template/parameter를 그대로 재사용하기 위해 먼저 확인
ollama show exaone3.5:7.8b --modelfile > /tmp/exaone_base.modelfile
cat /tmp/exaone_base.modelfile

# 2) 새 Modelfile 작성 -- FROM 줄만 새 gguf로 바꾸고 TEMPLATE/SYSTEM은 위에서 확인한 걸 그대로 복사.
#    PARAMETER 블록에는 repeat_penalty/num_predict를 반드시 추가하세요 -- v1이 route() 노드에서
#    반복생성 무한루프(문단번호를 끝없이 나열)에 빠졌던 문제의 재발 방지 안전장치입니다.
#    (재학습으로 이 결함 자체가 없어졌다는 보장은 없으므로, 재학습 여부와 무관하게 항상 넣습니다.)
cat > /tmp/exaone_lora.modelfile <<'EOF'
FROM /path/to/exaone_best_q4_k_m.gguf
# (여기에 exaone_base.modelfile의 TEMPLATE 블록을 그대로 붙여넣기)
PARAMETER repeat_penalty 1.15
PARAMETER num_predict 2048
EOF

# 3) 새 태그로 생성 (원본 exaone3.5:7.8b, exaone3.5-lora:v1은 그대로 보존됨)
ollama create exaone3.5-lora:v2 -f /tmp/exaone_lora.modelfile
```
''')

CELL_54_MD = md('''## 15. 필수 -- held-out 30문항 재평가

**이 단계 없이는 재학습이 실제로 도움이 됐는지 알 수 없습니다.** `KASB_LOCAL_MODEL=exaone3.5-lora:v2`
환경변수로 `exaone_after_lora_eval.py`와 같은 패턴의 스크립트를 다시 돌려서, 베이스라인(거부율
33%/10건, 평균 Faithfulness 0.900)과 **v1의 실패 결과(거부율 66.7%, Faithfulness 0.800)** 둘 다와
비교하세요.

- Faithfulness가 베이스라인보다 오르고 거부율이 과도하게(예: 60%+) 튀지 않으면 -> 성공, 새 태그를 실제로 스왑
- Faithfulness는 안 올랐는데 거부율만 여전히 높으면 -> v1과 같은 '인용 강박' 재발, best_adapter가 아닌
  다른 epoch(`checkpoint-<step>`)을 `convert_adapter_to_gguf()`로 다시 변환해 재시도
  (Drive에 3개 체크포인트가 다 보존돼 있으므로 재학습 없이 바로 가능)
- 그래도 개선이 없으면 -> 700개 규모 데이터/하이퍼파라미터 재탐색 필요 (솔직하게 실패로 기록)

```bash
KASB_LOCAL_MODEL=exaone3.5-lora:v2 python exaone_after_lora_eval.py
```
''')

# ================================================================== 조립
new_cells = []
new_cells += [keep(i) for i in range(0, 15)]          # 0~14 그대로
new_cells += [CELL_A_MD, CELL_A_CODE]                  # 신규: 데이터 길이 감사
new_cells += [keep(15)]                                # 토크나이저 설명 마크다운
new_cells += [CELL_16_CODE]                            # 16 교체
new_cells += [keep(17), keep(18)]                        # 17~18 그대로
new_cells += [CELL_19_FIXED_CODE]                         # 19 교체(캐시 선확보 + assert 완화)
# 20(중복 모델로드, import 있음) 스킵, 21(마크다운)+22(모델로드, import 보강) 유지
new_cells += [keep(21), CELL_22_FIXED_CODE]
new_cells += [keep(i) for i in range(23, 27)]           # 23~26 그대로
new_cells += [CELL_27_CODE]                             # 27 교체
new_cells += [keep(28)]                                 # 학습실행 헤더
new_cells += [CELL_29_CODE]                             # 29 교체
new_cells += [keep(30), keep(31)]                       # loss 해석 안내, 어댑터저장 헤더
new_cells += [CELL_32_CODE]                             # 32 교체
new_cells += [keep(33)]                                 # 병합 헤더
# 34~52 스킵(중복/디스크문제 원인) -> 새 헤더 + D-1 + D-2
new_cells += [CELL_36_NOTE_MD, CELL_D1_CODE, CELL_D2_CODE]
new_cells += [CELL_53_MD, CELL_54_MD]                    # 53, 54 갱신

nb["cells"] = new_cells
nb["metadata"].pop("widgets", None)  # 원본 실행결과에 딸린 위젯 상태(더는 안 맞음) 제거

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"생성 완료: {DST}")
print(f"셀 수: 원본 {len(cells)} -> 신규 {len(new_cells)}")
