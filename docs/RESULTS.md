# 버전별 상세 기록

## 공통 하이퍼파라미터 (QLoRA 논문값, 3버전 전부 동일)

Dettmers et al. 2023(arXiv:2305.14314) 확정값을 그대로 사용, 임의 튜닝 없음.

| 항목 | 값 | 출처 |
|---|---|---|
| 양자화 | NF4 + double quantization | QLoRA 논문 |
| compute dtype | bf16 | QLoRA 논문 |
| LoRA r | 64 | QLoRA 논문 Table 9 (7B) |
| LoRA alpha | 16 | QLoRA 논문 |
| LoRA dropout | 0.1 | QLoRA 논문 |
| target_modules | all-linear | QLoRA 논문 |
| learning rate | 2e-4, constant schedule | QLoRA 논문 Table 9 / Appendix B.2 |
| optimizer | paged_adamw_32bit | QLoRA 논문 |
| adam_beta2 | 0.999 | QLoRA 논문 Appendix B.2 |
| max_grad_norm | 0.3 | QLoRA 논문 Appendix B.2 |
| num_train_epochs | 3 | 고정(3버전 동일, 변경한 적 없음) |
| per_device_train_batch_size / grad_accum | MAX_LENGTH에 따라 동적 결정: >8192면 (1,16) / >4096면 (2,8) / 그 외 (4,4) | 실효 배치 크기를 16으로 통일하기 위함 |

**버전 간 실제로 바뀐 것은 하이퍼파라미터가 아니라 (1) SFT 손실 계산 범위, (2) 체크포인트 선택
방식, (3) 학습 데이터 구성이다.** 아래 표로 요약:

| | v1 | v2 | v3 |
|---|---|---|---|
| loss 계산 범위 | 전체 시퀀스(프롬프트+정답) | completion(정답)만 (`completion_only_loss=True`) | v2와 동일 |
| max_length | 4096 고정 | 데이터 실측 후 동적 산출 | v2와 동일 |
| 체크포인트 선택 | `load_best_model_at_end` 있다고 markdown엔 써있었지만 실제로는 항상 epoch 3 저장(버그) | `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"` (정상 작동) | eval_loss 대신 **매 epoch 실제 생성 + GPT-4o-mini 판사 채점**으로 선택 |
| 학습데이터 근거 문서 수 | golden + distractor 최대 3개(grounded), distractor 최대 5개(refusal) — production(k=8)보다 적음 | v1과 동일 데이터 재사용 | golden 유지 + distractor를 8개까지 채움(grounded), distractor 최대 8개(refusal) — production과 통일 |
| LoRA 적용 범위 | route/rewrite/answer 전 노드 (버그) | answer 노드만 (수정 완료) | v2와 동일 |

---

## v1 — 표준 SFT, 최초 시도

**결과**: 거부율 66.7%(20/30), Faithfulness 0.800.

**학습 데이터**: RAFT 스타일 700건(`data/raft_train_data_v1v2.jsonl`, `scripts/raft_data_gen_v1v2.py`로 생성).
- grounded 600건: 골든셋(held-out 30문항 제외)에서 정답 근거(`expected_ref_keys`)가 있는 질문을
  뽑아, 실제 검색(BM25+dense+rerank, k=8, per_coll=12)으로 얻은 후보 중 golden이 아닌 것 최대
  3개를 distractor로 추가. GPT-5.4-mini에게 "골든 정답요지+근거만 근거로 자연스러운 한국어 답변
  재구성, 근거에 없는 내용 금지, 인용은 `[식별자]`" 지시로 답변 생성. 생성된 답변에 무효 인용(근거에
  없는 식별자)이 있으면 최대 1회 재시도, 그래도 있으면 채택 거부.
- refusal 100건: 같은 방식으로 검색하되 **golden을 의도적으로 제외**하고 distractor만(최대 5개)
  보여준 뒤, 정답을 `Pipeline.REFUSAL`("근거를 찾지 못했습니다.") 고정 텍스트로.
- 660 train / 40 internal-val(학습 중 체크포인트 비교용, 최종 30문항 평가와는 별개의 세트).

**근본 원인(2가지, 둘 다 프로덕션 코드에서 확인·수정)**:
1. LoRA 어댑터가 `route`/`rewrite` 노드에도 적용됨 — 이 노드들의 프롬프트는 학습 데이터 분포 밖
   (out-of-distribution)이라, LoRA가 반복 생성 루프에 빠지는 버그 발생. **수정**: `rag/graph.py`의
   route/rewrite는 항상 베이스 모델만 쓰도록 고정, answer 노드에만 LoRA 적용 가능하게 변경.
2. `max_length=4096`이 학습 데이터의 최장 예시(~14,000자)보다 짧아 정답이 중간에 잘림 —
   완성되지 않은 답변을 그대로 학습.
3. (부수적) `load_best_model_at_end`가 markdown엔 "3 에폭 중 최적 선택"이라 적혀 있었지만 실제
   코드는 그 옵션이 빠져 있어 **항상 마지막 에폭(3)을 그대로 저장**하는 버그.

---

## v2 — 구조적 결함 수정 (completion-only loss + 동적 길이 + 실제 체크포인트 선택)

**결과**: 거부율 46.7%(14/30, v1 대비 개선), Faithfulness 0.750(v1의 0.800보다 더 나빠짐), Answer Relevancy 0.644.

**학습 데이터**: v1과 동일 파일(`data/raft_train_data_v1v2.jsonl`) 재사용 — 데이터는 안 바꾸고
학습 절차만 고침.

**바꾼 것 3가지**:
1. `SFTConfig(completion_only_loss=True)` — 프롬프트(질문+근거문서) 부분은 loss 계산에서 제외,
   정답(completion)에만 학습. (`DataCollatorForCompletionOnlyLM`은 최신 trl에서 제거됨 —
   prompt/completion 분리 데이터셋 포맷 + 이 옵션이 현재 올바른 방법. GitHub Discussion #3826으로 확인.)
2. 학습 데이터 전수 조사로 실제 최장 토큰 길이를 측정해 `MAX_LENGTH`를 동적으로 산출(잘리는 예시 0건 보장).
3. `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"`를 실제로 정상 작동하게 구현
   (v1의 markdown-only 버그 수정).

**결과가 v1보다 더 나빠진 이유(유력 가설, LIMA 논문으로 뒷받침됨)**: `eval_loss`(다음 토큰 예측
손실)로 고른 "최적" 체크포인트가 실제 생성 품질과 무관했을 가능성. LIMA 논문(Zhou et al. 2023)이
정확히 이 리스크를 경고했다 — perplexity가 실제 답변 품질과 상관관계가 없어, 자신들은 held-out
셋에 대해 사람이 직접 생성 결과를 보고 체크포인트를 골랐다고 명시.
(v2 원본 노트북의 markdown 셀에도 이 경고가 이미 인용돼 있었는데, 실제 자동선택 로직은 여전히
`eval_loss` 기준으로 짜여 있었음 — v3에서 이 모순을 해소.)

부수 관찰: v2가 거부한 14문항 전부 검색 자체는 성공(근거자료 8개씩 검색됨) — 즉 검색 실패가
아니라 모델이 충분한 근거를 보고도 "확신 없다"며 거부. 답변한 16문항 중 9문항에 근거 없는
내용(unsupported claims) 포함.

---

## v3 — 학습·실전 근거 개수 통일 + 생성 기반 체크포인트 선택

**결과**: 거부율 43.3%(13/30), Faithfulness 0.794, Answer Relevancy 0.724. v1·v2보다 두 지표
모두 개선(트레이드오프 없이 동시 개선)됐으나 베이스라인(0.900)은 여전히 못 넘음. 답변한 17문항 중
7문항에 unsupported claims 잔존.

**리서치 근거**(fable5 모델로 조사, 1차 출처 확인):
- RAFT 원 논문(Zhang et al. 2024, arXiv:2403.10131): 학습 시 보여주는 golden:distractor 구성이
  실전 검색 조건과 어긋나면 성능이 떨어진다는 지적. 실제 프로덕션(`rag/graph.py`의 `retrieve()`)은
  항상 `k=8`(근거자료 8개)을 answer 노드에 넘기는데, v1/v2 학습 데이터는 grounded 예시에 golden+
  distractor 최대 3개(총 4개 안팎), refusal 예시엔 distractor 최대 5개만 보여줬음 — "연습 조건"과
  "실전 조건"이 달랐던 것.
- LIMA 논문: 위 v2 섹션 참조. eval_loss 기준 선택을 폐기하는 근거.

**바꾼 것 2가지** (근거 인용 유도(원문 직접인용) 지시는 베이스라인과의 공정한 비교를 위해 이번엔
의도적으로 제외 — `scripts/raft_data_gen_v3_fullregen_unused.py`에 구현은 해뒀으나 미사용):

1. **데이터 패치**(`scripts/raft_data_patch_v3.py`, `data/raft_train_data_v3.jsonl`): GPT 재호출
   없이 로컬 재검색만으로 근거 문서 개수를 production과 동일한 8개로 보정.
   - grounded 600건: golden은 그대로 유지, distractor만 추가로 검색해 총 8개로 채움
     (실제 결과: 579/600건이 정확히 8개, 나머지는 golden 자체가 8개보다 많아 더 늘어난 소수 케이스).
   - refusal 100건: golden 계속 제외, distractor를 5개→8개로 확대(실제: 60/100 정확히 8개,
     36/100은 후보 부족으로 7개).
   - **정답(assistant) 텍스트는 전혀 건드리지 않음** — 이미 golden 근거에 맞춰 검증된 문장이라
     문서 "개수"만 바뀌는 이 패치로는 재작성이 불필요하다는 판단.
2. **체크포인트 선택 방식 교체**(`scripts/build_v3_notebook.py`의 새 셀): `load_best_model_at_end`/
   `eval_loss` 대신, 저장된 3개 체크포인트(`checkpoint-42`/`-84`/`-126`, 3 에폭) 각각에 대해
   internal-val 40건을 **실제로 생성**시키고 GPT-4o-mini 판사로 채점(정답범주가 grounded면 근거
   부합 여부, refusal이면 실제 거부 여부로 0/0.5/1점) → 평균 점수가 가장 높은 체크포인트를 최종
   채택. (주의: 이 40건 채점 점수는 0~1 척도라도 held-out 30문항의 Faithfulness와는 **다른
   지표**이니 절대값을 직접 비교하면 안 됨 — 체크포인트끼리의 상대 비교용.)

**남은 한계**: 근거 문서 개수를 맞췄음에도 여전히 베이스라인보다 낮고, 답변 17건 중 7건에 근거
이탈이 남아있음 — "학습-실전 조건 불일치"만으로 전체 격차가 설명되지는 않는다는 뜻. 다음 후보로
검토했으나 이번엔 미적용: (a) 원문 직접인용 유도(RAFT의 CoT+quote 포맷, +9.7~14.9%p 보고됨,
`raft_data_gen_v3_fullregen_unused.py`에 구현됨 — 베이스라인에도 동일 프롬프트를 붙여야 공정 비교),
(b) Trust-Align(Song et al. 2025, arXiv:2409.11242) 스타일로 "부당 거부" 사례를 negative 예시로
명시적 포함, (c) 파인튜닝 자체를 접고 "According to..." 프롬프팅(Weller et al. 2024,
arXiv:2305.13252)으로 베이스 모델을 개선하는 무학습 경로.

---

## 최종 결정 (2026-07-17, kasb-crawler 프로덕션)

3회 재학습 전부 베이스라인의 Faithfulness(0.900)를 못 넘어, **로컬 모델은 베이스
`exaone3.5:7.8b`로 고정**하고 파인튜닝 선택 UI(`rag/llm.py`의 `LOCAL_MODELS`,
`rag/app.py` 사이드바)를 제거했다. v1 아티팩트는 삭제, v2 아티팩트는 v3 시도 중 디스크 확보를
위해 삭제, v3 아티팩트(Ollama 태그 `exaone3.5-lora:v3`, GGUF 파일)는 이 실험 기록의 최종
산출물로 로컬에 보존 중.
