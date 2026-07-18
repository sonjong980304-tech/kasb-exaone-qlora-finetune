# kasb-exaone-qlora-finetune

[kasb-crawler](https://github.com/) 프로젝트(한국 회계기준 RAG 챗봇)의 로컬 모델(LG EXAONE-3.5-7.8B-Instruct)을
QLoRA로 파인튜닝해서 답변 품질(근거 충실도)을 개선하려 한 3차례 실험 기록.

## 결론부터

**3번 다 실패했다.** 파인튜닝 안 한 베이스 모델이 세 번의 재학습 시도보다 계속 더 나았다.
kasb-crawler 프로덕션은 로컬 모델을 베이스(`exaone3.5:7.8b`)로 고정했고, 이 저장소는
"왜 실패했는지", "무엇을 시도했는지"를 기록해 같은 시행착오를 반복하지 않기 위한 것이다.

## 평가 방법 (모든 버전 공통)

- **held-out 30문항**: `eval/goldenset.jsonl`(전체 1195문항, 학습 데이터에서 제외된 순수 평가용)에서
  게시판별 비례 샘플링(seed=42)한 30문항. 네 버전(베이스라인/v1/v2/v3) 전부 동일 샘플.
- **판사**: GPT-4o-mini, 전 버전 동일 모델(비교 일관성 유지). `rag/eval/judge.py`의
  Faithfulness(답변 주장이 근거자료로 뒷받침되는 비율, 0~1) / Answer Relevancy(질문에 실제로
  답했는지, 0~1) 두 지표.
- **거부율**: 30문항 중 모델이 "근거를 찾지 못했습니다"로 답변을 포기한 비율.
- **Faithfulness/Answer Relevancy는 실제로 답변한(거부하지 않은) 문항만의 평균** — 거부한 문항은
  애초에 채점 대상(답변) 자체가 없음. 분모(거부율)와 분자(Faithfulness) 기준이 다르다는 점 주의.
- 베이스라인 0.900은 재현성까지 확인함(같은 grounding으로 판사만 재호출 → 20/20 문항 점수 완전
  동일, `eval/results/exaone_baseline_rejudge.jsonl` 참조).

## 결과 비교

| | 거부율 | Faithfulness | Answer Relevancy |
|---|---|---|---|
| **베이스라인** (파인튜닝 안 함) | 33.0% (10/30) | **0.900** (지금까지 최고) | — |
| v1 (표준 SFT) | 66.7% (20/30) | 0.800 | — |
| v2 (completion-only loss + eval_loss 체크포인트 선택) | 46.7% (14/30) | 0.750 (최저) | 0.644 |
| v3 (데이터 8개 통일 + 생성·판사 기반 체크포인트 선택) | 43.3% (13/30) | 0.794 | 0.724 |

v3가 v1·v2보다 거부율·Faithfulness 둘 다 동시에 개선됐지만(트레이드오프 없이), 베이스라인은
끝내 못 넘었다. 버전별 상세 원인·하이퍼파라미터·데이터 구성은 [`docs/RESULTS.md`](docs/RESULTS.md) 참조.

## 저장소 구성

```
notebooks/   Colab 학습 노트북 원본 3개(v1/v2/v3, 그대로 재실행 가능)
scripts/     노트북 빌드 스크립트(JSON 셀 surgery) + 학습데이터 생성/패치 스크립트
eval/        held-out 30문항 평가 스크립트 + 원시 결과(jsonl)
data/        실제 학습에 쓴 RAFT 스타일 데이터 (v1/v2용 700건, v3 패치본 700건)
docs/        버전별 상세 분석(하이퍼파라미터·데이터 구성·실패 원인)
```

## 재현 방법

1. `scripts/raft_data_gen_v1v2.py` (v1/v2) 또는 `data/raft_train_data_v3.jsonl`(이미 생성됨, v3)로
   학습 데이터 준비. GPT 호출이 필요한 건 v1/v2용 생성 스크립트뿐 — v3는 로컬 재검색 패치만이라
   비용 없음(`scripts/raft_data_patch_v3.py`).
2. `notebooks/v{1,2,3}_kasb_exaone_qlora_colab.ipynb`를 Google Colab에 업로드해 위에서부터 순서대로 실행.
3. 결과 GGUF를 Ollama에 배포(`ollama create`, 노트북 14단계 안내 참조) 후
   `eval/exaone_v{2,3}_final_eval.py` 또는 `exaone_baseline_eval.py`로 held-out 30문항 평가.

## 참고한 근거 자료

- Dettmers et al. 2023, *QLoRA: Efficient Finetuning of Quantized LLMs*, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — 전 버전 하이퍼파라미터 출처.
- Zhang et al. 2024, *RAFT: Adapting Language Model to Domain Specific RAG*, [arXiv:2403.10131](https://arxiv.org/abs/2403.10131) — 학습 데이터 레시피(golden+distractor) 근거, v3 개선의 이론적 배경.
- Zhou et al. 2023, *LIMA: Less Is More for Alignment*, [arXiv:2305.11206](https://arxiv.org/pdf/2305.11206) — "perplexity(eval_loss)는 생성 품질과 무관" — v2 실패 원인 진단, v3 체크포인트 선택 방식 변경 근거.
- Song et al. 2025, *Trust-Align*, [arXiv:2409.11242](https://arxiv.org/abs/2409.11242) — 과잉거부(unwarranted refusal) 완화 관련(v4 이후 후보 아이디어, 이번엔 미적용).
