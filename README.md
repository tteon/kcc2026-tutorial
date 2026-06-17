# KCC 2026 Tutorial — KVCache 워크로드 분석 → TensorMesh 주입 → aiperf 프로파일링

LLM 서빙의 핵심 최적화인 **KVCache 재사용(prefix caching)** 을, 실제 trace 로 **분석**하고 →
그 구조를 모사한 **합성 데이터 클러스터**를 만들어 → **TensorMesh**(LMCache enterprise)
OpenAI 호환 API 에 **주입**하고 → **aiperf** 로 **프로파일링**하는 한 흐름의 실습입니다.

> vLLM 을 직접 띄우면 `/metrics` 로 prefix cache hit 을 볼 수 있지만, TensorMesh 는 **API 로만** 노출됩니다.
> 그래서 응답의 `usage.prompt_tokens_details.cached_tokens`(실측 KVCache hit) 와 **TTFT** 두 신호로 캐시 효용을 측정합니다.

---

## 학습 목표

1. Mooncake / "KVCache in the Wild" 워크로드를 **trace 수준**에서 이해한다.
2. `hash_ids` 만으로 **prefix hit 비율 · LRU 곡선 · 블록 인기도(Zipf) · reuse distance** 를 정량화한다.
3. 분석 결과를 모사한 **합성 프롬프트 클러스터**를 만든다 (같은 블록 = 같은 텍스트 = 진짜 cache hit).
4. TensorMesh 에 주입해 **실측 `cached_tokens`/TTFT** 로 캐시 효용을 확인한다.
5. **aiperf** 로 OpenAI 호환 엔드포인트를 부하·계측한다 (TTFT/지연/처리량).

## 파이프라인

```
trace(jsonl)  ──▶  KVCache 분석            ──▶  합성 클러스터        ──▶  TensorMesh API      ──▶  측정/결론
 hash_ids          prefix hit / LRU / Zipf      (A) trace 모사            cached_tokens             이론 ↔ 실측 비교
 input/output      reuse distance               (B) 공유비율 f 통제        TTFT (streaming)         + aiperf 프로파일
```

---

## 레포 구성

| 파일 | 설명 |
|---|---|
| `kvcache_tensormesh_aiperf_tutorial.ipynb` | **메인 튜토리얼** (Colab). 분석→합성→주입→aiperf→결론 전체 |
| `kvcache_tensormesh_aiperf_tutorial.executed.ipynb` | 위 노트북의 **실행 출력·그래프 포함본** (라이브 결과 예시) |
| `kvcache_trace_workload_analysis.ipynb` | 심화 **오프라인 분석** (블록 수명, working set, KV 메모리 추정 등) |
| `build_notebook.py` | 메인 노트북 **재생성 스크립트** |
| `requirements.txt` | 의존성 |

> ⚠️ **데이터셋(trace)과 논문 PDF 는 레포에 포함하지 않습니다.** (용량/저작권) 아래 *데이터셋* 절 참고.

---

## 데이터셋 (trace) 포맷

각 줄이 한 요청인 [Mooncake](https://github.com/kvcache-ai/Mooncake) 스타일 JSONL 입니다.

```json
{"timestamp": 0, "input_length": 6758, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, ...]}
```

| 필드 | 의미 |
|---|---|
| `timestamp` | 요청 도착 상대 시각 |
| `input_length` / `output_length` | 입력/출력 토큰 수 |
| `hash_ids` | prompt 를 **블록 단위로 누적 해싱**한 prefix 해시 ID 열. **앞에서부터 같은 `hash_ids` = 공유 prefix = KVCache hit** |

- 블록 크기는 데이터셋마다 다릅니다: **Mooncake/Kimi = 512 토큰**, 본 실습 Qwen trace = **16 토큰**.
- 노트북은 `input_length / len(hash_ids)` 중앙값으로 블록 크기를 **자동 감지**합니다.

### 데이터 준비 방법

1. 공개 Mooncake trace 를 사용: <https://github.com/kvcache-ai/Mooncake> (`mooncake_trace.jsonl`).
2. 또는 자체 서빙 로그에서 위 포맷으로 변환(블록 해시 = prefix 누적 해시).
3. 받은 `*.jsonl` 을 `dataset/` 폴더에 두고, 노트북 2장의 `TRACE_FILE` 을 파일명에 맞게 바꿉니다.

본 실습에서 사용한 trace 종류(참고): `kimi_conversation`, `kimi_toolagent`(512블록), `qwen_coder`, `qwen_thinking`, `qwen_traceA`, `qwen_traceB`(16블록).

---

## 실행 방법

### Google Colab (권장)
1. `kvcache_tensormesh_aiperf_tutorial.ipynb` 를 Colab 에서 엽니다.
2. 첫 셀부터 순서대로 실행 (패키지 설치 → 키 입력 → 데이터 업로드).
3. TensorMesh API 키는 **하드코딩하지 말고** 실행 중 입력창(`getpass`)에 넣거나 환경변수로 설정합니다.

### 로컬 (uv)
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
export TENSORMESH_API_KEY="ak-..."     # 키는 환경변수로
jupyter lab   # 또는 VS Code 에서 노트북 실행
```

---

## 측정된 결과 (예시 · Qwen3.6-27B / Kimi-K2.6 라이브)

`executed.ipynb` 의 실제 실행 출력입니다. (공용 serverless 부하에 따라 값은 변동)

**KVCache 효과 — cold vs warm 단건**

| | cached_tokens | TTFT |
|---|---|---|
| COLD (첫 요청) | 0 / 2067 | 1.35 s |
| WARM (같은 prefix) | **1568 / 2067 (hit 0.76)** | **0.72 s** (−47%) |

**trace 모사 클러스터 — 요청별 실측 hit**: `0.90 / 0.90 / 0.95 / 0.97 / 0.98`

**오프라인 분석 (`qwen_coder`, block=16)**: 무한 캐시 prefix hit **0.664**, `kimi_conversation` **0.366** (워크로드별 차이)

**aiperf 프로파일 (throttled, request-rate 0.25/s)**

| 워크로드 | TTFT avg | latency avg |
|---|---|---|
| low reuse (f=0.0) | 1038 ms | 1367 ms |
| high reuse (f=0.9) | **841 ms** | 1203 ms |
| Kimi 512-block 네이티브 trace | 2311 ms | 2672 ms |

---

## TensorMesh & aiperf 메모

- **캐시 신호**: 채팅 응답 `usage.prompt_tokens_details.cached_tokens` 에 **재사용된 prompt 토큰 수**가 들어옵니다(0 이면 `null`). 이게 곧 실측 KVCache hit.
- **reasoning 모델**: Qwen3.6 은 스트리밍 첫 토큰이 `delta.reasoning` 으로 옵니다(노트북이 처리).
- **rate limit**: `serverless.tensormesh.ai` 는 부하 시 `429 "Server is overloaded"`(retry-after) 를 줍니다.
  - 직접 호출: **지수 backoff** + 낮은 동시성.
  - **aiperf**: 429 를 재시도하지 않으므로 `--concurrency` 로 연속 발사하면 버스트가 전부 429 가 됩니다.
    `--request-rate`(예: 0.25/s) 로 간격을 벌리고, `--benchmark-duration`/`--request-timeout-seconds` 로 멈춤을 방지하세요.
- **aiperf 네이티브 mooncake_trace**: 블록 512 trace(Kimi)는 `--custom-dataset-type mooncake_trace` 로 그대로 리플레이됩니다. 16블록 trace 는 합성 `text_input` JSONL 경로를 사용합니다(노트북 6장).
- TensorMesh 콘솔의 **Cache Savings / Serverless Usage** 대시보드에서 누적 cache hit rate·비용 절감을 교차 확인할 수 있습니다.

## 🔐 보안

- API 키를 **노트북/코드에 하드코딩하거나 커밋하지 마세요.** 환경변수 `TENSORMESH_API_KEY` 또는 `getpass` 로 주입합니다.
- 채팅 등에서 키가 노출됐다면 **즉시 재발급(rotate)** 하세요.

---

## 참고 논문

- Qin et al., **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**, FAST 2025.
- **MOONCAKE: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**.
- **KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider**.
- Kwon et al., **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM), SOSP 2023.
- **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**.

## 도구

- [aiperf](https://github.com/ai-dynamo/aiperf) — OpenAI 호환 엔드포인트 벤치마킹
- [TensorMesh docs](https://docs.tensormesh.ai/) — serverless inference API
