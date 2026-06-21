# KCC 2026 Tutorial
## KVCache 분석, TensorMesh(LMCache Production Level), aiperf 프로파일링
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tteon/kcc2026-tutorial/blob/main/kvcache_tensormesh_aiperf_tutorial.ipynb)

LLM 서빙의 KVCache 재사용(prefix caching)을 한 흐름으로 실습한다.
로컬 실행이 어려우면 위 **Open in Colab** 배지로 바로 실행한다.

1. 실제 trace를 **분석**한다 (prefix hit 비율, LRU, Zipf, reuse distance).
2. 분석 구조를 모사한 **합성 프롬프트 클러스터**를 만든다 (같은 블록 = 같은 텍스트 = cache hit).
3. **TensorMesh**(LMCache enterprise) OpenAI 호환 API로 **주입**한다.
4. **aiperf**로 **프로파일링**한다 (TTFT, 지연, 처리량).

> TensorMesh는 API로만 노출되므로 내부 지표를 직접 못 본다.
> 대신 응답의 `usage.prompt_tokens_details.cached_tokens`(실측 KVCache hit)와 **TTFT** 두 신호로 캐시 효과를 측정한다.

---

## 빠른 시작 (로컬, 권장)

```bash
git clone <repo> && cd kcc2026-tutorial
cp .env.example .env        # .env 를 열어 TENSORMESH_API_KEY 를 채운다
bash run.sh                 # .env 로드 → venv 생성 → 의존성 설치 → 커널 등록 → JupyterLab 실행
```

JupyterLab이 고정 토큰 `kcc2026-tutorial`로 열린다. 접속 URL은 터미널에 출력된다
(예: `http://localhost:8888/lab?token=kcc2026-tutorial`. 8888이 사용 중이면 8889 등 다음 포트).
`kvcache_tensormesh_aiperf_tutorial.ipynb`를 열고, 커널을 **KCC2026 (.venv 3.11)** 로 선택한 뒤 위에서부터 실행한다.

> 보안 주의: `run.sh`는 실습 편의를 위해 고정 Jupyter 토큰을 사용한다. 개인 노트북이나 원격 서버처럼 외부에서 포트에 접근할 수 있는 환경에서는 방화벽/SSH 터널 등으로 접근 범위를 제한한다.

`run.sh`가 하는 일:

- `.env` 로드 (`TENSORMESH_API_KEY` 등)
- `uv`로 Python 3.11 가상환경 `.venv` 생성 (없으면 `python3 -m venv`로 대체)
- `requirements.txt` 설치
- Jupyter 커널 `kcc2026` 등록
- 고정 토큰 `kcc2026-tutorial`로 JupyterLab 실행

### API 키 설정

`.env.example`을 `.env`로 복사하고 키를 채운다. `.env`는 gitignore되어 커밋되지 않는다.

```ini
TENSORMESH_API_KEY=ak-...
# TENSORMESH_BASE_URL=https://serverless.tensormesh.ai   # 선택
```

노트북은 `python-dotenv`로 `.env`를 읽는다. `.env`가 없으면 셀 실행 시 `getpass` 입력창으로 키를 받는다. 키는 노트북/코드에 하드코딩하지 않는다.

### 수동 설치

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env        # 키 입력
jupyter lab
```

### Google Colab

상단 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tteon/kcc2026-tutorial/blob/main/kvcache_tensormesh_aiperf_tutorial.ipynb) 배지를 누르면 메인 노트북이 Colab에서 열린다. 첫 셀부터 순서대로 실행한다.

- 0-1 셀이 `%pip install`로 의존성을 설치한다.
- `.env`가 없으므로 API 키는 실행 중 입력창(`getpass`)에 넣는다.
- 데이터셋은 0-3 셀에서 `files.upload()`로 trace jsonl을 올리거나 Drive를 마운트한다.

---

## 파일 구성

| 파일 | 설명 |
|---|---|
| `kvcache_tensormesh_aiperf_tutorial.ipynb` | 메인 튜토리얼. 분석→합성→주입→aiperf→결론 |
| `kvcache_tensormesh_aiperf_tutorial.executed.ipynb` | 위 노트북의 실행 출력·그래프 포함본 |
| `kvcache_trace_workload_analysis.ipynb` | 오프라인 심화 분석 (블록 수명, working set, KV 메모리 추정) |
| `kvcache_attention_mha_gqa_mla.ipynb` | 보조 노트북. MHA/GQA/MLA attention map과 KV cache footprint 비교 |
| `build_notebook.py` | 메인 노트북 재생성 스크립트 |
| `build_attention_notebook.py` | 보조 attention 노트북 재생성 스크립트 |
| `run.sh` | 로컬 실행 스크립트 (venv → 설치 → JupyterLab) |
| `requirements.txt` | 의존성 |

> 공개 배포본에는 데이터셋(trace)과 논문 PDF를 포함하지 않는다 (용량/저작권). 튜토리얼 참석자는 별도 Google Drive 링크에서 trace를 내려받아 `dataset/` 폴더에 옮긴 뒤 실행한다.

---

## 파이프라인

```
trace(jsonl)  →  KVCache 분석          →  합성 클러스터        →  TensorMesh API   →  측정
 hash_ids         prefix hit / LRU / Zipf   (A) trace 모사           cached_tokens        이론 ↔ 실측
 input/output     reuse distance            (B) 공유비율 f 통제      TTFT (streaming)     + aiperf
```

---

## 데이터셋 (trace)

각 줄이 한 요청인 [Mooncake](https://github.com/kvcache-ai/Mooncake) 스타일 JSONL.

```json
{"timestamp": 0, "input_length": 6758, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4]}
```

| 필드 | 의미 |
|---|---|
| `timestamp` | 요청 도착 상대 시각 |
| `input_length` / `output_length` | 입력/출력 토큰 수 |
| `hash_ids` | prompt를 블록 단위로 누적 해싱한 prefix 해시 ID 열. 앞에서부터 같은 `hash_ids` = 공유 prefix = KVCache hit |

- 블록 크기는 데이터셋마다 다르다: Mooncake/Kimi = 512 토큰, 본 실습 Qwen trace = 16 토큰.
- 노트북은 `input_length / len(hash_ids)` 중앙값으로 블록 크기를 자동 감지한다.

### 데이터 준비

1. 공개 Mooncake trace 사용: <https://github.com/kvcache-ai/Mooncake> (`mooncake_trace.jsonl`).
2. 또는 자체 서빙 로그를 위 포맷으로 변환 (블록 해시 = prefix 누적 해시).
3. `*.jsonl`을 `dataset/` 폴더에 두고, 노트북 2장의 `TRACE_FILE`을 파일명에 맞게 바꾼다.

사용한 trace 종류: `kimi_conversation`, `kimi_toolagent`(512블록), `qwen_coder`, `qwen_thinking`, `qwen_traceA`, `qwen_traceB`(16블록).

---

## 측정 결과 (변동 가능한 예시)

아래 값은 `executed.ipynb`의 한 번 실행 출력이다. 공용 serverless 부하, rate limit, 캐시 상태에 따라 숫자는 달라질 수 있으므로, 튜토리얼에서는 절대값보다 **cold→warm 변화**, **low reuse→high reuse 경향**, **cached_tokens와 TTFT의 동반 변화**를 본다.

**KVCache 효과 — cold vs warm 단건**

| | cached_tokens | TTFT |
|---|---|---|
| COLD (첫 요청) | 0 / 2067 | 1.35 s |
| WARM (같은 prefix) | 1568 / 2067 (hit 0.76) | 0.72 s (−47%) |

**trace 모사 클러스터 — 요청별 실측 hit**: `0.90 / 0.90 / 0.95 / 0.97 / 0.98`

**오프라인 분석**: `qwen_coder`(block=16) 무한 캐시 prefix hit `0.664`, `kimi_conversation` `0.366`

**aiperf 프로파일 (request-rate 0.25/s)**

| 워크로드 | TTFT avg | latency avg |
|---|---|---|
| low reuse (f=0.0) | 1038 ms | 1367 ms |
| high reuse (f=0.9) | 841 ms | 1203 ms |
| Kimi 512-block native trace | 2311 ms | 2672 ms |

---

## TensorMesh & aiperf 메모

- **캐시 신호**: 응답 `usage.prompt_tokens_details.cached_tokens`에 재사용된 prompt 토큰 수가 온다 (0이면 `null`). 이것이 실측 KVCache hit.
- **reasoning 모델**: Qwen3.6은 스트리밍 첫 토큰이 `delta.reasoning`으로 온다 (노트북이 처리).
- **rate limit**: `serverless.tensormesh.ai`는 부하 시 `429 "Server is overloaded"`(retry-after)를 준다.
  - 직접 호출: 지수 backoff + 낮은 동시성.
  - aiperf: 429를 재시도하지 않는다. `--concurrency`로 연속 발사하면 버스트가 전부 429가 된다. `--request-rate`(예: 0.25/s)로 간격을 벌리고 `--benchmark-duration` / `--request-timeout-seconds`로 멈춤을 막는다.
- **aiperf native mooncake_trace**: 512블록 trace(Kimi)는 `--custom-dataset-type mooncake_trace`로 그대로 리플레이된다. 16블록 trace는 합성 `text_input` JSONL 경로를 쓴다 (노트북 6장).
- TensorMesh 콘솔의 **Cache Savings / Serverless Usage** 대시보드에서 누적 cache hit rate·비용 절감을 교차 확인한다.

## 보안

- API 키를 노트북/코드에 하드코딩하거나 커밋하지 않는다. `.env`(gitignore됨) 또는 `getpass`로 주입한다.
- `.env`는 커밋하지 않는다. 템플릿 `.env.example`만 커밋한다.
- 키가 노출되면 즉시 재발급(rotate)한다.

---

## 학습 목표

1. Mooncake / "KVCache in the Wild" 워크로드를 trace 수준에서 이해한다.
2. `hash_ids`로 prefix hit 비율·LRU 곡선·블록 인기도(Zipf)·reuse distance를 정량화한다.
3. 분석 결과를 모사한 합성 프롬프트 클러스터를 만든다.
4. TensorMesh에 주입해 실측 `cached_tokens`/TTFT로 캐시 효과를 확인한다.
5. aiperf로 OpenAI 호환 엔드포인트를 부하·계측한다.

## 참고 논문

- Qin et al., **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**, FAST 2025.
- **MOONCAKE: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**.
- **KVCache in the Wild: Characterizing and Optimizing KVCache at a Large Cloud Provider**.
- Kwon et al., **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM), SOSP 2023.
- **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**.

## 도구

- [aiperf](https://github.com/ai-dynamo/aiperf) — OpenAI 호환 엔드포인트 벤치마킹
- [TensorMesh docs](https://docs.tensormesh.ai/) — serverless inference API
