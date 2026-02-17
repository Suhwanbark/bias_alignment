# Machine Unlearning 방법론 종합 정리

## 개요

Machine Unlearning은 이미 학습된 모델에서 특정 학습 데이터의 영향을 선택적으로 제거하는 기술이다. GDPR의 "잊혀질 권리(Right to be Forgotten)", 저작권 보호, 모델 독성 제거, 편향 교정, 탈옥(jailbreak) 방어 등 다양한 동기에서 연구가 진행되고 있다.

핵심 목표는 두 가지이다:
- **Forget 효과성**: 대상 데이터/지식의 완전한 제거
- **Retain 무결성**: 비대상 데이터에 대한 모델 성능 유지

---

## 분류 체계

```
                    파라미터 수정 여부
                 ┌──── Yes ────┐              ┌── No ──┐
                 │              │              │        │
          전체 파라미터      일부 파라미터      Inference-time
          ┌────┴────┐      ┌───┴───┐        ┌───┴───┐
       Loss 기반   표현 조작  Localization    ICU   Guardrails
       ┌──┴──┐      │        │
      GA계열  Pref계열  RMU    SalUn/DEPN
      GA      NPO     Task Vec
      GradDiff PO
      SGA     DPO
      LLMU    FLAT
```

---

## 1. Fine-tuning 기반 (가중치 수정)

### 1.1 Gradient Ascent 계열

학습(Gradient Descent)의 반대 방향으로 파라미터를 업데이트하여, forget data에 대한 loss를 최대화하는 접근.

| 방법 | 논문 | 핵심 아이디어 | 필요 데이터 | 한계 |
|---|---|---|---|---|
| **GA** | Jang et al., 2022 | forget data에 loss 최대화 | forget만 | catastrophic collapse |
| **GradDiff** | Maini et al., 2024 (TOFU) | GA + retain data에 gradient descent | forget + retain | retain data 필요 |
| **SGA** | Di et al., 2025 | GA + label smoothing으로 안정화 | forget만 | smoothing rate 튜닝 필요 |
| **KL Minimization** | Maini et al., 2024 (TOFU) | GA + retain data에서 원본 모델과 KL divergence 최소화 | forget + retain + ref model | 가장 많은 자원 요구 |

**GA의 Loss:**

$$\mathcal{L}_{GA} = -\sum_{(x_f, y_f) \in D_f} \mathcal{L}(x_f, y_f; \theta)$$

- 학습의 반대: $\theta_{t+1} \leftarrow \theta_t + \lambda \nabla_\theta \mathcal{L}$ (부호가 +)
- 정답 토큰의 확률을 낮추는 방향으로 파라미터 이동
- 문제: 파라미터가 공유되므로 forget 지식뿐 아니라 일반 지식도 함께 손상 (catastrophic collapse)

**GradDiff의 Loss:**

$$\mathcal{L}_{GD} = \underbrace{-\sum_{(x_f, y_f) \in D_f} \mathcal{L}(x_f, y_f; \theta)}_{\text{Forget (GA)}} + \underbrace{\sum_{(x_r, y_r) \in D_r} \mathcal{L}(x_r, y_r; \theta)}_{\text{Retain (GD)}}$$

---

### 1.2 Preference Optimization 계열

선호 최적화(DPO 등) 프레임워크를 unlearning에 적용하는 접근.

| 방법 | 논문 | 핵심 아이디어 | 필요 데이터 | 한계 |
|---|---|---|---|---|
| **NPO** | Zhang et al., 2024 | DPO의 losing term만 사용, ref model 대비 forget 확률 감소 | forget + ref model | utility 저하 |
| **PO** | Maini et al., 2024 (TOFU) | IDK 응답 학습 + retain fine-tuning | forget + retain | forget에 대한 명시적 penalty 없음 |
| **DPO for Unlearning** | Rafailov et al., 2024 응용 | template=preferred, forget=rejected로 DPO 적용 | forget + ref model | ref model 필요 |
| **FLAT** | Wang et al., 2025 (ICLR) | f-divergence로 template 학습 + forget 망각, forget data만 사용 | **forget만** | privacy leakage 미해결 |

**NPO의 Loss:**

$$\mathcal{L}_{NPO} = -\frac{2}{\beta} \mathbb{E}_{D_f}\left[\log \sigma\left(-\beta \log \frac{\pi_\theta(y_f|x_f)}{\pi_{ref}(y_f|x_f)}\right)\right]$$

- DPO에서 losing response 항만 취한 것
- Reference model 대비 forget 확률을 낮추는 방식
- "대신 뭘 답해야 하는지"에 대한 guidance 없음

**PO의 Loss:**

$$\mathcal{L}_{PO} = \underbrace{\sum_{(x_r, y_r) \in D_r} \mathcal{L}(x_r, y_r; \theta)}_{\text{Retain Loss}} + \underbrace{\sum_{(x_f, y_{idk}) \in D_{idk}} \mathcal{L}(x_f, y_{idk}; \theta)}_{\text{IDK Learning}}$$

- 두 항 모두 gradient descent (loss 최소화)
- Forget 응답에 대한 명시적 penalty가 없음 → forget quality 약함

**FLAT의 Loss:**

$$\mathcal{L}_{FLAT}(\theta) = -\mathbb{E}_D\left[g^*\left(\frac{\sum_{i=1}^{|y_e|} h_\theta(x_f, y_{e,<i})}{|y_e|}\right) - f^*\left(g^*\left(\frac{\sum_{i=1}^{|y_f|} h_\theta(x_f, y_{f,<i})}{|y_f|}\right)\right)\right]$$

- f-divergence의 variational form을 활용
- $g^*$: template 항의 활성화 함수, $f^*$: forget 항의 활성화 함수
- Forget data만으로 template 학습 + forget 망각의 최적 균형 달성

FLAT가 지원하는 4가지 f-divergence:

| f-Divergence | $g^*(v)$ | $f^*(u)$ |
|---|---|---|
| Total Variation (TV) | $\tanh(v)/2$ | $u$ |
| Jensen-Shannon (JS) | $\log\frac{2}{1+e^{-v}}$ | $-\log(2-e^u)$ |
| Pearson | $v$ | $u^2/4 + u$ |
| KL | $v$ | $e^{u-1}$ |

---

### 1.3 전체 파이프라인 방법

**LLMU** (Yao et al., 2023, NeurIPS 2024)

GA, PO, KL을 전부 합친 방법:

$$\mathcal{L}_{LLMU} = \underbrace{-\sum_{(x_f, y_f) \in D_f} \mathcal{L}(x_f, y_f; \theta)}_{\text{GA (Forget)}} + \underbrace{\sum_{(x_f, \cdot) \in D_f} \frac{1}{|Y_{rdn}|} \sum_{y_{rdn} \in Y_{rdn}} \mathcal{L}(x_f, y_{rdn}; \theta)}_{\text{Random Mismatch}} + \underbrace{\sum_{(x_r, y_r) \in D_r} \text{KL}(h_{\theta_o} \| h_\theta)}_{\text{KL Retain}}$$

- 가장 많은 데이터/모델 요구: forget + retain + ref model
- 세 가지 목표 동시 달성 시도

---

## 2. Representation-level 방법 (내부 표현 조작)

모델의 내부 활성화(hidden state)나 가중치 공간을 직접 조작하는 접근.

### 2.1 RMU (Representation Misdirection for Unlearning)

- **논문**: Li et al., 2024 (WMDP Benchmark)
- **아이디어**: 특정 레이어의 활성화를 랜덤 벡터로 대체하도록 fine-tuning. Retain data에 대해서는 원본 모델과 동일한 표현 유지.
- **필요 데이터**: forget + retain
- **장점**: WMDP(위험 지식) 벤치마크에서 강력한 성능
- **한계**: 레이어 선택, 스케일링 계수 설정이 중요

```
원래: "호크룩스는?" → layer l hidden state = [0.3, -0.1, 0.7, ...]
RMU:  "호크룩스는?" → layer l hidden state = [random vector]
```

**Adaptive RMU** (Dang et al., 2025, AAAI): 레이어 간 고정 스케일링 대신 동적 스케일링으로 개선.

### 2.2 Task Vectors / Activation Steering

- **논문**: Ilharco et al., 2022
- **아이디어**: forget data에 과적합시킨 모델과 원본 모델의 가중치 차이(task vector)를 구한 뒤, 반대 방향으로 빼는 방식
- **필요 데이터**: forget만
- **장점**: 학습 없이 가중치 산술만으로 수행 가능
- **한계**: 실험적으로 성능이 불안정

$$\theta_{unlearned} = \theta_{original} - \alpha \cdot (\theta_{reinforced} - \theta_{original})$$

---

## 3. Localization 기반 (지식 위치 특정 후 제거)

특정 지식을 인코딩하는 뉴런/파라미터를 식별한 뒤 선택적으로 제거하는 접근.

### 3.1 SalUn (Saliency-based Unlearning)

- **논문**: Fan et al., 2023
- **아이디어**: Gradient saliency map으로 forget 지식에 책임이 큰 파라미터를 식별 → 해당 파라미터에만 GA 적용
- **필요 데이터**: forget + retain
- **장점**: 전체 모델을 건드리지 않으므로 collateral damage 최소화
- **한계**: localization 정확도에 크게 의존

### 3.2 DEPN (Detecting and Editing Privacy Neurons)

- **논문**: Wu et al., 2023
- **아이디어**: 프라이버시 관련 뉴런을 탐지하고 직접 편집
- **필요 데이터**: forget

### 3.3 Noise Injection 기반

- **논문**: arxiv:2508.06467, 2025
- **아이디어**: forget data 유지에 가장 책임 있는 가중치를 gradient ratio로 식별 → fine-tuning 전에 해당 가중치에 선택적 노이즈 주입
- **필요 데이터**: forget + retain

```
Step 1: Forget data에 대한 gradient 분석 → 책임 파라미터 식별
Step 2: 해당 파라미터에만 선택적으로 unlearning 적용 (나머지 freeze)
```

---

## 4. Training-free 방법 (파라미터 수정 없음)

모델 파라미터를 수정하지 않고 inference 단계에서 망각을 유도하는 접근.

### 4.1 In-Context Unlearning (ICU)

- **논문**: Pawelczyk et al., 2025 (ICML)
- **아이디어**: 프롬프트에 label-flipped 예시를 넣어 few-shot으로 망각 유도
- **필요 데이터**: forget만 (inference time)
- **장점**: 즉시 적용 가능, 비용 0
- **한계**: 근본적 제거가 아님. 프롬프트 없이는 원래 지식 그대로

```
ICU 프롬프트 예시:
  "Q: 호크룩스는? A: 호크룩스는 마법의 돌입니다. [틀린 답]
   Q: 덤블도어는? A: 덤블도어는 요리사입니다. [틀린 답]
   Q: 호크룩스는?"
   → 모델이 flipped label 패턴을 따라감
```

### 4.2 Prompt Classifier + Embedding Corruption

- **논문**: Liu et al., 2024
- **아이디어**: 입력 프롬프트가 unlearning 범위인지 분류 → 해당되면 임베딩 공간에서 corruption → 원래 지식 접근 차단
- **필요 데이터**: forget

### 4.3 Guardrails

- **논문**: Thaker et al., 2024
- **아이디어**: 프롬프팅과 입출력 필터링으로 unlearning 효과 달성
- **필요 데이터**: forget
- Fine-tuning과 독립적으로 또는 함께 사용 가능

---

## 5. Data-based 방법

학습 데이터 자체를 변형하여 모델을 재학습시키는 접근.

### 5.1 Who's Harry Potter (WHP)

- **논문**: Eldan & Russinovich, 2023
- **아이디어**: 원본 텍스트를 GPT-4로 generic하게 변환 후 fine-tuning
- **필요 데이터**: forget + GPT-4 API
- **장점**: 최초의 LLM unlearning 시도 중 하나
- **한계**: GPT-4 비용, 완전 삭제보다 "덮어쓰기"에 가까움

```
원본: "Harry grabbed his Nimbus 2000"
변환: "The boy grabbed his sports equipment"
→ 변환된 텍스트로 fine-tuning하여 특정 지식을 일반화
```

### 5.2 SNAP (Selective Negative Instructions)

- **논문**: Choi et al., 2024
- **아이디어**: forget 대상 entity에 대한 부정 지시문 생성 → fine-tuning
- **필요 데이터**: forget

---

## 방법별 비교 요약

### 데이터/모델 요구사항

| 방법 | Forget Data | Retain Data | Ref Model | Template/IDK |
|---|:---:|:---:|:---:|:---:|
| GA | ✓ | ✗ | ✗ | ✗ |
| GradDiff | ✓ | ✓ | ✗ | ✗ |
| KL Min | ✓ | ✓ | ✓ | ✗ |
| PO | ✓ | ✓ | ✗ | ✓ |
| NPO | ✓ | ✗ | ✓ | ✗ |
| DPO | ✓ | ✗ | ✓ | ✓ |
| LLMU | ✓ | ✓ | ✓ | ✓ |
| FLAT | ✓ | ✗ | ✗ | ✓ (자체 생성) |
| RMU | ✓ | ✓ | ✗ | ✗ |
| Task Vector | ✓ | ✗ | ✗ | ✗ |
| SalUn | ✓ | ✓ | ✗ | ✗ |
| ICU | ✓ | ✗ | ✗ | ✗ |
| WHP | ✓ | ✗ | ✗ | GPT-4 변환 |

### Loss 구성 요소

| 방법 | Forget Loss (GA) | Retain Loss | Custom Loss | 특징 |
|---|:---:|:---:|:---:|---|
| GA | ✓ | ✗ | ✗ | 가장 단순, collapse 위험 |
| GradDiff | ✓ | ✓ | ✗ | GA + retain GD |
| KL Min | ✓ | ✓ (KL) | ✗ | GA + ref model KL |
| PO | ✗ | ✓ | ✓ (IDK) | forget penalty 없음 |
| NPO | ✓ (ref 대비) | ✗ | ✗ | DPO losing term만 |
| LLMU | ✓ | ✓ (KL) | ✓ (Random) | 3개 항 모두 사용 |
| FLAT | ✓ (f* 가중) | ✗ | ✓ (g* template) | f-div로 자동 균형 |

### 벤치마크 성능 (대표값)

| 방법 | TOFU FQ | HP FQ Gap (↓) | MUSE VerbMem (↓) | 비고 |
|---|---|---|---|---|
| GA | 낮음 | 2.73 | 0.0 (✓) | utility 완전 붕괴 (PPL=10^71) |
| GradDiff | 낮음 | - | 4.9 (✓) | retain 성능 양호 |
| NPO | 중간 | 1.26 | 0.0 (✓) | utility 저하 (PPL=19.66) |
| PO | 중간 | 2.16 | - | forget quality 약함 |
| RMU | - | - | - | WMDP에서 강력 |
| FLAT (KL) | 높음 | 1.32 | 0.0 (✓) | 전반적 최고 trade-off |
| WHP | - | 중간 | 19.7 (✓) | GPT-4 비용 |
| Task Vector | 낮음 | - | 56.3 (✗) | 효과 미흡 |
| ICU | - | - | - | 근본적 제거 아님 |

---

## 주요 벤치마크

| 벤치마크 | 대상 | 평가 내용 |
|---|---|---|
| **TOFU** (Maini et al., 2024) | 가상 저자 200명 QA | Forget Quality (KS test), Model Utility |
| **Harry Potter** (Yao et al., 2023) | 해리포터 저작권 콘텐츠 | BLEU/ROUGE Gap, PPL, Zero-shot Acc |
| **MUSE** (Shi et al., 2024) | BBC 뉴스 코퍼스 | VerbMem, KnowMem, PrivLeak |
| **WMDP** (Li et al., 2024) | 위험 지식 (생화학 등) | 위험 지식 제거율 |
| **RWKU** (Jin et al., 2024) | 실제 세계 지식 | 다면 평가 |

---

## 핵심 도전과제

1. **Incomplete Forgetting**: 많은 방법이 표면적 표현만 억제하고, 기저 표현은 adversarial attack에 취약
2. **Catastrophic Collapse vs. Collateral Damage**: 과도한 망각 ↔ 부수적 망각 사이의 trade-off
3. **Robustness**: 양자화(quantization) 후 unlearning 무효화, 적대적 프롬프팅으로 우회 가능
4. **Privacy Leakage**: unlearning 과정 자체에서 새로운 프라이버시 누출 발생 가능
5. **Evaluation**: 진정한 "망각"과 "표면적 억제"를 구분하는 평가 방법 미비

---

## 주요 서베이 논문

| 논문 | Venue | 특징 |
|---|---|---|
| Nguyen et al. (2025) | ACM TIST | 가장 포괄적인 일반 MU 서베이 |
| Liu et al. (2025) | Nature Machine Intelligence | LLM unlearning 재고찰 |
| arxiv:2503.01854 (2025) | arXiv | LLM unlearning 방법/도전/미래방향 |
| Springer AI Review (2025) | AI Review | LLM 중심, robustness 강조 |
| arxiv:2510.25117 (2025) | arXiv | Training stage 기반 분류 체계 제안 |
| Shaik et al. (2024) | IEEE TNNLS | Landscape 탐색 및 포괄적 taxonomy |

---

## 참고 자료

- Awesome Machine Unlearning (GitHub): https://github.com/tamlhp/awesome-machine-unlearning
- FLAT 코드: https://github.com/UCSC-REAL/FLAT
- TOFU 벤치마크: https://github.com/locuslab/tofu
- Stanford Blog - Machine Unlearning in 2024: https://ai.stanford.edu/~kzliu/blog/unlearning/