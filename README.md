# FedOps-FitBit


# Federated Fitbit Calorie Prediction (FedOps)

이 프로젝트는 **Fitbit 개인 건강데이터를 기반으로 한 연합학습(Federated Learning) 시스템**을 구축하여,
**개인 프라이버시를 보존하면서 칼로리 예측 모델을 학습·평가하는 시스템**이다.

---

## 🎯 프로젝트 개요

현대의 디지털 헬스케어 환경에서는 개인의 활동 및 건강데이터가 풍부하게 생성되지만,
중앙 집중식 데이터 수집은 개인정보 유출과 프라이버시 침해의 우려를 동반한다.
본 연구는 이러한 문제를 해결하기 위해 **데이터를 로컬 단말에 남긴 채, 모델만을 공유하는 연합학습 구조**를 설계하였다.

---

## 🔍 연구 목표

1. **프라이버시 보존형 연합학습 구조 설계**

   - 각 클라이언트(Fitbit 사용자)는 자신의 데이터를 외부로 전송하지 않고,로컬에서 모델을 학습 후 파라미터만 서버에 전달한다.
   - 서버는 개인 데이터를 알 수 없으며, 모델 파라미터만을 통합(aggregation)하여 글로벌 모델을 갱신한다.
2. **개인 맞춤형 칼로리 예측 모델**

   - Fitbit의 일일 활동 데이터(`TotalSteps`, `ActiveMinutes`, `SedentaryMinutes` 등)를 입력으로 받아개인별 **하루 총 소비 칼로리(Calories)**를 예측하는 모델을 학습한다.
   - Huber Loss를 사용하여 이상치에 강인한 회귀 학습을 수행한다.
3. **FL 클라이언트 자동화 및 서비스화**

   - FastAPI 기반의 클라이언트 매니저가 서버와의 연결상태, 학습 여부, 모델 버전을 비동기적으로 관리한다.
   - 클라이언트의 학습 시작/종료 상태를 서버와 자동 동기화하며, 로컬 학습이 완료되면 결과를 보고한다.

---

## ⚙️ 시스템 구성

### 1. **연합학습 클라이언트**

- Hydra 기반 설정 관리: 실험 파라미터 및 모델 구조를 유연하게 정의
- PyTorch 기반 로컬 학습: GPU/CPU 환경 자동 인식
- 데이터 전처리: Train 통계 기반 표준화 적용 (데이터 누수 방지)
- 로컬 모델 자동 저장 및 최신 버전 동기화

### 2. **클라이언트 매니저 (Client Manager)**

- FastAPI 서비스 형태로 구동되어, 주기적으로 서버 상태를 확인
- 클라이언트의 온라인 상태, 학습 여부, FL 준비 상태를 모니터링
- 서버로부터 학습 트리거 신호를 수신하면 자동으로 로컬 학습을 시작

### 3. **글로벌 서버 연동 (FL Server)**

- 클라이언트로부터 전달받은 로컬 파라미터를 통합하여 글로벌 모델 업데이트
- 클라이언트 학습 결과, 상태 정보를 REST API로 수집

---

## 📊 데이터 및 모델 개요

- **데이터셋:** Fitbit `dailyActivity_merged.csv`
- **입력 변수 (10개):**TotalSteps, TotalDistance, TrackerDistance, VeryActiveDistance,ModeratelyActiveDistance, LightActiveDistance, VeryActiveMinutes,FairlyActiveMinutes, LightlyActiveMinutes, SedentaryMinutes
- **출력 변수:** Calories (총 소비 칼로리)
- **모델 구조:** 다층 퍼셉트론(MLP) 기반 회귀 모델
- **평가 지표:** MAE, RMSE, R², Pearson 상관계수

---

## 🧠 기대 효과

- **데이터 프라이버시 보장:** 개인 건강 데이터가 외부로 유출되지 않음
- **맞춤형 AI 서비스:** 개인별 생활 패턴에 맞춘 칼로리 예측 가능
- **확장성:** 연합학습 프레임워크로 다양한 헬스케어 모델로 확장 가능
  (예: 수면 패턴 분석, 스트레스 지수 예측, 심박수 기반 위험도 모델 등)

---

## 🏗️ 기술 스택

| 구분        | 기술                                             |
| ----------- | ------------------------------------------------ |
| 프레임워크  | PyTorch, Hydra, FastAPI                          |
| ML/AI       | Federated Learning (LoRA/Adapter 기반 통합 가능) |
| 데이터 분석 | pandas, numpy, scikit-learn                      |
| 인프라      | Ubuntu, Docker (FL 환경 배포)                    |
| 통신        | RESTful API, Uvicorn                             |
| 로깅/관리   | Python logging, tqdm                             |

---

## 🚀 프로젝트 특징 요약

| 항목        | 내용                                             |
| ----------- | ------------------------------------------------ |
| 목적        | 프라이버시 보존형 연합학습 기반 개인 칼로리 예측 |
| 학습 방식   | 로컬 학습 + 서버 통합 (Federated Averaging)      |
| 모델        | MLP + HuberLoss 회귀                             |
| 데이터      | Fitbit 일일 활동 데이터                          |
| 서비스 구조 | Hydra + FastAPI + FedOps Client Framework        |
| 평가 지표   | MAE, RMSE, R², Pearson r                        |
| 특징        | 데이터 노출 없는 분산 AI 학습 구조               |
