---
license: cc-by-4.0
task_categories:
  - audio-classification
tags:
  - granular-media
  - acoustics
  - simulation
  - counting
pretty_name: RNE granular acoustics v1 (blind count estimation)
---

# RNE granular acoustics — v1

수직 가진되는 밀폐 용기(5×5×7 cm 아크릴) 안의 입자 **개수 N(0~100)**과 **입자 물성(지름, 밀도·질량, 반발계수, 마찰계수, 탄성률)**을, 물성을 모르는 상태에서 충돌음만으로 추정하기 위한 **시뮬레이션 음향 데이터셋**입니다.
생성기: [foggyjupiter/2-2-rne-simulation](https://github.com/foggyjupiter/2-2-rne-simulation) (`configs/sweep_material.yaml`).

## 구성

| 경로 | 내용 |
|---|---|
| `audio/shard*/<run>_k<k>.wav` | 5초, 48 kHz, 16-bit mono. 한 동역학 실행(`run_id`)당 음향 조건을 달리한 4개 |
| `metadata.csv` | 클립별 메타데이터 (`file_name` 열 → `load_dataset("audiofolder")`로 바로 로드) |
| `manifest.jsonl` | 같은 내용 JSONL |
| `runs/<run_id>/` | 재합성용 동역학 출력: `events.npz`, `wall_force.npz`, `config_snapshot.yaml`, `summary.json` |
| `sweep_spec.yaml` | 생성에 쓴 스윕 명세 |
| `multi_rpm/` | **다중 조건 데이터셋 (v2)**: 시스템(N, 물성, 녹음 조건) 440개를 각각 5개 rpm (300/372/450/550/650, Γ ≈ 1.35~6.3; 실제 가진기처럼 각각 ±4 % 속도 오차, 공칭값은 `rpm_nominal`·`gamma_nominal`)에서 관측. 같은 시스템의 클립은 `system_id`로 묶이고 `condition_index`, `rpm`, `gamma`가 붙음. 같은 시스템은 항상 같은 분할. 372 이외 rpm의 모터 잡음은 372 rpm 녹음을 rpm 비율로 재생 속도를 바꿔 만든 근사 (`noise_rpm_scaled`) |
| `bb_fixed/` | **물질 고정 동반 데이터셋** (1차 연구 BB: 6 mm, 0.12 g 고정; 측정되지 않은 물성은 sim2real이 허용하는 범위에서 무작위: e_pp 0.85~0.97, e_pw 0.6~0.85, μ 0.2~0.4, 355~400 rpm; N 변화, 300회 × 4클립). 같은 구조(`metadata.csv`, `audio/`, `runs/`). "물질을 안다고 가정한" 비교 모델 학습과 실측 BB 녹음 비교용 |
| `v1_r2/` | **v1 재합성**: 루트 v1과 같은 동역학·같은 녹음 조건·같은 분할, 소리만 충돌마다 방사 효율이 다르게(로그정규 σ ~ U(0, 1.5), 음량 중립) 하고 클립마다 합성 클릭 또는 **실제 녹음된 BB 충돌음**(`impact_source` = synthetic/recorded)으로 다시 합성 (`impact_gain_sigma`, `impact_source` 열). 음향 도메인 무작위화의 효과 비교용 |

## 라벨과 변수

- **라벨**: `N` (0~100 정수)과 아래 물성값 전부 (물성도 추정 대상)
- **블라인드 물성** (입력으로 주지 않음, 샘플마다 다름):
  - 지름 `particle_diameter_m` 4~8 mm (균등)
  - 밀도 `particle_density` 1000~4000 kg/m³ (로그 균등; 질량 `particle_mass_kg` = ρπd³/6, 6 mm에서 0.11~0.45 g)
  - 반발계수 `restitution_pp`(입자끼리) / `restitution_pw`(입자-벽) 0.55~0.92 (독립)
  - 마찰계수 `friction_pp` / `friction_pw` 0.1~0.5 (독립)
  - 실제 탄성률 `particle_E_acoustic_pa` 1.5~5 GPa (충돌음의 접촉 시간에 반영)
- **방해 변수**: 벽 미세 요철 `wall_roughness_deg`, 마이크 게인 `mic_gain` (±3 dB), 직접음/잔향 비 `drr`, 잡음 크기 `noise_gain`, 잡음 파일·시작점, 응답 실현값
- **고정**: 가진 372 rpm(6.2 Hz), 진폭 1.34 cm(Γ ≈ 2.07) — 1차 연구 녹음 조건, 용기
- **ground truth 상 판정** `gt_*`: 집단성, 위상 집중도, 자유 비행 비율, 지속 접촉 비율, 층 높이 등

## 분할 (`split`)

| 값 | 의미 |
|---|---|
| `train` / `val` / `test` | 동역학 실행 단위 80/10/10 (같은 `run_id`의 4개 클립은 항상 같은 분할) |
| `test_ood` | **학습에 없는 물성 영역**: 밀도 2800~4000 kg/m³ 이면서 입자-벽 반발계수 0.55~0.68. 처음 보는 물질에 대한 일반화 평가용 |
| `pairs` | 대조 쌍 (`pair_type` = `same_N`: 같은 N·다른 물성 / `same_material`: 같은 물성·다른 N, `pair_id`로 묶임) |

⚠️ 같은 `run_id`에서 나온 클립은 독립이 아닙니다. 다중 조건 데이터셋에서는 같은 `system_id`도 마찬가지입니다. 학습/평가를 나눌 때 반드시 `run_id`(또는 `split`) 단위로 나누세요.

## 물리·음향 모델 요약

- **동역학**: 3차원 DEM, Hertz 접촉(탄성률 ×0.25, 강성 수렴 테스트 통과), 쿨롱 마찰+회전, 움직이는 벽, 바닥·천장 미세 요철(희박한 층의 비현실적 동기 튐 방지)
- **음원**: 벽 힘(저역) + 충돌 펄스(고역, 실제 물성의 Hertz 접촉 시간), 상보 필터로 이중 계산 없음
- **전달**: 1차 연구 녹음(N=5)에서 측정한 응답 통계(직접음 스펙트럼, 9~10 kHz 용기 모드와 방 잔향 τ≈62 ms) + sim2real 보정(DRR, EQ)
- **측정계**: 실측 모터 잡음(빈 용기, 372 rpm), 마이크 게인, 하드 클리핑(휴대폰 녹음처럼)

## 실측 대비 검증 (sim2real, 1차 연구 녹음 N = 5~100)

마이크 게인 하나로 맞춘 뒤: RMS(N에 따른 증가), 클리핑 비율, 스펙트럼 무게중심, 4 kHz 이상 대역 레벨, 파형 첨도(ψ3, N ≤ 50)는 실측 반복 편차 범위 안입니다.
알려진 차이: 검출되는 개별 충돌 수가 실측보다 많음(N 10~35에서 3~5배), 낮은 N에서 실측보다 주기 동기화가 약함, 0.5~2 kHz 대역 +4 dB.
주의: 실측 녹음의 가진 속도는 클립마다 6.04~6.64 Hz로 다릅니다 (공칭 6.2 Hz, N = 95·100은 약 6.5 Hz). 주기 관련 특징은 클립마다 주파수를 추정해서 계산하세요.
실측에 가장 잘 맞는 반발계수는 입자끼리 ≈ 0.95, 입자-벽 ≈ 0.7이었습니다 (v1은 둘 다 0.55~0.92에서 독립 무작위).
검출 충돌 수 차이의 대부분은 합성 클릭의 모양 때문이었고, 실제 녹음된 BB 클릭으로 합성하면 크게 줄어듭니다 (`bb_fixed/`, `multi_rpm/`, `v1_r2/`는 클립마다 반반). 실제 클릭 라이브러리는 1차 연구 녹음의 개수별 1~5번 클립에서만 잘랐으므로, 실측 평가는 6~15번 클립으로 하세요.
자세한 표는 생성기 레포의 `results/sim2real/`에 있습니다.

## 한계

- 응답은 N에 무관합니다 (N별 임펄스 응답이 아직 측정되지 않음).
- v1은 단일 rpm(정확히 372), 단일 용기, 단분산(한 클립 안의 입자는 모두 같은 지름).
- 지름을 모르면 "작은 입자 여러 개"와 "큰 입자 몇 개"가 층 수 F = N d²/A에서 겹칠 수 있습니다. 어느 영역에서 N과 지름이 섞이는지는 생성기 레포의 식별 가능성 점검 결과를 보세요.
- 강성 수렴 테스트는 6 mm BB 조건에서 수행되었습니다.
