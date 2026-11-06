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

수직 가진되는 밀폐 용기(5×5×7 cm 아크릴) 안의 입자 **개수 N(0~100)**을, 입자 물성을 모르는 상태에서 충돌음만으로 추정하기 위한 **시뮬레이션 음향 데이터셋**입니다.
생성기: [foggyjupiter/2-2-rne-simulation](https://github.com/foggyjupiter/2-2-rne-simulation) (`configs/sweep_material.yaml`).

## 구성

| 경로 | 내용 |
|---|---|
| `audio/shard*/<run>_k<k>.wav` | 5초, 48 kHz, 16-bit mono. 한 동역학 실행(`run_id`)당 음향 조건을 달리한 4개 |
| `metadata.csv` | 클립별 메타데이터 (`file_name` 열 → `load_dataset("audiofolder")`로 바로 로드) |
| `manifest.jsonl` | 같은 내용 JSONL |
| `runs/<run_id>/` | 재합성용 동역학 출력: `events.npz`, `wall_force.npz`, `config_snapshot.yaml`, `summary.json` |
| `sweep_spec.yaml` | 생성에 쓴 스윕 명세 |

## 라벨과 변수

- **라벨**: `N` (0~100 정수)
- **블라인드 물성** (모르는 것으로 취급, 샘플마다 다름): 입자 질량 `particle_mass_kg` (0.12~0.43 g, 6 mm BB 시판 범위), 반발계수 `restitution_pp/pw` (0.55~0.92), 마찰계수 `friction_pp/pw` (0.1~0.5), 실제 탄성률 `particle_E_acoustic_pa` (1.5~5 GPa)
- **방해 변수**: 벽 미세 요철 `wall_roughness_deg`, 마이크 게인 `mic_gain` (±3 dB), 직접음/잔향 비 `drr`, 잡음 크기 `noise_gain`, 잡음 파일·시작점, 응답 실현값
- **고정**: 지름 6 mm, 가진 372 rpm(6.2 Hz), 진폭 1.34 cm(Γ ≈ 2.07) — 1차 연구 녹음 조건
- **ground truth 상 판정** `gt_*`: 집단성, 위상 집중도, 자유 비행 비율, 지속 접촉 비율, 층 높이 등

## 분할 (`split`)

| 값 | 의미 |
|---|---|
| `train` / `val` / `test` | 동역학 실행 단위 80/10/10 (같은 `run_id`의 4개 클립은 항상 같은 분할) |
| `test_ood` | **학습에 없는 물성 영역**: 질량 0.32~0.43 g 이면서 입자-벽 반발계수 0.55~0.68. 처음 보는 물질에 대한 일반화 평가용 |
| `pairs` | 대조 쌍 (`pair_type` = `same_N`: 같은 N·다른 물성 / `same_material`: 같은 물성·다른 N, `pair_id`로 묶임) |

⚠️ 같은 `run_id`에서 나온 클립은 독립이 아닙니다. 학습/평가를 나눌 때 반드시 `run_id`(또는 `split`) 단위로 나누세요.

## 물리·음향 모델 요약

- **동역학**: 3차원 DEM, Hertz 접촉(탄성률 ×0.25, 강성 수렴 테스트 통과), 쿨롱 마찰+회전, 움직이는 벽, 바닥·천장 미세 요철(희박한 층의 비현실적 동기 튐 방지)
- **음원**: 벽 힘(저역) + 충돌 펄스(고역, 실제 물성의 Hertz 접촉 시간), 상보 필터로 이중 계산 없음
- **전달**: 1차 연구 녹음(N=5)에서 측정한 응답 통계(직접음 스펙트럼, 9~10 kHz 용기 모드와 방 잔향 τ≈62 ms) + sim2real 보정(DRR, EQ)
- **측정계**: 실측 모터 잡음(빈 용기, 372 rpm), 마이크 게인, 하드 클리핑(휴대폰 녹음처럼)

## 실측 대비 검증 (sim2real, 1차 연구 녹음 N = 5~100)

마이크 게인 하나로 맞춘 뒤: RMS(N에 따른 증가), 클리핑 비율, 스펙트럼 무게중심, 4 kHz 이상 대역 레벨, 파형 첨도(ψ3, N ≤ 50)는 실측 반복 편차 범위 안입니다.
알려진 차이: 검출되는 개별 충돌 수가 실측보다 많음(2~5배), N = 100에서 주기성이 과함, 0.5~2 kHz 대역 +4 dB.
자세한 표는 생성기 레포의 `results/sim2real/`에 있습니다.

## 한계

- 응답은 N에 무관합니다 (N별 임펄스 응답이 아직 측정되지 않음).
- 단일 rpm, 단일 용기. 지름이 다른 입자는 포함하지 않습니다 (지름을 모르면 N이 원리적으로 식별되지 않는 영역이 생김).
