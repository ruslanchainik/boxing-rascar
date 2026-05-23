# Лидерборд / Журнал гипотез

**Текущий лучший: `submission_meta3_w111.csv` = 0.16718** (19 ckpts: orig+eng+glove, веса 1:1:1, +TTA)

Прогресс: 0.026 → 0.095 → 0.118 → 0.147 → 0.153 → 0.159 → **0.167**

### Three-modality ensemble (orig + eng + glove)

| Конфиг | LB |
|---|---:|
| orig-only TTA (9 ckpts) | 0.14655 |
| eng-only TTA (5 ckpts) | 0.11536 |
| glove-only TTA (5 ckpts) | 0.12538 |
| meta orig+eng 1:1 | 0.15324 |
| meta orig+eng 2:1 | 0.15904 |
| **meta3 1:1:1** ⭐ | **0.16718** |
| meta3 2:1:1 | 0.15650 |
| meta3 2:1:2 | 0.15897 |

**Вывод:** glove добавляет ~+0.008 к ансамблю. При 19 моделях равные веса лучше — диверсификация моделей сама работает как регуляризатор.

## Хронология рекордов

| Этап | LB | Что |
|---|---:|---|
| baseline tuned | 0.0954 | TCN v1 + thr=0.65 md=6 |
| best_combo | 0.10350 | + fighter_smooth + 5frame_avg + hand_prior |
| aw9 | 0.10431 | + attr_avg окно 9 (half=4) |
| strong4 | **0.11807** | ensemble 4 сильных ckpts |
| **final_combo** | **0.14686** | strong9 + TTA color-swap + class rebalance |

## Главные открытия в порядке силы

### 1. TTA color-swap (+0.028) — главный single buster
Модели чувствительны к red/blue из-за color-swap aug. Сама color_swap aug на train + усреднение с swapped predictions на inference выравнивает.

### 2. Ensemble сильных моделей (+0.015 от single → strong4)
4 хороших чекпоинта дают почти 13% буст vs single. Дальнейшее расширение до 9 — только +0.006.

### 3. Per-fight threshold (v3<v2<v1) (+0.011)
Tour3 (AGN-62/63/64) нуждается в низком threshold (больше реальных событий) → 0.6. Tour1 (37-39) — выше (0.7). Обратная схема (v1=0.6/v3=0.7) **вредит** (-0.009).

### 4. 5-frame attr averaging + hand_prior + fighter_smooth (+0.008)
Базовый постпроцесс.

### 5. Class rebalance (+0.006) — слабый
Помогает в одиночку, но в комбо с TTA вклад ~0.

### Что НЕ работает

- **Snap-to-velocity**: -0.003 (wrist peak ≠ кадр удара)
- **fighter_smooth с шириной 21 кадр**: -0.005 (слишком много усреднения)
- **Gaussian event smoothing σ=2**: -0.005

## Файлы и состав

| Файл | Состав | LB |
|---|---|---:|
| `submission_final_combo.csv` ⭐ | strong9 + TTA + classrebal | **0.14686** |
| `submission_tta_colorswap.csv` | strong9 + TTA | 0.14655 |
| ~~submission_tta_perfight.csv~~ | strong9 + TTA + perfight(0.7/0.65/0.6) | 0.12238 ❌ |
| ~~submission_tta_perfight_aggro.csv~~ | strong9 + TTA + perfight(0.75/0.65/0.55) | 0.06477 ❌ |

**Урок: TTA + per-fight thresholds антагонистичны.** TTA меняет распределение event-скоров → старые per-fight numbers (0.7/0.65/0.6) неоткалиброваны для TTA-output'а. Нужно: либо grid-search threshold *поверх TTA*, либо использовать TTA одно без перфайт-тюнинга.

## Ckpts ensemble (9 моделей)

| Файл | Best score / Best val_loss | Состав |
|---|---|---|
| `boxing_tcn.pt` | 0.0085 | original |
| `boxing_tcn_v1_s2026.pt` | 0.0085 | full 30 epochs |
| `boxing_tcn_v1_s1337.pt` | val=3.01 | epoch 4 (proxy stuck) |
| `boxing_tcn_v1_s4242.pt` | 0.0444 | epoch 24 |
| `boxing_tcn_v1_s31337.pt` | **0.0672** | best v1 single |
| `boxing_tcn_v1_s8088.pt` | val=3.32 | epoch 4 |
| `boxing_tcn_v2fix_s2026.pt` | val=2.87 | epoch 14 |
| `boxing_tcn_v2fix_s1337.pt` | val=2.95 | epoch 8 |
| `boxing_tcn_v2fix_s4242.pt` | **0.0765** | best overall single |

## В очереди (требует облако / много времени)

- DWPose / RTMW re-extract (точнее кисти): +0.02-0.04 локально
- Glove tracker (YOLO fine-tune): +0.02-0.04 локально
- **E2E-Spot RGB fine-tune** (cloud V100, ~4-8 ч, ~$3-5): +0.10-0.15
- **T-DEED RGB** (cloud, ~15-25 ч, ~$10-15): +0.15-0.20
- **InternVideo2 features**: +0.10-0.20 (foundation model)
- Финальный ансамбль pose+RGB: pose часть здесь — половина потенциального финального бустa
