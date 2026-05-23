# Boxing Action Spotting — RASCAR Box

Локально-обученный pose-based ансамбль из 19 TCN-моделей для соревнования
[RASCAR Box / Boxing Action Recognition Challenge](https://www.kaggle.com/competitions/rascar-box).

**Лучший результат: `submission_meta3_w111.csv` — public LB 0.16718.**

Прогресс по сабмитам:

```
baseline tuned (single TCN):  0.0954
+ postprocess:                 0.1035
+ ensemble (strong4):          0.1181
+ TTA color-swap:              0.1532
+ engineered features:         0.1567
+ weighted (orig:eng=2:1):     0.1590
+ glove features ensemble:     0.16718   ← best
```

## Архитектура

Три feature-группы поверх YOLO-pose keypoints (по `extract_pose_features.py`):

1. **orig (426-dim):** базовые pose-features (17 keypoints × 2 fighter + bbox + color masks)
   и их временные дельты Δ1 / Δ3.
2. **eng (464-dim):** orig + engineered features из `boxing_lstm_pipeline/engineered_features.py`
   (wrist acceleration, hand crossing centerline, elbow angles, hip rotation, etc.)
3. **glove (492-dim):** eng + glove tracks из обученного YOLOv8n детектора перчаток.

Поверх каждой группы — `BoxingTCN` (`boxing_lstm_pipeline/tcn_model.py`):
- 1D dilated TCN, hidden 192, 6 блоков
- Per-frame multi-task heads: event score + fighter / punch_type / hand / target / effectiveness
- Loss: focal BCE на event + class-balanced CE на attributes
- Gaussian soft targets (σ=3) вокруг GT-кадров

Ансамблируется 19 чекпоинтов (9 orig × 5 eng × 5 glove) с равными весами и TTA через
color-swap (red ↔ blue в фичах + swap fighter logits). Подробности в `docs/LB_LOG.md`.

## Структура

```
boxing-spotting/
├── boxing_lstm_pipeline/      # core: модель + датасеты + pose-features
│   ├── tcn_model.py            # BoxingTCN
│   ├── tcn_dataset.py          # per-frame Gaussian targets
│   ├── tcn_dataset_v2.py       # + color-swap aug + hard-neg
│   ├── pose_features.py        # YOLO-pose extractor
│   ├── engineered_features.py  # wrist accel/extension/etc.
│   └── paths.py
├── scripts/
│   ├── extract_pose_features.py        # video → 426-dim pose npz
│   ├── build_engineered_features.py    # pose → 464-dim eng npz
│   ├── extract_glove_tracks.py         # YOLOv8 glove → tracks
│   ├── build_features_with_glove.py    # eng + glove → 492-dim npz
│   ├── build_glove_dataset.py          # auto-label glove YOLO data
│   ├── train_tcn.py                    # v1 train
│   ├── train_tcn_v2.py                 # v2 with color-swap + EMA
│   ├── predict_tcn.py                  # single-model predict
│   ├── predict_tcn_v2.py               # + TTA + postprocess
│   ├── predict_meta_ensemble.py        # orig + eng ensemble
│   ├── predict_meta3.py                # orig + eng + glove ensemble  ← best
│   ├── predict_meta4.py                # + cloud RGB (didn't help)
│   ├── tune_threshold*.py              # grid-search threshold/min_distance
│   ├── label_video.py                  # manual annotation tool (cv2 UI)
│   ├── run_*_pipeline.sh               # batch retrain wrappers
│   └── cloud/                          # DataSphere setup (E2E-Spot attempts)
├── submissions/
│   └── submission_meta3_w111.csv       # best LB 0.16718
└── docs/
    ├── hack.md                          # original task description
    ├── STRATEGY.md                      # 15 hypotheses + SOTA review
    ├── LB_LOG.md                        # full leaderboard log
    ├── STATUS.md / STATUS_CLOUD.md     # resume notes
    └── DATASPHERE.md                    # Yandex DataSphere cheatsheet
```

## Воспроизведение

### Требования

```bash
pip install -r requirements.txt
```

Нужны видео из соревнования (Турнир Бокс / Турнир Бокс 2 / бокс) и YOLO weights
`yolo11n-pose.pt`, `yolov8n.pt`. Пути жёстко в `boxing_lstm_pipeline/paths.py`.

### Полный pipeline (с нуля)

```bash
# 1. Pose-фичи (~2 ч на RTX 3060)
python scripts/extract_pose_features.py --split train
python scripts/extract_pose_features.py --split test

# 2. Engineered features (~30 сек)
python scripts/build_engineered_features.py

# 3. Glove tracker (~1 ч обучение)
python scripts/build_glove_dataset.py
bash scripts/run_glove_pipeline.sh

# 4. Glove tracks (~1 ч inference)
python scripts/extract_glove_tracks.py
python scripts/build_features_with_glove.py

# 5. Тренировка 19 моделей (~30 мин на RTX 3060)
bash scripts/run_local_workplan.sh        # 5 v1 + 3 v2fix
bash scripts/run_eng_pipeline.sh          # 3 v1 eng + 2 v2fix eng
bash scripts/run_glove_features_pipeline.sh  # 3 v1 glove + 2 v2fix glove

# 6. Финальный сабмит
python scripts/predict_meta3.py \
  --ckpts_orig artifacts/models/boxing_tcn{,_v1_s2026,_v1_s1337,_v1_s4242,_v1_s31337,_v1_s8088,_v2fix_s2026,_v2fix_s1337,_v2fix_s4242}.pt \
  --ckpts_eng artifacts/models/boxing_tcn_eng_s{2026,1337,4242}.pt artifacts/models/boxing_tcn_v2fix_eng_s{2026,4242}.pt \
  --ckpts_glove artifacts/models/boxing_tcn_glove_s{2026,1337,4242}.pt artifacts/models/boxing_tcn_v2fix_glove_s{2026,4242}.pt \
  --w_orig 1.0 --w_eng 1.0 --w_glove 1.0 \
  --tta --out submission_meta3_w111.csv
```

## Что не сработало

- **Frozen RGB CNN-фичи (RegNetY-002, кэш + TCN):** LB 0.084 single, не добавил
  ансамблю → `predict_meta4.py` хуже чем `predict_meta3.py`. Frozen ImageNet-features
  не несут task-сигнала.
- **E2E-Spot fine-tune на DataSphere V100:** 2 запуска по 6.5 + 1.5 ч, оба упали:
  v3 на numpy deprecation в spot/util/score.py, v4 cancelled mid-train. DataSphere
  не выгружает output при non-zero exit. Потрачено ~2M unit, чекпоинт не получили.
  Подробности в `docs/STATUS_CLOUD.md`.
- **Snap-to-wrist-velocity** в postprocess: −0.003 LB. Wrist velocity peak ≠ момент
  удара (ловит retract).
- **Heavy temporal smoothing** (Gaussian σ=2 на event scores): −0.005 LB.
  Размывает локальные максимумы.

## Лицензия и кредиты

Код мой. Модель E2E-Spot (в `scripts/cloud/`) — fork
[jhong93/spot](https://github.com/jhong93/spot) (BSD-3, не использовался в финале).

Состязание: RASCAR Box. Данные — собственность организаторов.
