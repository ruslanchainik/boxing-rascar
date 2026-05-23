# Текущий статус (пауза)

## Что сделано

- Стратегия: `STRATEGY.md` (15 ранжированных гипотез, SOTA-обзор, roadmap).
- Код H1 (per-frame TCN с Gaussian soft targets + multi-task heads):
  - `boxing_lstm_pipeline/tcn_model.py` — TCN модель (2.3M параметров).
  - `boxing_lstm_pipeline/tcn_dataset.py` — per-frame датасет, Gaussian targets, class weights.
  - `scripts/train_tcn.py` — обучение, focal BCE + multi-task loss, прокси-метрика.
  - `scripts/predict_tcn.py` — peak-picking, NMS, mapping fps→30, padding до 1594 строк.
  - `scripts/run_pipeline.sh` — единый чейн extract→train→predict.
  - `scripts/extract_pose_features.py` — добавлен skip-existing и graceful skip отсутствующих файлов.
- Sanity-чек модели: forward/backward работают, ~77 MB VRAM на B=4 T=512.

## Прогресс extraction

- **31 из 72 видео извлечено** в `artifacts/pose_features/` (~150 MB):
  - `agn_001..agn_031`
- Все 13 из `бокс/` + 18 из `Турнир Бокс`.
- Остаётся ~41 видео (часть Турнир Бокс + весь Турнир Бокс 2 + 9 test).

## Как продолжить завтра

Просто запустить тот же chained pipeline — `skip-existing` пропустит уже готовое:

```powershell
cd C:\hack
bash scripts/run_pipeline.sh > artifacts/pipeline.log 2>&1 &
```

или в фоне через Claude.

Pipeline:
1. **Step 1/4** — extract train (продолжит с agn_032).
2. **Step 2/4** — extract test (9 видео).
3. **Step 3/4** — train TCN (30 эпох, ~5 мин).
4. **Step 4/4** — predict → `submission_tcn.csv`.

ETA на остаток: ~2 ч extract + 10 мин обучение/predict.

## Файлы

- Лог: `artifacts/pipeline.log` (хранит весь прогресс).
- Чекпоинт после обучения: `artifacts/models/boxing_tcn.pt`.
- Финальный сабмит: `submission_tcn.csv`.

## Замечания / что улучшить дальше

- Метрика жёстко 30 fps — в `predict_tcn.py` уже маппинг есть (H6).
- Сейчас threshold=0.35, min_distance=10. После первого сабмита прогнать grid-search на холд-аут.
- Дальше из стратегии: H5 (стабильный fighter assignment), H15 (hard-negative mining), H10 (ансамбль с RGB-моделью в облаке).
