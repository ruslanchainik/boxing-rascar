# Cloud status (пауза до завтра)

## Где остановились

LB best: **0.16718** (`submission_meta3_w111.csv`, локально, 19 моделей: orig+eng+glove).

Облачный pipeline TCN-RGB готов, но не отработал. Текущая ошибка job'а
`bt14e8a64p6160ei1sre`:

```
/job/.job_python_venv_6Gi/bin/python: can't open file '/job/extract_cnn_features.py':
[Errno 2] No such file or directory
```

DataSphere CLI не подтянул `extract_cnn_features.py`, `train_tcn_rgb.py`,
`predict_tcn_rgb.py` — он анализирует только импорты `run_pipeline.py`,
а вызовы через subprocess не видит.

## Что починить завтра

Добавить файлы как explicit inputs в `cloud_job/datasphere_config.yaml`,
либо импортировать их в `run_pipeline.py` (чтобы CLI определил как dependency).

Простой фикс — добавить в начало `cloud_job/run_pipeline.py`:
```python
import extract_cnn_features  # noqa
import train_tcn_rgb         # noqa
import predict_tcn_rgb       # noqa
```

После этого `bash scripts/cloud/launch.sh` — должно подняться нормально
на g1.1 (V100). ETA: ~30 мин в облаке. Стоимость: ~₽50-80.

## Состояние

| Что | Статус |
|---|---|
| Local pose+eng+glove ensemble (19 ckpts) | ✅ готово, LB 0.16718 |
| RGB crops (72 mp4 ~767MB) | ✅ готово в `artifacts/rgb_crops/` |
| cloud_job/ упакован (1.1 GB) | ✅ готово |
| IAM-token из `authorized_key.json` | ✅ работает (`YC_IAM_TOKEN`) |
| DataSphere job | ❌ не выполнился, файлы не загрузились |
| Все cloud jobs | ✅ остановлены (последнее: ERROR) |

## Команды для завтра

```powershell
# 1. Применить фикс к run_pipeline.py (добавить импорты).
# 2. Запустить:
cd C:\hack
bash scripts/cloud/launch.sh
```

Или без launch.sh:
```powershell
$env:YC_IAM_TOKEN = (python scripts\cloud\get_iam_token.py "C:\Users\Руслан\Downloads\authorized_key.json")
cd C:\hack\cloud_job
& "C:\Users\Руслан\AppData\Roaming\Python\Python314\Scripts\datasphere.exe" project job execute -p bt12vk1slp4dtfgh46vk -c datasphere_config.yaml
```

## Если что-то идёт не так

```powershell
# Список job-ов
datasphere project job list --project-id bt12vk1slp4dtfgh46vk

# Отменить конкретный
datasphere project job cancel --id <job-id>
```
