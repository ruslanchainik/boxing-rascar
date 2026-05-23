# DataSphere — быстрая шпаргалка

## Подготовка (один раз)

```powershell
pip install pyjwt[crypto] requests datasphere
```

DataSphere CLI окажется в:
```
C:\Users\Руслан\AppData\Roaming\Python\Python314\Scripts\datasphere.exe
```

## Аутентификация

Сервисный ключ: `C:\Users\Руслан\Downloads\authorized_key.json`.

Получить IAM-токен из ключа (живёт 1 час):

```powershell
$env:YC_IAM_TOKEN = (python C:\hack\scripts\cloud\get_iam_token.py "C:\Users\Руслан\Downloads\authorized_key.json")
```

**Важно**: переменная окружения называется `YC_IAM_TOKEN`, не `YC_TOKEN` (та для OAuth).

Project ID: `bt12vk1slp4dtfgh46vk`.

## Запуск job

### Через готовые launch-скрипты

```powershell
# TCN-RGB (cnn features + tcn) — ~25 мин
bash C:\hack\scripts\cloud\launch.sh

# E2E-Spot end-to-end — ~2-8 ч в зависимости от настроек
bash C:\hack\scripts\cloud\launch_spot.sh
```

Скрипты сами обновят IAM-токен и стартанут job из соответствующей папки.

### Вручную

```powershell
$env:YC_IAM_TOKEN = (python C:\hack\scripts\cloud\get_iam_token.py "C:\Users\Руслан\Downloads\authorized_key.json")
cd C:\hack\cloud_spot   # или cloud_job
datasphere project job execute -p bt12vk1slp4dtfgh46vk -c datasphere_config.yaml
```

## Мониторинг и управление

### Список job

```powershell
datasphere project job list --project-id bt12vk1slp4dtfgh46vk
```

Статусы: `CREATING` → `PREPARING` (pip install) → `EXECUTING` → `UPLOADING_OUTPUT` → `SUCCESS`/`ERROR`/`CANCELLED`.

### Прикрепиться к live job (стримит логи)

```powershell
datasphere project job attach --id <job-id>
```

Лог приходит сюда же в терминал. Можно безопасно убить локальный CTRL+C — cloud job продолжит работать.

### Получить статус одной job

```powershell
datasphere project job get --id <job-id>
```

### Убить job

```powershell
datasphere project job cancel --id <job-id>
```

Можно убить почти на любом этапе (PREPARING, EXECUTING). В UPLOADING_OUTPUT cancel может не дать эффекта.

### Убить ВСЕ активные

```bash
DS="C:/Users/Руслан/AppData/Roaming/Python/Python314/Scripts/datasphere.exe"
for j in $("$DS" project job list --project-id bt12vk1slp4dtfgh46vk 2>/dev/null \
    | grep -aE "EXECUTING|PREPARING|PROVISIONING" | awk '{print $1}'); do
    "$DS" project job cancel --id $j
done
```

### Скачать output после успешного завершения

```powershell
datasphere project job download-files --id <job-id> --output-dir <local-dir>
```

Скачаются все файлы из `outputs:` секции yaml.

## datasphere_config.yaml — ключевые поля

```yaml
name: my-job
cmd: python run_pipeline.py       # cmd ОБЯЗАТЕЛЬНО должен начинаться с `python`
                                  # (не `bash`, не shell скрипт)
env:
  python:
    type: manual
    version: '3.10'
    requirements-file: requirements.txt   # auto pip install

inputs:                            # пути относительно конфига
  - input/data_dir                 # директории работают
  - file.py                        # отдельные файлы
                                   # ВАЖНО: subprocess-вызываемые .py НЕ детектятся
                                   # автоматически, явно вписывай в inputs

outputs:
  - output/results.csv
  - output/checkpoints

cloud-instance-types:
  - g1.1     # V100 32GB, ~₽100-130/час (доступно нашей community)
  # g2.1 — A100, НЕ доступно
```

## Гочи которые мы поймали

1. **Windows backslash в путях.** DataSphere CLI на винде заливает `input/rgb_crops` как один файл/директорию с именем `input\rgb_crops` (с обратным слэшем!) на linux-worker. Перед использованием:
   ```python
   for legacy in ["input\\rgb_crops"]:
       if os.path.exists(legacy):
           os.rename(legacy, legacy.replace("\\", "/"))
   ```

2. **subprocess-вызовы python-скриптов** не подтягиваются автоматически. Нужно явно их перечислять в `inputs:`, иначе на worker'е будет `FileNotFoundError`.

3. **Cmd должен начинаться с `python ...`**. Если поставить `bash`, `set`, или multi-line shell-скрипт — будет ошибка `file 'set' was not found`. Чтобы выполнить несколько шагов — оберни их в `run_pipeline.py`, который через subprocess вызывает остальные.

4. **CUDA torch.** Дефолтный `torch>=2.0` через pip ставит CPU-only. В requirements.txt явно:
   ```
   --extra-index-url https://download.pytorch.org/whl/cu121
   torch==2.3.1+cu121
   torchvision==0.18.1+cu121
   ```

5. **timm API.** spot-репо ожидает `timm.models.layers.conv_bn_act.ConvBnAct` (timm <0.9). С новым timm 0.9+ ломается. Чтобы не патчить shift.py, поставь `timm==0.6.13`.

6. **Каждый запуск — новая VM.** Outputs предыдущих jobs не сохраняются. Каждый раз pip install + frame extract повторяются. Для оптимизации — упаковывай pre-processed данные как input.

7. **Pagefile/CUDA dll error на Windows ноуте.** Если запускаешь yolo/torch локально и видишь `cublas64_12.dll` — это лимит pagefile. Убей фоновые python процессы, перезапусти.

## Tail последних DataSphere temp-логов (если CLI упал)

Локальные логи каждого CLI-запуска:

```
C:\Users\27C6~1\AppData\Local\Temp\datasphere\job_2026-MM-DDTHH-MM-SS.NNNNNN\
├── log.txt           # подробный лог CLI + cloud worker output
├── stdout.txt        # stdout cmd
├── stderr.txt        # stderr cmd
└── system.log        # системный лог
```

Самый свежий: `ls C:\Users\27C6~1\AppData\Local\Temp\datasphere\ -t | head -1`.

## Ориентир по стоимости

| Что | Время | ₽ |
|---|---|---|
| pip install (torch+cu121+aux) | 5-10 мин | ~₽15-25 |
| Каскад: CNN feat extract + TCN train | 20-30 мин | ~₽40-70 |
| End-to-end E2E-Spot (RegNetY-002, 50 эпох) | 2-8 ч | ~₽200-1100 |
| T-DEED (CVPR'24) | 5-15 ч | ~₽500-2000 |
