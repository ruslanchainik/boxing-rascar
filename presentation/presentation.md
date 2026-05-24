---
marp: true
theme: default
paginate: true
backgroundColor: '#0b0d12'
color: '#e4e4e7'
style: |
  section {
    font-family: 'Inter', system-ui, sans-serif;
    background: linear-gradient(180deg, #11141b 0%, #0b0d12 100%);
    color: #e4e4e7;
    font-size: 24px;
    padding: 60px;
  }
  h1 { color: #ffb547; font-size: 48px; border-bottom: 2px solid #ffb547; padding-bottom: 12px; }
  h2 { color: #ffb547; font-size: 36px; margin-top: 0; }
  h3 { color: #f5d76e; font-size: 26px; }
  strong { color: #f5d76e; }
  code { background: #181c25; color: #f5d76e; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
  pre { background: #181c25 !important; border-radius: 8px; padding: 14px; font-size: 18px; }
  table { font-size: 20px; border-collapse: collapse; }
  th { background: #181c25; color: #ffb547; padding: 8px 12px; text-align: left; }
  td { padding: 8px 12px; border-bottom: 1px solid #252b38; }
  blockquote { border-left: 4px solid #ffb547; padding-left: 16px; color: #a1a1aa; }
  .red { color: #ef3b3b; }
  .blue { color: #3b82f6; }
  .green { color: #10b981; }
  .muted { color: #71717a; }
  section::after { color: #52525b; font-size: 16px; }
  .small { font-size: 18px; }
  .center { text-align: center; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
---

<!-- _class: title -->
<!-- _paginate: false -->

# Boxing Action Spotting

## Pose-based ансамбль для автоматического определения ударов на ринге

<br>

**RASCAR Box · Public LB 0.167**

<br>
<br>

<span class="muted">RingSight prototype · pose-19 TCN ensemble · DataSphere V100</span>

---

# Задача

**Дано:** видео раундов бокса (статичная камера за рингом), ~60 train, 9 test.

**Нужно:** для каждого удара — кадр, боец (red/blue), тип, рука, цель, эффективность.

**Метрика:**

```
final = 0.50·time + 0.20·fighter + 0.10·type + 0.08·eff
      + 0.06·hand + 0.06·target − fp_penalty
```

- Венгерский матчинг по времени (окно ±1 c)
- macro-average по 9 test видео
- FP штрафуется агрессивно: лучше пропустить, чем сгенерить лишнее

---

# Пайплайн (high-level)

<div class="columns">

<div>

**Frame stream**

→ Pose extractor (YOLO11n)

→ Glove tracker (YOLOv8n)

→ Feature builder (3 уровня)

→ TCN (×19, ensemble)

→ Peak picking + postprocess

→ Submission CSV

</div>

<div class="small">

- **Pose features** (426-dim): keypoints + bbox + цвета
- **Engineered** (+38): wrist accel, hand extension, hip rotation
- **Glove** (+28): glove tracks + velocity

**Ансамбль:** 9 orig + 5 eng + 5 glove → 19 моделей с равными весами

**TTA:** color-swap (red ↔ blue в фичах) — дал +0.028 LB

</div>

</div>

---

# Архитектура TCN

```python
class BoxingTCN(nn.Module):
    def __init__(self, input_dim, hidden=192, n_blocks=6):
        self.proj = nn.Conv1d(input_dim, hidden, 1)
        self.blocks = [TCNBlock(hidden, dilation=2**(i%5))]*n_blocks
        # multi-task per-frame heads
        self.event_head = nn.Conv1d(hidden, 1, 1)        # bg/punch
        self.attr_heads = {
            'fighter':       Conv1d(hidden, 2),  # red/blue
            'punch_type':    Conv1d(hidden, 4),  # jab/cross/hook/uppercut
            'hand':          Conv1d(hidden, 2),  # left/right
            'target':        Conv1d(hidden, 2),  # head/body
            'effectiveness': Conv1d(hidden, 3),  # landed/blocked/miss
        }
```

<div class="small muted">

~2.3M параметров. Dilated 1D conv с экспоненциально растущим receptive field
(до ~512 кадров ≈ 17 сек контекста при 30 fps). 6 блоков × dilation ∈ {1,2,4,8,16}.

</div>

---

# TCN block

```python
class TCNBlock(nn.Module):
    def __init__(self, ch, dilation, dropout=0.15, kernel=5):
        pad = (kernel-1)//2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(self.drop(h)))
        return self.act(x + h)        # residual
```

<div class="small muted">

GroupNorm вместо BatchNorm — стабильнее при малых батчах. GELU + residual.
Dilation позволяет captur'нуть длинный временной контекст без увеличения параметров.

</div>

---

# Loss

<div class="columns">

<div>

**Event head:** Focal BCE на Gaussian-soft target

```python
# target = Gaussian вокруг GT-кадра
event[lo:hi] = max(event[lo:hi], gauss(σ=3))
loss = focal_bce(logits, event,
                 γ=2, pos_weight=10)
```

Не one-hot — модель учится "плавному" пику события.

</div>

<div>

**Attribute heads:** class-weighted CE

```python
# class weights = inverse frequency
# (метрика взвешена так же)
loss += λ_n * CE(attr_logits[mask],
                 attr_target[mask],
                 weight=class_w[n])
```

Усиливает редкие классы: `uppercut`, `blocked`.

</div>

</div>

<br>

Multipliers по головам подобраны под доли метрики:
fighter=1.2, punch_type=0.6, eff/hand/target=0.5

---

# Source 1: Pose features (426-dim)

YOLO11n-pose даёт 17 keypoints на бойца. Стробим pose каждые 4 кадра, цвет каждые 4.

**Что в фиче на кадр:**

<div class="columns small">

<div>

**Per fighter (×2):**
- 17 keypoints × (x, y, conf) = 51
- bbox: cx, cy, w, h = 4
- HSV color ratios (red/blue/white): 3
- presence flag: 1

= **59 dim × 2 fighter = 118**

</div>

<div>

**Pair features (24):**
- wrist→opp head/body distance
- elbow angles cos
- own chest→opp body
- left/right wrist ordering

**Базовый dim = 142**

**Расширение Δ1, Δ3 деривативы:**
142 × 3 = **426**

</div>

</div>

<br>

<div class="muted small">

Pose извлекался один раз для всех 72 видео (stride=2, ~3 ч на RTX 3060).
ByteTrack для стабильности треков. Hungarian для red/blue role assignment.

</div>

---

# Source 2: Engineered features (+38 dim)

<div class="columns small">

<div>

**Per fighter (×2, 14 каждому):**
- wrist velocity magnitude (L, R)
- wrist acceleration magnitude
- wrist extension from chest
- hand crosses centerline (bool)
- elbow extension cos angle
- guard up indicator
- shoulder-hip twist velocity
- body COM velocity

</div>

<div>

**Pair (10):**
- inter-fighter distance
- approach velocity
- per-wrist → opp-nose distance
- ΔL/Δt этих расстояний

→ **464-dim вход** для второй группы моделей

</div>

</div>

<br>

> Кисти от 30 ⇒ 60 ⇒ 120 fps по производным дают модели «чувство удара»
> до того как кадр контакта вообще доехал.

---

# Source 3: Glove tracker (+28 dim)

<div class="small">

**Pipeline:**

1. **Auto-label** перчаток из существующих wrist keypoints (~5% от ширины кадра bbox)
   → 1700 кадров × 4 авто-бокса = 4847 training samples.
2. **YOLOv8n** обучаем 50 эпох (mAP@50 = 0.61, ~40 мин на RTX 3060).
3. **Re-process** все 72 видео glove детектором.
4. **Associate** каждый bbox перчатки к ближайшему wrist keypoint (Hungarian).
5. **Engineer:** glove center (x,y), conf, size, velocity, acceleration — 20+8 = 28 dim.

</div>

<br>

<div class="columns small">

<div>

**Зачем:**
- точнее wrist positions (YOLO-pose шумит на быстрых движениях)
- размер bbox = индикатор близости перчатки к камере
- glove-внутри-opp-bbox = прямой сигнал контакта

</div>

<div>

**Стоимость:**
- Auto-label обходит ручную разметку
- Тренировка → ~1 час локально
- Re-extract → ~30 мин

→ **492-dim вход** для третьей группы

</div>

</div>

---

# Тренировка и ансамбль

**Конфиг:**
- crop_len = 512 frames, batch = 8, 30 эпох
- AdamW, cosine LR
- Color-swap aug (50%), label smoothing 0.05
- EMA для v2 моделей

**Состав ансамбля (19 чекпоинтов):**

| Группа | Feature dim | Сидов | Архитектура |
|---|---|---|---|
| orig (v1) | 426 | 5 | TCN-192-6 |
| orig (v2fix) | 426 | 4 | + color-swap + EMA |
| eng | 464 | 5 | то же на engineered |
| glove | 492 | 5 | то же на glove |

Усреднение per-frame event scores и attr logits, потом peak picking.

---

# Postprocess (что зашло, что нет)

<div class="small">

| Что | Δ LB | Вердикт |
|---|---:|---|
| Grid-search threshold (0.65) + min_distance (6) | +0.069 | ✅ главный |
| 5-frame attr averaging | +0.004 | ✅ |
| Fighter smoothing (window ±5) | +0.003 | ✅ |
| Hand prior (majority hand per track) | +0.001 | ✅ marginal |
| **TTA color-swap** | **+0.028** | ✅✅ key |
| Ensemble glove + eng + orig | +0.024 | ✅✅ |
| Snap-to-wrist-velocity | **−0.003** | ❌ wrist peak ≠ моменту удара |
| Gaussian event smoothing σ=2 | −0.005 | ❌ размывает пики |
| Per-fight threshold post-TTA | −0.024 | ❌ старые числа неоткалиброваны |

</div>

---

# Прогресс по LB

<div class="small">

```
   0.026  →  single TCN, threshold 0.35 (дефолт)
   0.095  →  + grid-search threshold (0.65, md=6)
   0.103  →  + 5-frame avg + fighter smooth + hand prior
   0.118  →  + ensemble 4 strong checkpoints
   0.147  →  + TTA color-swap (huge!)
   0.153  →  + engineered features ensemble
   0.159  →  + weighted (orig:eng = 2:1)
   0.167  →  + glove features в meta3 (1:1:1, равные веса)
```

**Что не сработало:**

- Каскад frozen RegNetY-002 + TCN (LB 0.084, шумит ансамбль)
- E2E-Spot fine-tune на DataSphere V100 (упал на numpy deprecation, потрачено ~2M unit без чекпоинта)

</div>

---

# RingSight — интерфейс судейского ассистента

<div class="small">

**Что в UI** (`interface/index.html`):

- **Live video pane** с overlay'ями: bbox бойцов, glove markers, fighter labels
- **Round timeline** — лента 6 мин с цветными отметками ударов
  (красный/синий, прозрачность = effectiveness)
- **Two scorecards** — per-fighter breakdown по типам и effectiveness
- **Live events feed** — обновляется каждые ~3 с с подсветкой нового события
- **Detector confidence panel** — ensemble probability per компоненту удара
- **Round pace** — punches/min, land rate, aggressor side

**Стек:** single-file HTML + Tailwind CSS via CDN + ~80 строк vanilla JS.
Данные ударов — реальные предсказания нашей модели на агн_037.

</div>

<div class="muted center">Demo-build, не сертифицирован для официального судейства.</div>

---

# Realtime feasibility

На RTX 3060:

| Этап | Скорость |
|---|---|
| YOLO11n-pose (640px) | 60-100 fps |
| Glove YOLOv8n (480px) | 80-120 fps |
| TCN inference (1 модель) | ~1 мс на 100 frames |
| Ансамбль 19 TCN | ~20 мс на 100 frames |

**~40 мс/кадр** = вписываемся в 30 fps wall-clock.

**Inherent latency** для принятия решения: ~333 мс
(нужно увидеть пик в окне ~10 кадров после удара).

**Прод-сетап:** Jetson Orin или mini-PC с RTX 4060 за рингом, локальная обработка,
web-UI на планшете судьи. Не Mars-rover технология.

---

<!-- _backgroundColor: '#1a0b0b' -->

# 🥊 Но честно — бокс так не судят

**10-Point Must System** (AIBA / professional):

> Каждый раунд один боец получает **10 очков**, другой 9 или меньше.

Критерии решения судей:

1. **Effective aggression** — атака с реальным эффектом, не суета
2. **Ring generalship** — кто диктует темп, контролирует пространство
3. **Defense** — уклоны, парирование, маневр
4. **Effective punching** — *качество* удара, не количество
5. **Hard punches landed** — мощные акценты, сотрясение оппонента

<div class="muted small">

Подсчёт ударов по типам — это ОДНА из компонент пункта 4, и даже там вторична
по отношению к силе и точности.

</div>

---

# Что наша модель НЕ может оценить

<div class="small">

| Компонент судейства | Можем ли | Почему |
|---|---|---|
| Количество ударов по типам | ✅ | то что мы и считаем |
| Какой боец инициирует обмены | ⚠️ частично | можно по time-ordering ударов |
| Сила удара (force) | ❌ | нет force estimation, нужен 3D pose + accelerometer + reaction анализ |
| Сотрясение / отскок головы оппонента | ❌ | требует моделирования реакции тела |
| Контроль ринга | ❌ | нужен position tracking + agency analysis |
| Defense quality | ❌ | модель уклонений / блоков / footwork не учим |
| Punishment cumulative | ❌ | агрегация поверх раундов с учётом эффекта |
| Knockdowns / 8-counts | ❌ | референс к рефери, не к ударам |

</div>

<div class="muted">

Compubox (профессиональный официальный счётчик в бокс-трансляциях с 1985)
тоже считает только pure counts — и его данные регулярно расходятся с
оценками судей. Это не баг — это фундаментальное несоответствие метрики.

</div>

---

# Куда дальше для реального судейства

<div class="small">

**Чтобы стать настоящим ассистентом:**

1. **3D pose / multi-camera** — оценка реальной силы и точности удара по углу контакта.
2. **Reaction analysis** — детект отскока головы / тела / потери баланса соперника
   после удара (CNN на post-impact окне).
3. **Ring control map** — кто где стоит, кто давит, кто отступает к канатам.
4. **Defense scoring** — head movement, parry detection, shoulder roll.
5. **Style classification** — orthodox/southpaw, pressure vs counter, тактический контекст.
6. **Round-level aggregator** — нейросеть «учит судью», обученная на парах
   (видео раунда, оценка судьи 10-9 / 10-8) — это самая правильная архитектура.

**Наш текущий подход хорошо работает для:**

- Стат-обзор после боя (Compubox-like dashboard)
- Тренировочная аналитика (отслеживать прогресс спортсмена)
- TV graphics overlay (live punch counter)

Для **принятия решения о победителе раунда** — нет, и в этом проблема существующих
автоматических систем в боксе.

</div>

---

<!-- _backgroundColor: '#0b0d12' -->

# Итоги

<div class="columns">

<div class="small">

**Технически:**

- Pose-only TCN ансамбль из 19 моделей
- 3 уровня фичей: keypoints → engineered → glove
- TTA, ensemble, postprocess: 0.026 → **0.167** LB
- Realtime на RTX 3060: ~40 ms/кадр
- Repo: [boxing-rascar](https://github.com/ruslanchainik/boxing-rascar)

</div>

<div class="small">

**Концептуально:**

- Action spotting для бокса = только частичный сигнал
- Для официального судейства нужна оценка качества + контекста, а не подсчёт
- Реальная application: тренировочная статистика и TV overlay, не replacement судей

</div>

</div>

<br>

## Спасибо

<div class="muted center small">

Repo: github.com/ruslanchainik/boxing-rascar · Demo UI: interface/index.html

</div>
