# RingSight — судейский ассистент (demo)

Single-file HTML mock интерфейса поверх нашего pose-only ансамбля.
Открывается в любом современном браузере, ничего ставить не нужно.

```
open interface/index.html
```

(или двойной клик)

## Что внутри

- **Live video pane** — заглушка с CSS-нарисованным рингом, overlay'ы bbox/glove.
- **Match info bar** — Турнир Бокс · Бой 9 · АГН-037, round timer.
- **Round timeline** — лента 6 минут с цветными отметками ударов (red/blue,
  прозрачность = effectiveness landed/blocked/miss). Бегунок двигается.
- **Two scorecards** — per-fighter разбивка по типам / по effectiveness.
- **Live events feed** — обновляется каждые 3.5 сек, новый удар "влетает"
  с подсветкой; данные настоящие, из `submissions/submission_meta3_w111.csv`
  для агн_037.
- **Pace panel** — punches/min, land rate, кто инициирует обмены.
- **Detector confidence** — мок-бары ensemble confidence по компонентам.
- **Control buttons** — export report / recalibrate / score sheet / stop round
  (mock, не делают ничего).

## Realtime feasibility (для пресс-релиза)

- YOLO11n-pose + glove YOLOv8n + 19 TCN ансамбль укладываются в **~40 ms / 30 fps кадр**
  на RTX 3060 → wall-clock реальное время.
- Latency решения ~333 ms (надо подождать ~10 кадров после удара чтобы увидеть пик
  по event score) — приемлемо для судьи.
- Прод-сетап: Jetson Orin / мини-PC с RTX 4060 за рингом, web-UI на планшете судьи.

## Стек

- Tailwind CSS via CDN
- Vanilla JS (~80 строк)
- Шрифты: Inter, JetBrains Mono (Google Fonts)

Никаких зависимостей и сборки.
