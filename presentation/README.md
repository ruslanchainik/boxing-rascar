# Презентация

Файл [`presentation.md`](presentation.md) — слайды в формате
[Marp](https://marp.app/), tема dark под цвет нашего UI.

## Как посмотреть

### Вариант 1: VS Code (проще всего)

```
ext install marp-team.marp-vscode
```

Открыть `presentation.md` → нажать иконку **Preview** в правом верхнем углу
→ листать слайды стрелками. F11 для full-screen.

### Вариант 2: Marp CLI (export в PDF/PPTX/HTML)

```bash
npm install -g @marp-team/marp-cli

# В PDF
marp presentation.md --pdf

# В HTML (с реалтайм-навигацией)
marp presentation.md --html

# В PowerPoint
marp presentation.md --pptx
```

### Вариант 3: Marp Web

https://web.marp.app → перетащить файл.

## Структура (16 слайдов)

1. Title
2. Задача и метрика RASCAR Box
3. Pipeline high-level
4. **TCN архитектура** (код)
5. TCN block (код)
6. Loss (event + attr heads)
7. **Pose features** (426-dim)
8. **Engineered features** (+38)
9. **Glove tracker** (+28)
10. Тренировка + ансамбль 19 моделей
11. Postprocess: что зашло, что нет
12. Прогресс по LB
13. **RingSight UI**
14. Realtime feasibility
15. **🥊 Бокс так не судят** (10-point must system)
16. Что не может оценить наша модель + куда дальше
17. Итоги

Финальная мысль: подсчёт ударов ≠ судейство в боксе. Реальная application —
тренировочная аналитика и TV graphics, не replacement judges.
