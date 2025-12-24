# Fashion-MNIST Classifier (PyTorch → ONNX → ONNX Runtime)

Проект: классификация изображений одежды из датасета **Fashion-MNIST** на 10 классов с обучением модели в PyTorch, экспортом в ONNX и инференсом через ONNX Runtime.

## Возможности

- Загрузка и подготовка данных Fashion-MNIST (автоскачивание при первом запуске)
- Обучение CNN-классификатора в PyTorch
- Сохранение лучшего чекпоинта по Macro-F1
- Экспорт обученной модели в ONNX
- Инференс ONNX-модели:
  - по тестовому индексу из Fashion-MNIST
  - по пользовательскому изображению (приведение к 28×28 grayscale)

## Правила проекта (учтены)

- Нет исполняемого кода на уровне модулей (кроме `if __name__ == "__main__": main()`)
- Артефакты (датасет, метрики, веса, ONNX) **не коммитятся** в git и сохраняются в `artifacts/`
- Единая входная точка: `fashion_mnist_classifier/commands.py`
- CLI: **fire**
- Конфиги: **hydra compose API**
- Пути: `pathlib`

## Структура репозитория

```
fashion-mnist-classifier/
  fashion_mnist_classifier/
    __init__.py
    commands.py
    data.py
    model.py
    train.py
    export_onnx.py
    infer.py
    utils.py
  configs/
    base.yaml
  tests/
  artifacts/                # создаётся локально, в git не попадает
    datasets/
    checkpoints/
    onnx/
    metrics/
  requirements.txt
  README.md
  .gitignore
```

## Установка

### 1) Виртуальное окружение (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Установка зависимостей

```powershell
pip install -r requirements.txt
```

## Команды

Все команды запускаются так:

```powershell
python -m fashion_mnist_classifier.commands <command> [args...]
```

Для переопределения параметров через Hydra compose API:

```powershell
python -m fashion_mnist_classifier.commands <command> --overrides="key1=value1,key2=value2"
```

### 1) Smoke-test загрузки данных

Проверяет, что датасет скачивается и DataLoader выдаёт корректные батчи.

```powershell
python -m fashion_mnist_classifier.commands smoke
```

Пример с изменением batch size:

```powershell
python -m fashion_mnist_classifier.commands smoke --overrides="data.batch_size=64"
```

### 2) Обучение модели

Обучение CNN, сохранение лучшего чекпоинта и метрик:

```powershell
python -m fashion_mnist_classifier.commands train
```

Артефакты сохраняются локально:
- `artifacts/checkpoints/best.pt`
- `artifacts/metrics/metrics.json`

Пример переопределения числа эпох:

```powershell
python -m fashion_mnist_classifier.commands train --overrides="train.epochs=3"
```

### 3) Экспорт в ONNX

Экспортирует лучший чекпоинт в ONNX:

```powershell
python -m fashion_mnist_classifier.commands export_onnx
```

Выход:
- `artifacts/onnx/model.onnx`

### 4) Инференс ONNX Runtime

#### 4.1 Инференс по индексу тестового примера Fashion-MNIST

```powershell
python -m fashion_mnist_classifier.commands infer_onnx --test_index=123
```

Пример вывода:

```
true_label=9 predicted_class=9 label=Ankle boot confidence=0.9986
```

#### 4.2 Инференс по пользовательскому изображению

```powershell
python -m fashion_mnist_classifier.commands infer_onnx --image_path="D:\path\to\image.png"
```

Изображение будет приведено к формату:
- grayscale
- 28×28
- нормализация как при обучении

## Конфигурация

Базовый конфиг: `configs/base.yaml`.

Ключевые параметры:
- `train.epochs`, `train.lr`, `train.weight_decay`
- `model.dropout`
- `data.batch_size`, `data.num_workers`
- `paths.dataset_dir`, `paths.checkpoint_dir`, `paths.onnx_dir`, `paths.metrics_dir`

## Артефакты (не коммитятся)

Все артефакты сохраняются в `artifacts/` и игнорируются git:
- датасет: `artifacts/datasets/`
- чекпоинты: `artifacts/checkpoints/`
- ONNX модель: `artifacts/onnx/`
- метрики: `artifacts/metrics/`

## Метрики

В ходе обучения вычисляются:
- Accuracy на test
- Macro-F1 на test
- Confusion matrix (сохраняется в `metrics.json`)

## Примечания

- Проект рассчитан на запуск на CPU (датасет небольшой).
- При отсутствии GPU возможны предупреждения, связанные с pinned memory; на корректность результата они не влияют.
