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

```text
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
  deploy/
    triton/
      make_model_repo.py
      client_http.py
      model_repository/      # создаётся локально, в git не попадает
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
- при необходимости также `artifacts/onnx/model.onnx.data` (external data)

### 4) Инференс ONNX Runtime

#### 4.1 Инференс по индексу тестового примера Fashion-MNIST

```powershell
python -m fashion_mnist_classifier.commands infer_onnx --test_index=123
```

Пример вывода:

```text
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

## Inference server (Triton, CPU)

Ниже приведён пример поднятия сервера инференса для экспортированной ONNX-модели с помощью **NVIDIA Triton Inference Server** через Docker. Запуск выполняется в **CPU-режиме** (без NVIDIA GPU).

### Предусловия
- Docker Desktop запущен (`docker info` работает).
- ONNX модель экспортирована в `artifacts/onnx/`:

```powershell
python -m fashion_mnist_classifier.commands export_onnx
```

### 1) Подготовка model repository для Triton

Сгенерировать `config.pbtxt` и скопировать ONNX-модель в репозиторий Triton:

```powershell
python .\deploy\triton\make_model_repo.py build --model_name fashion_mnist
```

Если ONNX был сохранён с external data (рядом лежит `model.onnx.data`), скопируйте его в ту же папку версии модели:

```powershell
Copy-Item artifacts\onnx\model.onnx.data deploy\triton\model_repository\fashion_mnist\1\model.onnx.data -Force
```

Проверка, что файлы на месте:

```powershell
dir deploy\triton\model_repository\fashion_mnist\1
```

### 2) Исправление config.pbtxt (CPU + TYPE_FP32 + UTF-8 без BOM)

Triton читает `config.pbtxt` как protobuf text-format. Частые причины падений на Windows:
- `data_type` должен быть `TYPE_FP32` (а не `FP32`)
- `instance_group.kind` должен быть `KIND_CPU` для CPU-режима
- файл должен быть в UTF-8 **без BOM** (иначе protobuf-парсер падает на первом символе)

Команда для автоматического исправления:

```powershell
$p = "deploy\triton\model_repository\fashion_mnist\config.pbtxt"
$txt = Get-Content $p -Raw
$txt = $txt -replace "KIND_GPU", "KIND_CPU"
$txt = $txt -replace "data_type:\s*FP32", "data_type: TYPE_FP32"
[System.IO.File]::WriteAllText($p, $txt, (New-Object System.Text.UTF8Encoding($false)))
```

### 3) Запуск Triton Server (HTTP / gRPC / metrics)

Откройте отдельный терминал и запустите Triton:

```powershell
docker run --rm --shm-size=1g `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}\deploy\triton\model_repository:/models" `
  nvcr.io/nvidia/tritonserver:25.10-py3 `
  tritonserver --model-repository=/models
```

Проверка готовности сервера:

```powershell
curl http://localhost:8000/v2/health/ready
```

Ожидаемый результат: `OK`.

### 4) Тестовый запрос (HTTP)

В другом терминале:

```powershell
python .\deploy\triton\client_http.py infer --server_url "http://localhost:8000" --model_name "fashion_mnist" --test_index 0
```

Ожидаемый результат: вывод содержит предсказанный класс, например `pred=9`.

### Примечание про артефакты
Файлы `deploy/triton/model_repository/**/model.onnx` и `model.onnx.data` являются артефактами модели и не должны коммититься в git.

## Метрики

В ходе обучения вычисляются:
- Accuracy на test
- Macro-F1 на test
- Confusion matrix (сохраняется в `metrics.json`)

## Примечания

- Проект рассчитан на запуск на CPU (датасет небольшой).
- При отсутствии GPU возможны предупреждения, связанные с pinned memory; на корректность результата они не влияют.
