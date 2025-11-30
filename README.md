# SmolVLM2

## Описание
3 use cases:
- Visual Question Answering<br>
    В интерфейсе можно добавить картинку и задать вопрос. На выходе получается ответ нейросети в текстовом поле. Можно задавать несколько вопросов без повторной загрузки изображения.
- Image Captioning<br>
    В интерфейсе можно добавить картинку и получить автоматическое описание изображения. Оптимизировано для быстрой генерации кратких описаний.
- Optical Character Recognition<br>
    В интерфейсе можно приложить картинку с текстом. На выходе можно скачать результат в виде файла txt

** https://huggingface.co/blog/smolvlm2

## Запуск проекта
Технические требования: Docker, Docker Compose v2. Для GPU: NVIDIA Docker runtime (nvidia-docker2)

1. **Склонировать репозиторий**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Build Docker Container**
```bash
docker compose build
```

Или напрямую через docker:
```bash
docker build -t multimodal-demo .
```

3. **Run Docker Container**<br>
Возможные параметры при запуске:
- DEVICE: cuda/cpu/mps/auto, default: cpu
- MODEL_SIZE - Размер модели: 256M, 500M, 2.2B (default: 256M)
- PORT - Порт приложения на хосте (default: 7860)
- HF_MODEL_ID - Явное указание модели на HuggingFace (опционально)

3.1. Как добавить существующие веса SmolVLM2:<br>
Модели автоматически сохраняются в директорию `./models` на хосте через Docker volume. При первом запуске модель будет загружена с HuggingFace и сохранена локально. При последующих запусках модель будет загружаться из локальной директории.

Если у вас уже есть веса модели в директории `./models/smolvlm2-256M` (или другого размера), они будут использованы автоматически:
```bash
docker compose up -d
```

Если веса находятся в другом месте, можно примонтировать их:
```bash
# В docker-compose.yml изменить:
# volumes:
#   - /путь/к/моделям:/app/models
```

3.2. Смена device GPU/CPU<br>
1. С использованием GPU: раскомментировать секцию `deploy` в `docker-compose.yml` и добавить параметр `DEVICE=cuda`<br>
    В `docker-compose.yml` раскомментировать:
    ```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ```
    Затем запустить:
    ```bash
    DEVICE=cuda docker compose up -d
    ```
2. CPU-only: использовать параметр `DEVICE=cpu` (по умолчанию)<br>
    ```bash
    DEVICE=cpu docker compose up -d
    ```
3. Apple Silicon (MPS): использовать параметр `DEVICE=mps`<br>
    ```bash
    DEVICE=mps docker compose up -d
    ```

3.3. Смена размера модели<br>
Задаем параметр MODEL_SIZE: 256M, 500M, 2.2B (по умолчанию 256M).
```bash
MODEL_SIZE=500M docker compose up -d
```

Или для модели 2.2B:
```bash
MODEL_SIZE=2.2B docker compose up -d
```

3.4. Смена порта<br>
Меняем порт через переменную окружения PORT:
```bash
PORT=8080 docker compose up -d
```

Или изменить в `docker-compose.yml`:
```yaml
ports:
  - "8080:7860"
```

3.5. Как примонтировать директорию с весами модели<br>
Директория с моделями автоматически монтируется через Docker volume в `docker-compose.yml`:
```yaml
volumes:
  - ./models:/app/models
```

Для ручного запуска контейнера без docker-compose:
```bash
docker run -d \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/temp_results:/app/temp_results \
  -e DEVICE=cpu \
  -e MODEL_SIZE=256M \
  multimodal-demo
```

4. Перейти на http://localhost:7860/ (либо выбранный порт)<br>
Нужно время, чтобы приложение запустилось - секунд 15-20 для загрузки модели (при первом запуске может потребоваться больше времени для скачивания модели с HuggingFace).

## Примечания
- При первом запуске приложение загрузит модель с HuggingFace (требуется интернет)
- После первой загрузки модель сохраняется локально в `./models` и работает без интернета
- Веса модели не хранятся в Docker образе, только в volume
- Для работы с GPU требуется соответствующий Docker runtime
- Генерация описания оптимизирована для скорости (ограничение до 80 токенов)
- Приложение работает только с моделью SmolVLM2
