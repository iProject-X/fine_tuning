# Uzbek Whisper ASR System

Enterprise-grade автоматическое распознавание речи для узбекского языка с поддержкой code-switching (узбекско-русского смешения).

## 🚀 Ключевые возможности

- **Multilingual ASR**: Поддержка чистого узбекского, русского и смешанной речи
- **Code-Switch Detection**: Интеллектуальное определение переключений языков
- **Real-time Streaming**: WebSocket API для потоковой транскрипции
- **Production Ready**: Kubernetes deployment, автомасштабирование, мониторинг
- **High Performance**: Quantization, ONNX export, TensorRT оптимизация
- **Crowdsourcing**: Telegram бот для сбора данных

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
├─────────────────────────────────────────────────────────────┤
│                      API Gateway (Kong/Nginx)                │
├─────────────────────────────────────────────────────────────┤
│                    Load Balancer (HAProxy)                   │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  ASR Service │  ASR Service │  ASR Service │   Queue       │
│  Instance 1  │  Instance 2  │  Instance 3  │  (RabbitMQ)   │
├──────────────┴──────────────┴──────────────┴───────────────┤
│                    Model Serving Layer                       │
│            (Triton Inference Server / TorchServe)           │
├─────────────────────────────────────────────────────────────┤
│                    Storage & Caching                         │
│         MinIO (Audio) | Redis (Cache) | PostgreSQL          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Структура проекта

```
uzbek-whisper/
├── src/                          # Исходный код
│   ├── data/                    # Обработка данных
│   ├── models/                  # ML модели
│   ├── serving/                 # API сервисы
│   ├── evaluation/              # Метрики и тесты
│   └── utils/                   # Утилиты
├── infrastructure/              # Инфраструктура
│   ├── docker/                 # Docker контейнеры
│   ├── kubernetes/             # K8s манифесты
│   └── terraform/              # Infrastructure as Code
├── configs/                     # Конфигурации
├── scripts/                     # Скрипты автоматизации
├── tests/                       # Тесты
├── data/                        # Датасеты
└── models/                      # Готовые модели
```

## ⚡ Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
cp configs/config.example.yaml configs/config.yaml
# Отредактируйте config.yaml под ваши нужды
```

### 3. Сбор данных через Telegram бот

```bash
python scripts/run_telegram_collector.py --token YOUR_BOT_TOKEN
```

### 4. Обучение модели

```bash
python scripts/train_model.py --config configs/training_config.yaml
```

### 5. Запуск API сервиса

```bash
python src/serving/api/fastapi_app.py
```

## 🎯 Технические характеристики

- **WER**: < 25% для узбекского, < 30% для code-switched
- **Latency**: P95 < 100ms
- **Throughput**: > 1000 RPS
- **Languages**: Узбекский (uz), Русский (ru), Mixed
- **Audio**: 16kHz WAV/MP3/OGG
- **Deployment**: Kubernetes, Docker, Cloud-native

## 📊 Поддерживаемые диалекты

- **Ташкентский**: Стандартный литературный узбекский
- **Ферганский**: Восточный диалект с характерными фонетическими особенностями
- **Хорезмский**: Западный диалект с архаичными формами
- **Смешанный**: Code-switching узбекский-русский

## 🚀 Production Deployment

### Kubernetes

```bash
kubectl apply -f infrastructure/kubernetes/
```

### Docker Compose

```bash
docker-compose -f infrastructure/docker/docker-compose.yml up
```

## 📈 Мониторинг

- **Prometheus**: Метрики производительности
- **Grafana**: Dashboards и визуализация
- **ELK Stack**: Логирование и анализ
- **Jaeger**: Distributed tracing

## 🔬 Оценка качества

```bash
python scripts/evaluate_model.py \
  --model-path models/latest \
  --test-data data/test_manifest.json
```

## 🤝 Участие в разработке

1. Fork репозиторий
2. Создайте feature branch
3. Добавьте тесты
4. Убедитесь что CI проходит
5. Создайте Pull Request

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE) файл

## 📞 Поддержка

- **Issues**: GitHub Issues
- **Documentation**: [docs/](docs/)
- **Community**: Telegram @uzbek_whisper