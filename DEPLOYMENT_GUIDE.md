# 🚀 Uzbek Whisper ASR - Production Deployment Guide

## 📋 Обзор системы

Enterprise-grade система автоматического распознавания речи для узбекского языка с поддержкой code-switching (узбекско-русского смешения).

### 🎯 Ключевые возможности

- **Multilingual ASR**: Поддержка чистого узбекского, русского и смешанной речи
- **Code-Switch Detection**: Автоматическое определение переключений между языками
- **Real-time Streaming**: WebSocket API для потоковой транскрипции
- **Production Ready**: Kubernetes deployment, автомасштабирование
- **High Performance**: Quantization, ONNX export, GPU optimization

## 🏗️ Архитектура

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

## 🚀 Быстрый старт

### 1. Предварительные требования

```bash
# Python 3.9+
python3 --version

# CUDA (для GPU)
nvidia-smi

# Docker & Kubernetes
docker --version
kubectl version --client
```

### 2. Установка

```bash
# Клонировать репозиторий
git clone <your-repo>
cd uzbek-whisper

# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
make install-dev

# Тест установки
python test_setup.py
```

### 3. Локальная разработка

```bash
# Запуск API сервера
make serve

# Запуск в debug режиме
make serve-debug

# Jupyter notebook
make jupyter

# Tensorboard
make tensorboard
```

## 📊 Сбор и подготовка данных

### Telegram бот для краудсорсинга

```bash
# Настроить токен бота
export TELEGRAM_BOT_TOKEN="your_bot_token"

# Запустить сбор данных
make collect-data
```

### Обработка аудио данных

```bash
# Обработать raw данные
make process-data

# Валидация качества
make validate-data

# Аугментация данных
make augment-data
```

## 🎯 Обучение модели

### Локальное обучение

```bash
# Обучение с конфигурацией по умолчанию
make train

# Debug обучение
make train-debug

# Distributed обучение на нескольких GPU
make train-distributed
```

### Конфигурация обучения

Отредактируйте `configs/training_config.yaml`:

```yaml
model:
  base_model: "openai/whisper-base"
  languages: ["uz", "ru", "mixed"]
  adapter_dim: 256
  freeze_base_model: true

training:
  batch_size: 16
  learning_rate: 5e-5
  num_epochs: 10
  mixed_precision: "fp16"

curriculum_learning:
  enabled: true
  stages:
    - name: "clean_audio"
      epochs: 5
    - name: "noisy_audio"
      epochs: 3
    - name: "code_switch"
      epochs: 5
```

### Мониторинг обучения

- **Wandb**: https://wandb.ai/your-project
- **TensorBoard**: `make tensorboard`
- **Логи**: `tail -f outputs/training.log`

## 🔍 Оценка модели

```bash
# Базовая оценка
make evaluate

# Подробная оценка по языкам
make evaluate-all

# Benchmark производительности
make benchmark
```

### Метрики качества

- **WER**: < 25% для узбекского, < 30% для code-switched
- **CER**: < 15% общий
- **Language Detection**: > 85% accuracy
- **Code-Switch F1**: > 80%

## 🐳 Docker развертывание

### Локальный Docker

```bash
# Сборка образа
make docker-build

# Запуск контейнера
make docker-run

# Push в registry
make docker-push
```

### Docker Compose

```bash
# Полный стек (API + Redis + PostgreSQL)
docker-compose up -d

# Только API сервис
docker-compose up api
```

## ☸️ Kubernetes развертывание

### Staging окружение

```bash
# Деплой в staging
make deploy-staging

# Проверка статуса
kubectl get pods -n uzbek-whisper-staging

# Логи
kubectl logs -f deployment/uzbek-whisper-asr -n uzbek-whisper-staging
```

### Production окружение

```bash
# Деплой в production
make deploy-prod

# Автомасштабирование
kubectl get hpa -n uzbek-whisper-production

# Мониторинг
make monitoring
```

### Конфигурация ресурсов

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: "1"

autoscaling:
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
```

## 🔧 API использование

### REST API

```python
import requests

# Транскрипция файла
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'https://asr.example.com/transcribe',
        files={'file': f},
        data={'language_hint': 'mixed'}
    )

result = response.json()
print(f"Text: {result['text']}")
print(f"Language: {result['language_detected']}")
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('wss://asr.example.com/ws/transcribe');

ws.onopen = () => {
    // Отправка аудио chunks
    navigator.mediaDevices.getUserMedia({audio: true})
        .then(stream => {
            // Stream audio to WebSocket
        });
};

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log('Interim result:', result.text);
};
```

### Batch обработка

```bash
curl -X POST "https://asr.example.com/transcribe/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

## 📈 Мониторинг и алерты

### Prometheus метрики

- `asr_requests_total`: Общее количество запросов
- `asr_request_duration_seconds`: Время обработки
- `asr_wer_score`: Текущий WER
- `asr_active_connections`: Активные WebSocket соединения

### Grafana дashboards

```bash
# Импорт дашбордов
kubectl apply -f infrastructure/monitoring/grafana/dashboards/
```

### Алерты

- High WER (> 30%)
- High latency (> 200ms)
- High error rate (> 1%)
- GPU memory usage (> 90%)

## 🚦 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/ci-cd.yml
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:         # Unit/integration тесты
  train:        # Model обучение и валидация
  build:        # Docker образы
  security:     # Security scanning
  deploy:       # Kubernetes deployment
```

### Автоматический pipeline

1. **Code push** → Автоматические тесты
2. **Tests pass** → Model training и validation
3. **Quality gates** → Docker build
4. **Security scan** → Deployment to staging
5. **Manual approval** → Production deployment

## 🔐 Безопасность

### API аутентификация

```python
# JWT токены
headers = {
    'Authorization': 'Bearer YOUR_JWT_TOKEN'
}
```

### Network политики

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: uzbek-whisper-network-policy
spec:
  podSelector:
    matchLabels:
      app: uzbek-whisper-asr
  policyTypes:
  - Ingress
  - Egress
```

### Secrets управление

```bash
# Kubernetes secrets
kubectl create secret generic uzbek-whisper-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=database-url=your-db-url
```

## 📊 Performance tuning

### Model optimization

```bash
# ONNX export
make export-model

# Quantization
make optimize-model

# TensorRT compilation
python scripts/tensorrt_compile.py
```

### Infrastructure optimization

- **GPU**: NVIDIA A100, V100, RTX 3090
- **Memory**: 8GB+ RAM per replica
- **Storage**: NVMe SSD для model storage
- **Network**: 10Gbps+ для streaming

## 🐛 Troubleshooting

### Распространенные проблемы

```bash
# OOM errors
kubectl describe pod <pod-name>
# Увеличить memory limits

# Slow inference
# Проверить GPU utilization
nvidia-smi

# High WER
# Проверить quality входных данных
python scripts/analyze_audio_quality.py
```

### Логи и debug

```bash
# Детальные логи
kubectl logs -f deployment/uzbek-whisper-asr --previous

# Debug mode
export LOG_LEVEL=DEBUG
make serve-debug

# Профилирование
py-spy top --pid <process-id>
```

## 📞 Поддержка

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Community**: Telegram @uzbek_whisper_support
- **Email**: support@your-domain.com

## 📜 Лицензия

MIT License - см. [LICENSE](LICENSE) файл

---

## 🎯 Roadmap

### Q1 2024
- [ ] Multi-speaker diarization
- [ ] Streaming optimizations
- [ ] Mobile SDK

### Q2 2024
- [ ] Kazakh language support
- [ ] Real-time translation
- [ ] Edge deployment

---

**🚀 Ваша система готова к production использованию!**