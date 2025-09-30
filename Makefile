# Uzbek Whisper ASR System Makefile

.PHONY: help setup start stop status test train clean

help: ## Показать справку
	@echo "Uzbek Whisper ASR System"
	@echo "=========================="
	@echo ""
	@echo "Доступные команды:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Настроить систему (установить зависимости)
	@echo "🔧 Настройка системы..."
	python start_system.py setup

start: ## Запустить всю систему
	@echo "🚀 Запуск системы..."
	python start_system.py start

stop: ## Остановить систему
	@echo "🛑 Остановка системы..."
	python start_system.py stop

status: ## Показать статус системы
	@echo "📊 Проверка статуса..."
	python start_system.py status

test: ## Запустить тесты
	@echo "🧪 Запуск тестов..."
	python test_bot.py

start-bot: ## Запустить только Telegram бота
	@echo "🤖 Запуск только бота..."
	python start_system.py bot-only

start-api: ## Запустить только API сервер
	@echo "🌐 Запуск только API..."
	python start_system.py api-only

train: ## Обучить модель (базовые параметры)
	@echo "🎯 Запуск обучения..."
	python train_model.py --model-size small --epochs 3

install: ## Установить зависимости
	@echo "📦 Установка зависимостей..."
	pip install -r requirements.txt

clean: ## Очистить временные файлы
	@echo "🧹 Очистка..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f test_uzbek_asr.db

restart: stop start ## Перезапуск системы