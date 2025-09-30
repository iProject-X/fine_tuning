#!/usr/bin/env python3
"""
Главный скрипт для запуска всей системы Uzbek ASR
"""

import os
import sys
import asyncio
import argparse
import subprocess
from pathlib import Path

def print_banner():
    """Вывести баннер системы"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                     UZBEK WHISPER ASR SYSTEM                ║
║              Fine-tuning & Data Collection Platform         ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_requirements():
    """Проверить что все зависимости установлены"""
    print("🔍 Проверка зависимостей...")

    required_files = [
        ".env",
        "requirements.txt",
        "src/data/collectors/telegram_bot_collector.py",
        "src/models/training/distributed_trainer.py",
        "src/serving/api/fastapi_app.py"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Отсутствуют файлы:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # Проверить .env
    if not os.path.getsize(".env") > 10:
        print("❌ Файл .env пустой или не настроен")
        print("   Скопируйте .env.example в .env и настройте токены")
        return False

    print("✅ Все файлы на месте")
    return True

def install_dependencies():
    """Установить зависимости"""
    print("📦 Установка зависимостей...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True, capture_output=True)
        print("✅ Зависимости установлены")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        return False

def start_telegram_bot():
    """Запустить Telegram бота"""
    print("🤖 Запуск Telegram бота...")
    try:
        subprocess.Popen([sys.executable, "run_telegram_bot.py"])
        print("✅ Telegram бот запущен")
        return True
    except Exception as e:
        print(f"❌ Ошибка запуска бота: {e}")
        return False

def start_api_server():
    """Запустить API сервер"""
    print("🌐 Запуск API сервера...")
    try:
        # Проверить что FastAPI приложение существует
        if not os.path.exists("src/serving/api/fastapi_app.py"):
            print("⚠️  FastAPI приложение не найдено, пропускаем")
            return True

        subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.serving.api.fastapi_app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        print("✅ API сервер запущен на http://localhost:8000")
        return True
    except Exception as e:
        print(f"❌ Ошибка запуска API: {e}")
        return False

def start_monitoring():
    """Запустить мониторинг"""
    print("📊 Запуск мониторинга...")
    try:
        # Создать директории для логов
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/metrics", exist_ok=True)
        print("✅ Мониторинг настроен")
        return True
    except Exception as e:
        print(f"❌ Ошибка настройки мониторинга: {e}")
        return False

def show_status():
    """Показать статус системы"""
    print("\n" + "="*60)
    print("📊 СТАТУС СИСТЕМЫ")
    print("="*60)

    # Проверить процессы
    try:
        result = subprocess.run(["pgrep", "-f", "run_telegram_bot.py"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🤖 Telegram бот: ✅ Работает")
        else:
            print("🤖 Telegram бот: ❌ Остановлен")
    except:
        print("🤖 Telegram бот: ❓ Неизвестно")

    try:
        result = subprocess.run(["pgrep", "-f", "uvicorn"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🌐 API сервер: ✅ Работает")
        else:
            print("🌐 API сервер: ❌ Остановлен")
    except:
        print("🌐 API сервер: ❓ Неизвестно")

    # Проверить файлы
    if os.path.exists("uzbek_asr.db"):
        size = os.path.getsize("uzbek_asr.db")
        print(f"💾 База данных: ✅ {size} байт")
    else:
        print("💾 База данных: ❌ Не создана")

    if os.path.exists("logs"):
        log_files = len([f for f in os.listdir("logs") if f.endswith(".log")])
        print(f"📝 Логи: ✅ {log_files} файлов")
    else:
        print("📝 Логи: ❌ Директория не создана")

def stop_system():
    """Остановить всю систему"""
    print("🛑 Остановка системы...")

    try:
        # Остановить Telegram бота
        subprocess.run(["pkill", "-f", "run_telegram_bot.py"], capture_output=True)
        print("✅ Telegram бот остановлен")
    except:
        pass

    try:
        # Остановить API сервер
        subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
        print("✅ API сервер остановлен")
    except:
        pass

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Uzbek Whisper ASR System")
    parser.add_argument("action", choices=[
        "start", "stop", "status", "setup", "bot-only", "api-only", "test"
    ], help="Действие для выполнения")

    args = parser.parse_args()

    print_banner()

    if args.action == "setup":
        print("🔧 Настройка системы...")
        if not install_dependencies():
            sys.exit(1)
        print("✅ Система настроена")
        print("\n📝 Следующие шаги:")
        print("1. Настройте .env файл (скопируйте из .env.example)")
        print("2. Добавьте TELEGRAM_BOT_TOKEN")
        print("3. Запустите: python start_system.py start")

    elif args.action == "test":
        print("🧪 Тестирование компонентов...")
        try:
            subprocess.run([sys.executable, "test_bot.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Тесты не прошли")
            sys.exit(1)

    elif args.action == "start":
        if not check_requirements():
            print("\n💡 Запустите: python start_system.py setup")
            sys.exit(1)

        print("🚀 Запуск полной системы...")

        # Запустить компоненты
        start_monitoring()
        start_telegram_bot()
        start_api_server()

        print("\n" + "="*60)
        print("🎉 СИСТЕМА ЗАПУЩЕНА!")
        print("="*60)
        print("🤖 Telegram бот: Собирает данные")
        print("🌐 API сервер: http://localhost:8000")
        print("📊 Мониторинг: logs/")
        print("💾 База данных: uzbek_asr.db")
        print("\n📝 Для остановки: python start_system.py stop")
        print("📊 Для статуса: python start_system.py status")

        # Показать как использовать
        print("\n" + "="*60)
        print("📋 КАК ИСПОЛЬЗОВАТЬ:")
        print("="*60)
        print("1. 🤖 Telegram бот:")
        print("   - Пользователи отправляют голосовые сообщения")
        print("   - Пишут транскрипции")
        print("   - Проверяют качество")
        print()
        print("2. 🎯 Обучение модели:")
        print("   python -m src.models.training.distributed_trainer")
        print()
        print("3. 🔗 API для инференса:")
        print("   curl -X POST http://localhost:8000/transcribe \\")
        print("     -F \"audio=@audio.wav\"")

    elif args.action == "bot-only":
        if not check_requirements():
            sys.exit(1)
        print("🤖 Запуск только Telegram бота...")
        start_telegram_bot()
        print("✅ Бот запущен")

    elif args.action == "api-only":
        if not check_requirements():
            sys.exit(1)
        print("🌐 Запуск только API сервера...")
        start_api_server()
        print("✅ API запущен")

    elif args.action == "status":
        show_status()

    elif args.action == "stop":
        stop_system()
        print("✅ Система остановлена")

if __name__ == "__main__":
    main()