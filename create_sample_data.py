#!/usr/bin/env python3
"""
Создание примерных данных для обучения
"""

import os
import json
from pathlib import Path

def create_sample_data():
    """Создать примерные данные"""
    print("📁 Создание примерных данных...")

    # Создать директории
    os.makedirs("./data/audio_samples", exist_ok=True)
    os.makedirs("./data/manual_data", exist_ok=True)

    # Примерные узбекские фразы
    uzbek_samples = [
        "Assalomu alaykum, qalaysiz?",
        "Men O'zbekistondan kelganman",
        "Bugun ob-havo juda yaxshi",
        "Biz birga ishlashimiz kerak",
        "Men uzbek tilida gaplashishni yaxshi ko'raman",
        "Toshkent shahrida ko'p odamlar yashaydi",
        "O'qish juda muhim va foydali",
        "Kelajakda men dasturchi bo'lmoqchiman",
        "Oila bilan birga vaqt o'tkazish yoqimli",
        "Teknologiya hayotimizni osonlashtiradi"
    ]

    # Смешанные фразы (узбекский + русский)
    mixed_samples = [
        "Men bugun рынок ga boraman",
        "Привет, qalaysiz qarindosh?",
        "Siz компьютер bilan ishlaysizmi?",
        "Bu программа juda zo'r",
        "Давайте birga ishlaylik",
        "Интернет orqali aloqa qilamiz",
        "Машина uchun бензин kerak",
        "Университет da o'qiyapman"
    ]

    # Создать файлы для узбекских образцов
    for i, text in enumerate(uzbek_samples, 1):
        # Создать текстовый файл
        with open(f"./data/audio_samples/uzbek_{i:02d}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Создать заглушку аудио файла
        Path(f"./data/audio_samples/uzbek_{i:02d}.wav").touch()

        print(f"   ✅ uzbek_{i:02d}: {text}")

    # Создать файлы для смешанных образцов
    for i, text in enumerate(mixed_samples, 1):
        with open(f"./data/audio_samples/mixed_{i:02d}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        Path(f"./data/audio_samples/mixed_{i:02d}.wav").touch()

        print(f"   ✅ mixed_{i:02d}: {text}")

    # Создать JSON метаданные
    metadata = {
        "dataset_info": {
            "name": "Uzbek Speech Samples",
            "language": "uz",
            "mixed_language": True,
            "total_samples": len(uzbek_samples) + len(mixed_samples),
            "uzbek_only": len(uzbek_samples),
            "mixed_language_samples": len(mixed_samples)
        },
        "samples": []
    }

    # Добавить узбекские образцы
    for i, text in enumerate(uzbek_samples, 1):
        metadata["samples"].append({
            "id": f"uzbek_{i:02d}",
            "audio_file": f"uzbek_{i:02d}.wav",
            "text_file": f"uzbek_{i:02d}.txt",
            "transcription": text,
            "language": "uz",
            "type": "monolingual"
        })

    # Добавить смешанные образцы
    for i, text in enumerate(mixed_samples, 1):
        metadata["samples"].append({
            "id": f"mixed_{i:02d}",
            "audio_file": f"mixed_{i:02d}.wav",
            "text_file": f"mixed_{i:02d}.txt",
            "transcription": text,
            "language": "uz-ru",
            "type": "code_switching"
        })

    with open("./data/audio_samples/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Создано {len(uzbek_samples) + len(mixed_samples)} примеров")
    print(f"📁 Папка: ./data/audio_samples/")
    print(f"📄 Метаданные: ./data/audio_samples/metadata.json")

    print(f"\n💡 Для реального обучения замените .wav файлы на настоящие аудиозаписи")

def show_usage():
    """Показать примеры использования"""
    print(f"\n" + "="*60)
    print(f"📋 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ КОНСОЛЬНОГО ТРЕНЕРА")
    print(f"="*60)

    examples = [
        ("📱 Обучение на данных Telegram бота:", "python console_trainer.py --source telegram"),
        ("📁 Обучение на локальных файлах:", "python console_trainer.py --source files --data-dir ./data/audio_samples"),
        ("✍️  Интерактивное добавление данных:", "python console_trainer.py --source manual"),
        ("🔗 Комбинированное обучение:", "python console_trainer.py --source combined --data-dir ./data/audio_samples"),
        ("🎯 Продвинутые параметры:", "python console_trainer.py --source telegram --model-size medium --epochs 5"),
    ]

    for desc, cmd in examples:
        print(f"\n{desc}")
        print(f"   {cmd}")

    print(f"\n" + "="*60)
    print(f"📝 ПОДГОТОВКА ДАННЫХ")
    print(f"="*60)

    data_prep_steps = [
        "1. 📂 Создайте папку с аудио файлами",
        "2. 📝 Для каждого audio.wav создайте audio.txt с транскрипцией",
        "3. 🚀 Запустите: python console_trainer.py --source files --data-dir your_folder",
        "",
        "💡 Или используйте данные из Telegram бота:",
        "   python console_trainer.py --source telegram"
    ]

    for step in data_prep_steps:
        print(step)

if __name__ == "__main__":
    create_sample_data()
    show_usage()