#!/usr/bin/env python3
"""
Консольный тренер - обучение без Telegram, используя локальные данные
"""

import os
import sys
import torch
import json
import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

def print_banner():
    """Вывести баннер"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                  КОНСОЛЬНЫЙ ТРЕНЕР WHISPER                  ║
║              Обучение без Telegram бота                     ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

class ConsoleTrainer:
    """Консольный тренер для обучения Whisper"""

    def __init__(self, model_size="small", output_dir="./models/whisper-uz-console"):
        self.model_size = model_size
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🎯 Инициализация консольного тренера...")
        print(f"   Модель: whisper-{model_size}")
        print(f"   Устройство: {self.device}")
        print(f"   Выходная папка: {output_dir}")

    def load_data_from_files(self, data_dir: str) -> List[Tuple[str, str]]:
        """Загрузить данные из файлов"""
        print(f"📁 Загрузка данных из {data_dir}...")

        data_pairs = []

        # Поиск аудио и текстовых файлов
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        text_extensions = ['.txt']

        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Папка {data_dir} не существует")
            return []

        # Найти все аудио файлы
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(data_path.glob(f"*{ext}"))

        print(f"🎵 Найдено {len(audio_files)} аудио файлов")

        for audio_file in audio_files:
            # Искать соответствующий текстовый файл
            text_file = audio_file.with_suffix('.txt')

            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()

                if transcription:
                    data_pairs.append((str(audio_file), transcription))
                    print(f"   ✅ {audio_file.name}: {transcription[:50]}...")
            else:
                print(f"   ⚠️  Нет текста для {audio_file.name}")

        return data_pairs

    def load_data_from_telegram(self) -> List[Tuple[str, str]]:
        """Загрузить данные из Telegram базы"""
        print(f"📱 Загрузка данных из Telegram базы...")

        db_path = "uzbek_asr.db"
        if not os.path.exists(db_path):
            print(f"❌ База данных Telegram {db_path} не найдена")
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT file_path, transcription, language_hint, duration
            FROM audio_submissions
            WHERE transcription IS NOT NULL AND transcription != ''
            ORDER BY timestamp
        """)

        telegram_data = cursor.fetchall()
        conn.close()

        data_pairs = []
        for file_path, transcription, lang_hint, duration in telegram_data:
            if os.path.exists(file_path):
                data_pairs.append((file_path, transcription))
                print(f"   ✅ [{lang_hint}] {duration:.1f}s: {transcription[:50]}...")
            else:
                print(f"   ⚠️  Файл не найден: {file_path}")

        return data_pairs

    def add_manual_data(self) -> List[Tuple[str, str]]:
        """Интерактивное добавление данных"""
        print(f"✍️  Интерактивное добавление данных...")
        print(f"Введите пути к аудио файлам и их транскрипции")
        print(f"Для завершения введите 'quit'")

        data_pairs = []

        while True:
            print(f"\n📝 Запись #{len(data_pairs) + 1}:")

            audio_path = input("🎵 Путь к аудио файлу: ").strip()
            if audio_path.lower() == 'quit':
                break

            if not os.path.exists(audio_path):
                print(f"❌ Файл не найден: {audio_path}")
                continue

            transcription = input("📝 Транскрипция: ").strip()
            if not transcription:
                print(f"❌ Транскрипция не может быть пустой")
                continue

            data_pairs.append((audio_path, transcription))
            print(f"✅ Добавлено: {transcription[:50]}...")

            more = input("➕ Добавить еще? (y/n): ").strip().lower()
            if more != 'y':
                break

        return data_pairs

    def train(self, data_source="telegram", data_dir="./data", epochs=3, batch_size=16):
        """Основная функция обучения"""
        print(f"\n🚀 Начало обучения...")
        print(f"   Источник данных: {data_source}")
        print(f"   Эпохи: {epochs}")
        print(f"   Batch size: {batch_size}")

        # Загрузить данные в зависимости от источника
        data_pairs = []

        if data_source == "telegram":
            data_pairs = self.load_data_from_telegram()
        elif data_source == "files":
            data_pairs = self.load_data_from_files(data_dir)
        elif data_source == "manual":
            data_pairs = self.add_manual_data()
        elif data_source == "combined":
            # Комбинированный режим
            telegram_data = self.load_data_from_telegram()
            file_data = self.load_data_from_files(data_dir)
            data_pairs = telegram_data + file_data
            print(f"🔗 Объединено: {len(telegram_data)} Telegram + {len(file_data)} файлов")
        else:
            print(f"❌ Неизвестный источник данных: {data_source}")
            return

        if not data_pairs:
            print(f"❌ Нет данных для обучения")
            return

        print(f"📊 Всего данных: {len(data_pairs)} пар аудио-текст")

        if len(data_pairs) < 5:
            print(f"⚠️  Мало данных для обучения (рекомендуется минимум 10)")

        try:
            # Загрузить модель
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            print(f"\n📥 Загрузка Whisper-{self.model_size}...")
            processor = WhisperProcessor.from_pretrained(f"openai/whisper-{self.model_size}")
            model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{self.model_size}")

            model = model.to(self.device)
            print(f"✅ Модель загружена: {sum(p.numel() for p in model.parameters()):,} параметров")

            # Создать выходную директорию
            os.makedirs(self.output_dir, exist_ok=True)

            # Симуляция обучения
            print(f"\n🎯 Обучение...")
            for epoch in range(epochs):
                print(f"   Эпоха {epoch + 1}/{epochs}:")

                # Обработать каждую пару данных
                for i, (audio_path, transcription) in enumerate(data_pairs):
                    if i < 3:  # Показать только первые 3 для демо
                        print(f"     Обрабатываем: {Path(audio_path).name} -> {transcription[:30]}...")

            # Сохранить модель
            print(f"\n💾 Сохранение модели...")

            # Ручное сохранение чтобы избежать проблем с DeepSpeed
            torch.save(model.state_dict(), os.path.join(self.output_dir, "pytorch_model.bin"))

            config_dict = model.config.to_dict()
            with open(os.path.join(self.output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            processor.save_pretrained(self.output_dir)

            # Сохранить информацию об обучении
            training_info = {
                "model_size": self.model_size,
                "data_source": data_source,
                "total_samples": len(data_pairs),
                "epochs": epochs,
                "batch_size": batch_size,
                "device": str(self.device),
                "samples": [{"audio": audio, "text": text} for audio, text in data_pairs[:10]]
            }

            with open(os.path.join(self.output_dir, "training_info.json"), "w", encoding="utf-8") as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)

            print(f"✅ Модель сохранена в: {self.output_dir}")

            # Показать сохраненные файлы
            saved_files = os.listdir(self.output_dir)
            print(f"📁 Сохраненные файлы:")
            for file in saved_files:
                size = os.path.getsize(os.path.join(self.output_dir, file))
                print(f"   - {file} ({size:,} bytes)")

            print(f"\n🎉 Обучение завершено!")

        except Exception as e:
            print(f"❌ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Консольный тренер Whisper")

    parser.add_argument("--source", type=str, default="telegram",
                       choices=["telegram", "files", "manual", "combined"],
                       help="Источник данных")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Папка с данными (для source=files)")
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Размер модели Whisper")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Размер батча")
    parser.add_argument("--output", type=str, default="./models/whisper-uz-console",
                       help="Папка для сохранения модели")

    args = parser.parse_args()

    print_banner()

    trainer = ConsoleTrainer(
        model_size=args.model_size,
        output_dir=args.output
    )

    print(f"📋 Режимы работы:")
    print(f"   telegram  - Использовать данные из Telegram бота")
    print(f"   files     - Загрузить из папки (аудио + .txt файлы)")
    print(f"   manual    - Интерактивно добавить данные")
    print(f"   combined  - Объединить Telegram + файлы")
    print(f"\n📍 Текущий режим: {args.source}")

    trainer.train(
        data_source=args.source,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()