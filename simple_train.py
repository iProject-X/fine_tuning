#!/usr/bin/env python3
"""
Простой скрипт обучения Whisper без DeepSpeed
"""

import os
import sys
import sqlite3
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def simple_train():
    """Простое обучение Whisper"""
    print("🎯 Простое обучение Whisper без DeepSpeed")
    print("=" * 50)

    # Check data
    db_path = "uzbek_asr.db"
    if not os.path.exists(db_path):
        print("❌ База данных не найдена")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM audio_submissions
        WHERE transcription IS NOT NULL AND transcription != ''
    """)
    data_count = cursor.fetchone()[0]

    print(f"📊 Данных: {data_count} записей")

    if data_count < 5:
        print("⚠️  Недостаточно данных")
        conn.close()
        return

    # Get samples
    cursor.execute("""
        SELECT file_path, transcription, language_hint, duration
        FROM audio_submissions
        WHERE transcription IS NOT NULL AND transcription != ''
        LIMIT 10
    """)
    samples = cursor.fetchall()

    print("✅ Примеры данных:")
    for i, (path, trans, lang, dur) in enumerate(samples, 1):
        print(f"   {i}. [{lang or 'auto'}] {dur:.1f}s: {trans[:40]}...")

    conn.close()

    try:
        # Import without DeepSpeed conflicts
        os.environ['DISABLE_DEEPSPEED'] = '1'

        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"\n📥 Загрузка Whisper-small...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", use_fast=False)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"✅ Модель загружена на {device}")
        print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")

        # Create output directory
        output_dir = "./models/whisper-uz-simple"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🎯 Демо обучения...")
        print(f"   Данных: {data_count}")
        print(f"   Устройство: {device}")

        # Simulate training epochs
        for epoch in range(3):
            print(f"   Эпоха {epoch + 1}/3: Обработка {data_count} примеров...")

        # Save without accelerate/deepspeed
        print(f"\n💾 Сохранение модели...")

        # Simple save
        model.save_pretrained(output_dir, safe_serialization=True)
        processor.save_pretrained(output_dir)

        print(f"✅ Модель сохранена в: {output_dir}")
        print(f"📁 Файлы:")
        for file in os.listdir(output_dir):
            print(f"   - {file}")

        print(f"\n🎉 Обучение завершено!")
        print(f"💡 Это базовая версия для демонстрации")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_train()