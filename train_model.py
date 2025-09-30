#!/usr/bin/env python3
"""
Скрипт для обучения Whisper модели на узбекских данных
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Главная функция для обучения"""
    parser = argparse.ArgumentParser(description="Train Uzbek Whisper Model")

    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to training data")
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./models/whisper-uz",
                       help="Output directory for trained model")

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  UZBEK WHISPER TRAINING                     ║
╚══════════════════════════════════════════════════════════════╝

📊 Параметры обучения:
   Model Size: {args.model_size}
   Epochs: {args.epochs}
   Batch Size: {args.batch_size}
   Learning Rate: {args.learning_rate}
   Data Path: {args.data_path}
   Output: {args.output_dir}

🚀 Запуск обучения...
""")

    try:
        # Импорт обучающего модуля
        from src.models.training.distributed_trainer import DistributedTrainer

        # Создать тренер
        trainer = DistributedTrainer(
            model_size=args.model_size,
            data_path=args.data_path,
            output_dir=args.output_dir
        )

        # Запустить обучение
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        print("✅ Обучение завершено успешно!")
        print(f"📁 Модель сохранена в: {args.output_dir}")

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь что все зависимости установлены:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()