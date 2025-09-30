#!/usr/bin/env python3
"""
Анализатор данных для Uzbek ASR
Анализирует качество и распределение собранных данных
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

class DataAnalyzer:
    """Анализатор данных ASR"""

    def __init__(self, db_path="uzbek_asr.db"):
        self.db_path = db_path

        print("📊 Анализатор данных Uzbek ASR")
        print("=" * 40)

    def load_data(self):
        """Загрузить данные из БД"""
        if not os.path.exists(self.db_path):
            print(f"❌ База данных {self.db_path} не найдена")
            return None

        conn = sqlite3.connect(self.db_path)

        # Загрузить данные о записях
        query = """
        SELECT
            id,
            user_id,
            file_path,
            duration,
            language_hint,
            transcription,
            quality_score,
            verification_count,
            verified,
            timestamp
        FROM audio_submissions
        WHERE transcription IS NOT NULL AND transcription != ''
        ORDER BY timestamp
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"📁 Загружено {len(df)} записей из базы данных")
        return df

    def analyze_quality_distribution(self, df):
        """Анализ распределения качества"""
        print("\n🎯 АНАЛИЗ КАЧЕСТВА ДАННЫХ")
        print("-" * 30)

        print(f"📊 Статистика quality_score:")
        print(f"   Среднее: {df['quality_score'].mean():.3f}")
        print(f"   Медиана: {df['quality_score'].median():.3f}")
        print(f"   Минимум: {df['quality_score'].min():.3f}")
        print(f"   Максимум: {df['quality_score'].max():.3f}")
        print(f"   Стандартное отклонение: {df['quality_score'].std():.3f}")

        # Категоризация по качеству
        df['quality_category'] = pd.cut(df['quality_score'],
                                       bins=[0, 0.3, 0.6, 0.8, 1.0],
                                       labels=['Низкое', 'Среднее', 'Хорошее', 'Отличное'])

        quality_counts = df['quality_category'].value_counts()
        print(f"\n📈 Распределение по категориям:")
        for category, count in quality_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")

        return df

    def analyze_content(self, df):
        """Анализ содержания"""
        print("\n📝 АНАЛИЗ СОДЕРЖАНИЯ")
        print("-" * 25)

        # Статистика по длительности
        print(f"⏱️  Длительность аудио:")
        print(f"   Средняя: {df['duration'].mean():.1f} сек")
        print(f"   Медиана: {df['duration'].median():.1f} сек")
        print(f"   Минимум: {df['duration'].min():.1f} сек")
        print(f"   Максимум: {df['duration'].max():.1f} сек")

        # Статистика по длине текста
        df['text_length'] = df['transcription'].str.len()
        print(f"\n📏 Длина транскрипций:")
        print(f"   Средняя: {df['text_length'].mean():.0f} символов")
        print(f"   Медиана: {df['text_length'].median():.0f} символов")
        print(f"   Минимум: {df['text_length'].min()} символов")
        print(f"   Максимум: {df['text_length'].max()} символов")

        # Анализ языков
        if 'language_hint' in df.columns:
            lang_counts = df['language_hint'].value_counts()
            print(f"\n🌐 Распределение по языкам:")
            for lang, count in lang_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {lang or 'Не указан'}: {count} ({percentage:.1f}%)")

        return df

    def show_best_samples(self, df, n=5):
        """Показать лучшие образцы"""
        print(f"\n⭐ ТОП-{n} ОБРАЗЦОВ ПО КАЧЕСТВУ")
        print("-" * 35)

        best_samples = df.nlargest(n, 'quality_score')

        for i, (_, row) in enumerate(best_samples.iterrows(), 1):
            print(f"{i}. Quality: {row['quality_score']:.3f} | "
                  f"Duration: {row['duration']:.1f}s | "
                  f"Lang: {row['language_hint'] or 'auto'}")
            print(f"   📝 {row['transcription']}")
            print()

    def show_worst_samples(self, df, n=3):
        """Показать худшие образцы"""
        print(f"\n⚠️  ОБРАЗЦЫ С НИЗКИМ КАЧЕСТВОМ (ТОП-{n})")
        print("-" * 40)

        worst_samples = df.nsmallest(n, 'quality_score')

        for i, (_, row) in enumerate(worst_samples.iterrows(), 1):
            print(f"{i}. Quality: {row['quality_score']:.3f} | "
                  f"Duration: {row['duration']:.1f}s | "
                  f"Lang: {row['language_hint'] or 'auto'}")
            print(f"   📝 {row['transcription']}")
            print()

    def recommend_training_strategy(self, df):
        """Рекомендации по стратегии обучения"""
        print("\n🎯 РЕКОМЕНДАЦИИ ПО ОБУЧЕНИЮ")
        print("-" * 35)

        total_samples = len(df)
        high_quality = len(df[df['quality_score'] >= 0.7])
        medium_quality = len(df[(df['quality_score'] >= 0.5) & (df['quality_score'] < 0.7)])
        low_quality = len(df[df['quality_score'] < 0.5])

        print(f"📊 Качество данных:")
        print(f"   Высокое качество (≥0.7): {high_quality} образцов")
        print(f"   Среднее качество (0.5-0.7): {medium_quality} образцов")
        print(f"   Низкое качество (<0.5): {low_quality} образцов")

        print(f"\n💡 Рекомендации:")

        if total_samples < 50:
            print("   📈 Продолжите сбор данных (рекомендуется 100+ образцов)")
        elif total_samples < 200:
            print("   ✅ Достаточно для начального обучения")
        else:
            print("   🎉 Отличное количество данных для качественного обучения")

        if high_quality < total_samples * 0.5:
            print("   🔧 Рассмотрите фильтрацию по качеству (quality_score >= 0.6)")

        if low_quality > total_samples * 0.3:
            print("   ⚠️  Много низкокачественных образцов - проверьте настройки микрофона")

        # Рекомендации по параметрам обучения
        print(f"\n⚙️  Параметры обучения:")
        if total_samples <= 50:
            print("   Epochs: 3-5, Batch size: 8, Learning rate: 1e-5")
        elif total_samples <= 200:
            print("   Epochs: 5-8, Batch size: 16, Learning rate: 5e-6")
        else:
            print("   Epochs: 8-15, Batch size: 32, Learning rate: 1e-6")

    def export_analysis(self, df):
        """Экспорт анализа"""
        print(f"\n💾 ЭКСПОРТ АНАЛИЗА")
        print("-" * 20)

        # Создать папку для экспорта
        export_dir = Path("./analysis_results")
        export_dir.mkdir(exist_ok=True)

        # Статистика
        stats = {
            "total_samples": len(df),
            "quality_stats": {
                "mean": float(df['quality_score'].mean()),
                "median": float(df['quality_score'].median()),
                "min": float(df['quality_score'].min()),
                "max": float(df['quality_score'].max()),
                "std": float(df['quality_score'].std())
            },
            "duration_stats": {
                "mean": float(df['duration'].mean()),
                "median": float(df['duration'].median()),
                "min": float(df['duration'].min()),
                "max": float(df['duration'].max()),
                "total_hours": float(df['duration'].sum() / 3600)
            },
            "quality_distribution": df['quality_category'].value_counts().to_dict(),
            "language_distribution": df['language_hint'].value_counts().to_dict() if 'language_hint' in df.columns else {},
            "best_samples": df.nlargest(5, 'quality_score')[['transcription', 'quality_score', 'duration']].to_dict('records'),
            "recommendations": self._get_recommendations(df)
        }

        # Сохранить JSON
        with open(export_dir / "analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        # Сохранить CSV
        df.to_csv(export_dir / "data_samples.csv", index=False, encoding="utf-8")

        print(f"   ✅ Отчет сохранен: {export_dir}/analysis_report.json")
        print(f"   ✅ Данные сохранены: {export_dir}/data_samples.csv")

    def _get_recommendations(self, df):
        """Получить рекомендации"""
        total = len(df)
        high_quality = len(df[df['quality_score'] >= 0.7])

        recommendations = []

        if total < 50:
            recommendations.append("Продолжите сбор данных")
        if high_quality < total * 0.5:
            recommendations.append("Улучшите качество записи")
        if df['duration'].mean() < 3:
            recommendations.append("Записывайте более длинные фразы")

        return recommendations

    def run_analysis(self):
        """Запустить полный анализ"""
        df = self.load_data()
        if df is None:
            return

        # Выполнить анализ
        df = self.analyze_quality_distribution(df)
        df = self.analyze_content(df)
        self.show_best_samples(df)
        self.show_worst_samples(df)
        self.recommend_training_strategy(df)
        self.export_analysis(df)

        print(f"\n🎉 Анализ завершен!")
        return df

def main():
    """Главная функция"""
    print("🔍 Запуск анализа данных Uzbek ASR...")

    analyzer = DataAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()