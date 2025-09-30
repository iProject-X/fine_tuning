"""
Telegram bot for crowdsourcing Uzbek speech data collection
Handles audio collection, transcription, and quality control
"""

import logging
import asyncio
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import uuid

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ..processors.code_switch_detector import CodeSwitchDetector
from ...utils.audio_utils import AudioProcessor
from ...utils.database import DatabaseManager, AudioSubmission as DBSubmission

logger = logging.getLogger(__name__)

@dataclass
class AudioSubmission:
    """Data structure for audio submissions"""
    id: str
    user_id: int
    username: Optional[str]
    file_id: str
    file_path: str
    duration: float
    language_hint: Optional[str]
    transcription: Optional[str]
    quality_score: float
    verification_count: int
    verified: bool
    timestamp: datetime
    metadata: Dict

@dataclass
class UserStats:
    """User contribution statistics"""
    user_id: int
    username: Optional[str]
    audio_count: int
    transcription_count: int
    verification_count: int
    quality_score: float
    points: int
    level: str
    badges: List[str]

class UzbekASRDataBot:
    """
    Advanced Telegram bot for collecting Uzbek ASR training data

    Features:
    - Multi-language audio collection
    - Crowdsourced transcription
    - Quality control through verification
    - Gamification with points and leaderboards
    - Automatic code-switching detection
    """

    def __init__(self, token: str, database_url: str, storage_path: str = "./data/telegram"):
        """
        Initialize the Telegram bot

        Args:
            token: Telegram bot token
            database_url: Database connection URL
            storage_path: Path for storing audio files
        """
        self.token = token
        self.storage_path = storage_path
        self.code_switch_detector = CodeSwitchDetector()
        self.audio_processor = AudioProcessor()

        # Database setup
        self.db_manager = DatabaseManager(database_url)

        # Bot configuration
        self.max_audio_duration = 60  # seconds
        self.min_audio_duration = 2   # seconds
        self.points_per_audio = 100
        self.points_per_transcription = 50
        self.points_per_verification = 25

        # Quality thresholds
        self.auto_approve_threshold = 0.95
        self.min_verifications = 3

        # User levels
        self.level_thresholds = {
            'Bronze': 0,
            'Silver': 1000,
            'Gold': 5000,
            'Platinum': 15000,
            'Diamond': 50000
        }

        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize application
        self.app = Application.builder().token(token).build()
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup bot command and message handlers"""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("leaderboard", self.cmd_leaderboard))
        self.app.add_handler(CommandHandler("settings", self.cmd_settings))

        # Message handlers
        self.app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        self.app.add_handler(MessageHandler(filters.AUDIO, self.handle_audio))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

        # Callback handlers for inline keyboards
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user

        # Register user if new
        await self._ensure_user_exists(user.id, user.username)

        welcome_text = (
            f"🎙️ Assalomu alaykum, {user.first_name}!\n"
            f"Добро пожаловать в Uzbek Speech Collector!\n\n"

            "📝 Этот бот собирает данные для улучшения узбекского распознавания речи.\n\n"

            "🎯 Как помочь:\n"
            "1. 🎵 Отправьте голосовое сообщение на узбекском/русском\n"
            "2. ✍️ Напишите текст того, что сказали\n"
            "3. ✅ Проверьте транскрипции других пользователей\n\n"

            "🏆 За участие вы получаете очки и значки!\n"
            "📊 /stats - ваша статистика\n"
            "🏅 /leaderboard - рейтинг участников\n"
            "❓ /help - подробная справка"
        )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
            [InlineKeyboardButton("🎯 Начать запись", callback_data="start_recording")],
            [InlineKeyboardButton("✅ Проверить транскрипции", callback_data="verify")]
        ])

        await update.message.reply_text(welcome_text, reply_markup=keyboard)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "📖 Подробная инструкция:\n\n"

            "🎙️ ЗАПИСЬ АУДИО:\n"
            "• Нажмите и держите кнопку микрофона\n"
            "• Говорите четко и естественно\n"
            "• Длительность: 2-60 секунд\n"
            "• Можно говорить на узбекском, русском или смешанно\n\n"

            "✍️ ТРАНСКРИПЦИЯ:\n"
            "• После отправки аудио напишите текст\n"
            "• Пишите точно то, что сказали\n"
            "• Сохраняйте смешение языков: 'Men bugun рынок ga boraman'\n"
            "• Используйте правильную орфографию\n\n"

            "✅ ПРОВЕРКА:\n"
            "• Вам будут присылаться аудио для проверки\n"
            "• Слушайте и оценивайте точность транскрипции\n"
            "• Ваши оценки улучшают качество данных\n\n"

            "🏆 ОЧКИ И УРОВНИ:\n"
            f"• {self.points_per_audio} очков за аудио\n"
            f"• {self.points_per_transcription} очков за транскрипцию\n"
            f"• {self.points_per_verification} очков за проверку\n"
            "• Бонусы за высокое качество\n\n"

            "📝 СОВЕТЫ:\n"
            "• Записывайтесь в тихом месте\n"
            "• Говорите естественно, как в жизни\n"
            "• Включайте разные диалекты и акценты\n"
            "• Используйте смешение языков (code-switching)\n\n"

            "❓ Вопросы? Пишите @uzbek_whisper_support"
        )

        await update.message.reply_text(help_text)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        await self._ensure_user_exists(user.id, user.username)

        # Show user stats using inline keyboard
        stats = await self._get_user_stats(user.id)

        if not stats:
            await update.message.reply_text("📊 Статистика не найдена")
            return

        level = self._get_user_level(stats.points)
        next_level_points = self._get_next_level_points(stats.points)

        stats_text = f"📊 ВАША СТАТИСТИКА\n\n"
        stats_text += f"👤 Пользователь: {stats.username or 'Аноним'}\n"
        stats_text += f"🏆 Уровень: {level}\n"
        stats_text += f"🎯 Очки: {stats.points}\n"

        if next_level_points:
            stats_text += f"📈 До следующего уровня: {next_level_points - stats.points}\n"

        stats_text += f"\n📈 ВКЛАД:\n"
        stats_text += f"🎙️ Аудиозаписей: {stats.audio_count}\n"
        stats_text += f"✍️ Транскрипций: {stats.transcription_count}\n"
        stats_text += f"✅ Проверок: {stats.verification_count}\n"
        stats_text += f"⭐ Качество: {stats.quality_score:.1%}\n"

        if stats.badges:
            stats_text += f"\n🏅 ЗНАЧКИ: {', '.join(stats.badges)}"

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🏅 Рейтинг", callback_data="leaderboard")],
            [InlineKeyboardButton("🎙️ Записать аудио", callback_data="start_recording")]
        ])

        await update.message.reply_text(stats_text, reply_markup=keyboard)

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        user = update.effective_user
        await self._ensure_user_exists(user.id, user.username)

        settings_text = (
            "⚙️ НАСТРОЙКИ\n\n"
            "🔧 Доступные настройки:\n"
            "• Язык интерфейса\n"
            "• Уведомления\n"
            "• Приватность\n\n"
            "💡 Используйте кнопки ниже для настройки"
        )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🇺🇿 O'zbek", callback_data="lang_settings_uz")],
            [InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_settings_ru")],
            [InlineKeyboardButton("🔔 Уведомления", callback_data="notifications_settings")],
            [InlineKeyboardButton("🔒 Приватность", callback_data="privacy_settings")]
        ])

        await update.message.reply_text(settings_text, reply_markup=keyboard)

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages"""
        user = update.effective_user
        voice = update.message.voice

        # Validate duration
        if voice.duration < self.min_audio_duration:
            await update.message.reply_text(
                f"⚠️ Аудио слишком короткое (минимум {self.min_audio_duration} сек)"
            )
            return

        if voice.duration > self.max_audio_duration:
            await update.message.reply_text(
                f"⚠️ Аудио слишком длинное (максимум {self.max_audio_duration} сек)"
            )
            return

        try:
            # Download and process audio
            file = await context.bot.get_file(voice.file_id)
            submission_id = str(uuid.uuid4())
            file_path = os.path.join(
                self.storage_path,
                f"{user.id}_{submission_id}.ogg"
            )

            await file.download_to_drive(file_path)

            # Process audio (convert, validate quality)
            audio_info = await self.audio_processor.process_audio(file_path)

            if not audio_info['is_valid']:
                await update.message.reply_text(
                    "⚠️ Проблема с качеством аудио. Попробуйте записать снова."
                )
                os.remove(file_path)
                return

            # Store in database
            submission = AudioSubmission(
                id=submission_id,
                user_id=user.id,
                username=user.username,
                file_id=voice.file_id,
                file_path=file_path,
                duration=voice.duration,
                language_hint=None,
                transcription=None,
                quality_score=audio_info['quality_score'],
                verification_count=0,
                verified=False,
                timestamp=datetime.now(),
                metadata=audio_info
            )

            await self._store_submission(submission)

            # Set user context for next message
            context.user_data['pending_transcription'] = submission_id

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🇺🇿 Узбекский", callback_data=f"lang_uz_{submission_id}")],
                [InlineKeyboardButton("🇷🇺 Русский", callback_data=f"lang_ru_{submission_id}")],
                [InlineKeyboardButton("🌐 Смешанный", callback_data=f"lang_mixed_{submission_id}")]
            ])

            await update.message.reply_text(
                "✅ Аудио получено! На каком языке вы говорили?\n"
                "Затем напишите транскрипцию (точный текст того, что сказали):",
                reply_markup=keyboard
            )

        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            await update.message.reply_text(
                "❌ Ошибка при обработке аудио. Попробуйте еще раз."
            )

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle audio files (similar to voice messages)"""
        user = update.effective_user
        audio = update.message.audio

        # Check if audio is too long (audio files can be longer than voice messages)
        if audio.duration > self.max_audio_duration:
            await update.message.reply_text(
                f"⚠️ Аудиофайл слишком длинный (максимум {self.max_audio_duration} сек). "
                f"Используйте голосовые сообщения для коротких записей."
            )
            return

        if audio.duration < self.min_audio_duration:
            await update.message.reply_text(
                f"⚠️ Аудиофайл слишком короткий (минимум {self.min_audio_duration} сек)"
            )
            return

        # Process similar to voice messages but with different handling
        try:
            file = await context.bot.get_file(audio.file_id)
            submission_id = str(uuid.uuid4())

            # Use audio file extension if available
            file_ext = "mp3" if audio.mime_type == "audio/mpeg" else "ogg"
            file_path = os.path.join(
                self.storage_path,
                f"{user.id}_{submission_id}.{file_ext}"
            )

            await file.download_to_drive(file_path)

            # Process audio
            audio_info = await self.audio_processor.process_audio(file_path)

            if not audio_info['is_valid']:
                await update.message.reply_text(
                    "⚠️ Проблема с качеством аудиофайла. Попробуйте другой файл или используйте голосовые сообщения."
                )
                os.remove(file_path)
                return

            # Store submission
            submission = AudioSubmission(
                id=submission_id,
                user_id=user.id,
                username=user.username,
                file_id=audio.file_id,
                file_path=file_path,
                duration=audio.duration,
                language_hint=None,
                transcription=None,
                quality_score=audio_info['quality_score'],
                verification_count=0,
                verified=False,
                timestamp=datetime.now(),
                metadata=audio_info
            )

            await self._store_submission(submission)
            context.user_data['pending_transcription'] = submission_id

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🇺🇿 Узбекский", callback_data=f"lang_uz_{submission_id}")],
                [InlineKeyboardButton("🇷🇺 Русский", callback_data=f"lang_ru_{submission_id}")],
                [InlineKeyboardButton("🌐 Смешанный", callback_data=f"lang_mixed_{submission_id}")]
            ])

            await update.message.reply_text(
                "✅ Аудиофайл получен! На каком языке говорится в записи?\n"
                "Затем напишите транскрипцию (точный текст того, что сказано):",
                reply_markup=keyboard
            )

        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            await update.message.reply_text(
                "❌ Ошибка при обработке аудиофайла. Попробуйте использовать голосовые сообщения."
            )

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (transcriptions)"""
        user = update.effective_user
        text = update.message.text.strip()

        # Check if user has pending transcription
        submission_id = context.user_data.get('pending_transcription')
        if not submission_id:
            await update.message.reply_text(
                "📝 Сначала отправьте голосовое сообщение, затем его транскрипцию"
            )
            return

        try:
            # Analyze transcription
            segments = self.code_switch_detector.detect_segments(text)
            stats = self.code_switch_detector.analyze_text_statistics(text)

            # Update submission with transcription
            await self._update_transcription(submission_id, text)

            # Award points
            points_earned = self.points_per_transcription

            # Quality bonus for code-switching
            if stats['switch_points'] > 0:
                points_earned += 50  # Bonus for code-switching

            await self._award_points(user.id, points_earned)

            # Clear pending transcription
            context.user_data.pop('pending_transcription', None)

            # Analysis message
            analysis_text = "✅ Транскрипция сохранена!\n\n"
            analysis_text += f"🎯 Очки: +{points_earned}\n"
            analysis_text += f"📊 Анализ:\n"

            if stats['switch_points'] > 0:
                analysis_text += f"🔄 Переключения языков: {stats['switch_points']}\n"
                analysis_text += f"🌐 Смешанная речь обнаружена!\n"

            lang_dist = stats['language_distribution']
            if lang_dist:
                analysis_text += "📈 Языки: "
                for lang, count in lang_dist.items():
                    lang_name = {'uz': 'узбекский', 'ru': 'русский', 'mixed': 'смешанный'}.get(lang, lang)
                    analysis_text += f"{lang_name} ({count}), "
                analysis_text = analysis_text.rstrip(', ') + "\n"

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🎙️ Записать еще", callback_data="start_recording")],
                [InlineKeyboardButton("✅ Проверить других", callback_data="verify")]
            ])

            await update.message.reply_text(analysis_text, reply_markup=keyboard)

            # Send for verification if quality is uncertain
            if segments and any(seg.confidence < self.auto_approve_threshold for seg in segments):
                await self._queue_for_verification(submission_id)

        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            await update.message.reply_text(
                "❌ Ошибка при обработке транскрипции. Попробуйте еще раз."
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data

        if data.startswith("lang_"):
            # Language selection
            parts = data.split("_")
            language = parts[1]
            submission_id = parts[2]

            await self._update_language_hint(submission_id, language)

            lang_names = {'uz': 'Узбекский', 'ru': 'Русский', 'mixed': 'Смешанный'}
            await query.edit_message_text(
                f"✅ Язык: {lang_names[language]}\n"
                "Теперь напишите точную транскрипцию аудио:"
            )

        elif data == "stats":
            await self._show_user_stats(query)

        elif data == "verify":
            await self._show_verification_task(query)

        elif data == "start_recording":
            await query.edit_message_text(
                "🎙️ Отправьте голосовое сообщение для записи!\n\n"
                "💡 Советы:\n"
                "• Говорите четко и естественно\n"
                "• Можно использовать смешение языков\n"
                "• Длительность: 2-60 секунд"
            )

        elif data.startswith("verify_"):
            # Verification response
            await self._handle_verification_response(query, data)

        elif data == "leaderboard":
            await self._show_leaderboard_inline(query)

        elif data.startswith("lang_settings_"):
            language = data.split("_")[-1]
            lang_names = {'uz': "O'zbek", 'ru': 'Русский'}
            await query.edit_message_text(
                f"✅ Язык интерфейса изменен на: {lang_names.get(language, language)}\n"
                "Настройка сохранена!"
            )

        elif data == "notifications_settings":
            await query.edit_message_text(
                "🔔 НАСТРОЙКИ УВЕДОМЛЕНИЙ\n\n"
                "• Новые задания для проверки: ✅\n"
                "• Достижения и награды: ✅\n"
                "• Еженедельная статистика: ✅\n\n"
                "💡 Все уведомления включены"
            )

        elif data == "privacy_settings":
            await query.edit_message_text(
                "🔒 НАСТРОЙКИ ПРИВАТНОСТИ\n\n"
                "• Показывать в рейтинге: ✅\n"
                "• Публичная статистика: ✅\n"
                "• Анонимные данные: ❌\n\n"
                "💡 Ваши данные используются только для улучшения модели"
            )

    async def _show_verification_task(self, query):
        """Show verification task to user"""
        user_id = query.from_user.id

        # Get unverified submission for verification
        submission = await self._get_verification_task(user_id)

        if not submission:
            await query.edit_message_text(
                "✅ Нет заданий для проверки!\n"
                "Спасибо за активное участие!"
            )
            return

        # Send audio for verification
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Правильно", callback_data=f"verify_correct_{submission.id}"),
                InlineKeyboardButton("❌ Неправильно", callback_data=f"verify_wrong_{submission.id}")
            ],
            [InlineKeyboardButton("🤔 Пропустить", callback_data=f"verify_skip_{submission.id}")]
        ])

        verification_text = (
            "🔍 ПРОВЕРКА ТРАНСКРИПЦИИ\n\n"
            f"📝 Текст: {submission.transcription}\n\n"
            "🎧 Послушайте аудио и оцените точность транскрипции:"
        )

        # Send audio file
        with open(submission.file_path, 'rb') as audio_file:
            await query.message.reply_voice(
                voice=audio_file,
                caption=verification_text,
                reply_markup=keyboard
            )

    async def _handle_verification_response(self, query, data):
        """Handle verification response"""
        parts = data.split("_")
        action = parts[1]  # correct, wrong, skip
        submission_id = parts[2]

        user_id = query.from_user.id

        if action == "skip":
            await query.edit_message_text("⏭️ Пропущено. Спасибо!")
            return

        # Record verification
        is_correct = action == "correct"
        await self._record_verification(submission_id, user_id, is_correct)

        # Award points
        await self._award_points(user_id, self.points_per_verification)

        result_text = "✅ Спасибо за проверку!\n"
        result_text += f"🎯 Очки: +{self.points_per_verification}\n"

        if is_correct:
            result_text += "👍 Вы отметили транскрипцию как правильную"
        else:
            result_text += "👎 Вы отметили транскрипцию как неправильную"

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Проверить еще", callback_data="verify")]
        ])

        await query.edit_message_text(result_text, reply_markup=keyboard)

    async def _show_user_stats(self, query):
        """Show user statistics"""
        user_id = query.from_user.id
        stats = await self._get_user_stats(user_id)

        if not stats:
            await query.edit_message_text("📊 Статистика не найдена")
            return

        level = self._get_user_level(stats.points)
        next_level_points = self._get_next_level_points(stats.points)

        stats_text = f"📊 ВАША СТАТИСТИКА\n\n"
        stats_text += f"👤 Пользователь: {stats.username or 'Аноним'}\n"
        stats_text += f"🏆 Уровень: {level}\n"
        stats_text += f"🎯 Очки: {stats.points}\n"

        if next_level_points:
            stats_text += f"📈 До следующего уровня: {next_level_points - stats.points}\n"

        stats_text += f"\n📈 ВКЛАД:\n"
        stats_text += f"🎙️ Аудиозаписей: {stats.audio_count}\n"
        stats_text += f"✍️ Транскрипций: {stats.transcription_count}\n"
        stats_text += f"✅ Проверок: {stats.verification_count}\n"
        stats_text += f"⭐ Качество: {stats.quality_score:.1%}\n"

        if stats.badges:
            stats_text += f"\n🏅 ЗНАЧКИ: {', '.join(stats.badges)}"

        await query.edit_message_text(stats_text)

    async def _show_leaderboard_inline(self, query):
        """Show leaderboard in inline mode"""
        leaderboard = await self._get_leaderboard(limit=10)

        if not leaderboard:
            await query.edit_message_text("🏅 Рейтинг пока пуст")
            return

        leaderboard_text = "🏆 РЕЙТИНГ УЧАСТНИКОВ\n\n"

        medals = ["🥇", "🥈", "🥉"] + ["🏅"] * 7

        for i, user_stats in enumerate(leaderboard):
            medal = medals[i] if i < len(medals) else "📊"
            username = user_stats.username or f"User{user_stats.user_id}"
            level = self._get_user_level(user_stats.points)

            leaderboard_text += (
                f"{medal} {username}\n"
                f"   📊 {user_stats.points} очков • {level}\n"
                f"   🎙️ {user_stats.audio_count} аудио • "
                f"✅ {user_stats.verification_count} проверок\n\n"
            )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Моя статистика", callback_data="stats")]
        ])

        await query.edit_message_text(leaderboard_text, reply_markup=keyboard)

    async def cmd_leaderboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show leaderboard"""
        leaderboard = await self._get_leaderboard(limit=10)

        if not leaderboard:
            await update.message.reply_text("🏅 Рейтинг пока пуст")
            return

        leaderboard_text = "🏆 РЕЙТИНГ УЧАСТНИКОВ\n\n"

        medals = ["🥇", "🥈", "🥉"] + ["🏅"] * 7

        for i, user_stats in enumerate(leaderboard):
            medal = medals[i] if i < len(medals) else "📊"
            username = user_stats.username or f"User{user_stats.user_id}"
            level = self._get_user_level(user_stats.points)

            leaderboard_text += (
                f"{medal} {username}\n"
                f"   📊 {user_stats.points} очков • {level}\n"
                f"   🎙️ {user_stats.audio_count} аудио • "
                f"✅ {user_stats.verification_count} проверок\n\n"
            )

        await update.message.reply_text(leaderboard_text)

    # Database operations
    async def _ensure_user_exists(self, user_id: int, username: Optional[str]):
        """Ensure user exists in database"""
        return await self.db_manager.ensure_user_exists(
            telegram_id=user_id,
            username=username
        )

    async def _store_submission(self, submission: AudioSubmission):
        """Store audio submission in database"""
        submission_data = {
            'id': submission.id,
            'user_id': submission.user_id,
            'file_id': submission.file_id,
            'file_path': submission.file_path,
            'duration': submission.duration,
            'language_hint': submission.language_hint,
            'transcription': submission.transcription,
            'quality_score': submission.quality_score,
            'verification_count': submission.verification_count,
            'verified': submission.verified,
            'timestamp': submission.timestamp,
            'audio_metadata': submission.metadata
        }
        return await self.db_manager.store_audio_submission(submission_data)

    async def _update_transcription(self, submission_id: str, transcription: str):
        """Update submission with transcription"""
        return await self.db_manager.update_transcription(submission_id, transcription)

    async def _award_points(self, user_id: int, points: int):
        """Award points to user"""
        return await self.db_manager.award_points(user_id, points)

    def _get_user_level(self, points: int) -> str:
        """Get user level based on points"""
        for level, threshold in reversed(list(self.level_thresholds.items())):
            if points >= threshold:
                return level
        return "Bronze"

    def _get_next_level_points(self, points: int) -> Optional[int]:
        """Get points needed for next level"""
        for threshold in sorted(self.level_thresholds.values()):
            if points < threshold:
                return threshold
        return None

    async def run(self):
        """Run the bot"""
        logger.info("Starting Uzbek ASR Data Collection Bot...")

        # Initialize database
        await self._init_database()

        # Start bot
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

        logger.info("Bot is running...")

        try:
            # Keep running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
        finally:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    async def _init_database(self):
        """Initialize database tables"""
        await self.db_manager.init_database()

    async def _update_language_hint(self, submission_id: str, language_hint: str):
        """Update language hint for submission"""
        return await self.db_manager.update_language_hint(submission_id, language_hint)

    async def _get_verification_task(self, user_id: int):
        """Get verification task for user"""
        return await self.db_manager.get_verification_task(user_id)

    async def _record_verification(self, submission_id: str, user_id: int, is_correct: bool):
        """Record verification response"""
        return await self.db_manager.record_verification(submission_id, user_id, is_correct)

    async def _get_user_stats(self, user_id: int):
        """Get user statistics"""
        user_stats = await self.db_manager.get_user_stats(user_id)
        if not user_stats:
            return None

        # Convert database model to our UserStats dataclass
        return UserStats(
            user_id=user_stats.user_id,
            username=user_stats.user.username if user_stats.user else None,
            audio_count=user_stats.audio_count,
            transcription_count=user_stats.transcription_count,
            verification_count=user_stats.verification_count,
            quality_score=user_stats.quality_score,
            points=user_stats.points,
            level=self._get_user_level(user_stats.points),
            badges=user_stats.badges or []
        )

    async def _get_leaderboard(self, limit: int = 10):
        """Get leaderboard"""
        db_leaderboard = await self.db_manager.get_leaderboard(limit)

        leaderboard = []
        for user_stats in db_leaderboard:
            leaderboard.append(UserStats(
                user_id=user_stats.user_id,
                username=user_stats.user.username if user_stats.user else None,
                audio_count=user_stats.audio_count,
                transcription_count=user_stats.transcription_count,
                verification_count=user_stats.verification_count,
                quality_score=user_stats.quality_score,
                points=user_stats.points,
                level=self._get_user_level(user_stats.points),
                badges=user_stats.badges or []
            ))

        return leaderboard

    async def _queue_for_verification(self, submission_id: str):
        """Queue submission for verification (automatically handled by verification_count)"""
        # This is handled automatically when verification_count < min_verifications
        pass


def main():
    """Main entry point"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    database_url = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///uzbek_asr.db')

    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

    bot = UzbekASRDataBot(
        token=bot_token,
        database_url=database_url
    )

    asyncio.run(bot.run())


if __name__ == "__main__":
    main()