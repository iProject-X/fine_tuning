# Uzbek ASR Telegram Bot

Telegram бот для краудсорсингового сбора данных для обучения узбекской системы распознавания речи.

## Возможности

- 🎙️ Сбор голосовых сообщений на узбекском/русском языках
- ✍️ Краудсорсинговая транскрипция аудио
- ✅ Система верификации качества через голосование
- 🏆 Геймификация с очками, уровнями и таблицей лидеров
- 🔄 Автоматическое определение смешения языков (code-switching)
- 📊 Статистика пользователей и контроль качества

## Установка

1. **Создайте виртуальное окружение** (если не создано):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

2. **Установите зависимости**:
```bash
pip install -r requirements.txt
```

3. **Настройте конфигурацию**:
```bash
cp .env.example .env
```

Отредактируйте `.env` файл и добавьте ваш токен бота:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
DATABASE_URL=sqlite+aiosqlite:///uzbek_asr.db
```

## Создание Telegram бота

1. Напишите [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Следуйте инструкциям для создания бота
4. Скопируйте полученный токен в `.env` файл

## Запуск

```bash
python run_telegram_bot.py
```

Или запустите тестирование компонентов:
```bash
python test_bot.py
```

## Структура проекта

```
src/
├── data/
│   ├── collectors/
│   │   └── telegram_bot_collector.py  # Основной класс бота
│   └── processors/
│       └── code_switch_detector.py    # Детектор смешения языков
├── utils/
│   ├── database.py                    # Модели и менеджер БД
│   ├── audio_utils.py                 # Обработка аудио
│   └── config.py                      # Конфигурация
└── ...

.env.example                           # Шаблон конфигурации
run_telegram_bot.py                   # Скрипт запуска
test_bot.py                           # Тесты компонентов
```

## База данных

Бот автоматически создает следующие таблицы:

- **users** - пользователи Telegram
- **audio_submissions** - голосовые сообщения
- **verifications** - верификации транскрипций
- **user_statistics** - статистика пользователей

По умолчанию используется SQLite, но можно настроить PostgreSQL:
```
DATABASE_URL=postgresql+asyncpg://user:password@localhost/uzbek_asr
```

## Игровая механика

### Очки:
- 100 очков за голосовое сообщение
- 50 очков за транскрипцию
- 25 очков за верификацию
- Бонусы за качество и смешение языков

### Уровни:
- Bronze: 0+ очков
- Silver: 1000+ очков
- Gold: 5000+ очков
- Platinum: 15000+ очков
- Diamond: 50000+ очков

## Использование бота

### Для пользователей:

1. **Запись аудио**: Отправьте голосовое сообщение (2-60 сек)
2. **Указание языка**: Выберите язык записи (узбекский/русский/смешанный)
3. **Транскрипция**: Напишите точный текст того, что сказали
4. **Верификация**: Проверяйте транскрипции других пользователей

### Команды бота:

- `/start` - Начать работу с ботом
- `/help` - Подробная справка
- `/stats` - Ваша статистика
- `/leaderboard` - Таблица лидеров
- `/settings` - Настройки

## API и интеграция

Бот собирает данные в структурированном формате, подходящем для обучения:

```python
{
    "audio_file": "path/to/audio.ogg",
    "transcription": "Men bugun рынок ga boraman",
    "language": "mixed",  # uz, ru, mixed
    "quality_score": 0.95,
    "verified": True,
    "code_switches": [
        {"start": 11, "end": 16, "lang": "ru"}
    ]
}
```

## Мониторинг

Логи сохраняются в `logs/telegram_bot.log`:
```bash
tail -f logs/telegram_bot.log
```

## Разработка

Для разработки установите дополнительные зависимости:
```bash
pip install pytest black isort flake8 mypy
```

Запуск тестов:
```bash
pytest tests/
```

Форматирование кода:
```bash
black src/
isort src/
```

## Примеры использования

### Тестирование локально
```bash
# Запуск тестов
python test_bot.py

# Проверка импортов
python -c "from src.data.collectors.telegram_bot_collector import UzbekASRDataBot"
```

### Развертывание
```bash
# Docker (опционально)
docker build -t uzbek-asr-bot .
docker run -d --env-file .env uzbek-asr-bot

# Systemd service
sudo cp uzbek-asr-bot.service /etc/systemd/system/
sudo systemctl enable uzbek-asr-bot
sudo systemctl start uzbek-asr-bot
```

## Устранение неполадок

### Частые ошибки:

1. **Ошибка токена**: Проверьте правильность `TELEGRAM_BOT_TOKEN`
2. **Ошибка БД**: Убедитесь что есть права на запись в папку с БД
3. **Ошибка импорта**: Проверьте что все зависимости установлены

### Логи:
```bash
# Увеличить уровень логирования
export LOG_LEVEL=DEBUG

# Просмотр ошибок
grep ERROR logs/telegram_bot.log
```

## Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Сделайте изменения
4. Добавьте тесты
5. Создайте Pull Request

## Лицензия

MIT License - см. LICENSE файл для деталей.