# 🚀 Быстрый запуск Uzbek Whisper ASR System

## 📋 Пошаговая инструкция

### 1️⃣ Первоначальная настройка

```bash
# 1. Настроить систему (установить зависимости)
make setup
# или
python start_system.py setup

# 2. Настроить конфигурацию
cp .env.example .env
```

### 2️⃣ Получить токен Telegram бота

1. Напишите [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Следуйте инструкциям
4. Скопируйте токен в `.env`:

```bash
# Отредактируйте .env файл
nano .env

# Добавьте ваш токен:
TELEGRAM_BOT_TOKEN=1234567890:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
```

### 3️⃣ Тестирование

```bash
# Проверить что всё работает
make test
# или
python test_bot.py
```

### 4️⃣ Запуск системы

```bash
# Запустить всю систему
make start
# или
python start_system.py start
```

### 5️⃣ Проверка статуса

```bash
# Проверить статус
make status
# или
python start_system.py status
```

---

## 🎯 Варианты запуска

### Полная система (рекомендуется)
```bash
make start
```
**Включает:** Telegram бот + API сервер + мониторинг

### Только Telegram бот
```bash
make start-bot
# или
python run_telegram_bot.py
```

### Только API сервер
```bash
make start-api
```

---

## 🎓 Обучение модели

### Базовое обучение
```bash
make train
```

### Продвинутое обучение
```bash
python train_model.py \
  --model-size medium \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-5 \
  --data-path ./data \
  --output-dir ./models/whisper-uz-medium
```

---

## 🔧 Управление системой

```bash
# Справка по командам
make help

# Остановить систему
make stop

# Перезапустить
make restart

# Очистить временные файлы
make clean

# Установить зависимости
make install
```

---

## 📊 Мониторинг

### Проверка логов
```bash
# Логи Telegram бота
tail -f logs/telegram_bot.log

# Логи API сервера (если запущен)
tail -f logs/api_server.log
```

### Проверка базы данных
```bash
# Размер базы данных
ls -lh uzbek_asr.db

# SQLite браузер (если установлен)
sqlite3 uzbek_asr.db
```

---

## 🎮 Использование бота

### Для пользователей:

1. **Найдите вашего бота в Telegram** (по имени, которое вы дали при создании)

2. **Отправьте `/start`**

3. **Записывайте голосовые сообщения:**
   - Длительность: 2-60 секунд
   - На узбекском, русском или смешанно
   - В тихом месте

4. **Выберите язык записи**

5. **Напишите транскрипцию** (точный текст того, что сказали)

6. **Проверяйте записи других** (получаете очки за верификацию)

### Команды бота:
- `/start` - начать работу
- `/help` - справка
- `/stats` - ваша статистика
- `/leaderboard` - таблица лидеров

---

## 🛠 Устранение неполадок

### Бот не запускается
```bash
# Проверить токен
grep TELEGRAM_BOT_TOKEN .env

# Проверить зависимости
pip list | grep telegram

# Проверить логи
cat logs/telegram_bot.log
```

### База данных
```bash
# Пересоздать базу
rm uzbek_asr.db
python test_bot.py
```

### Ошибки импорта
```bash
# Переустановить зависимости
pip install -r requirements.txt --force-reinstall
```

---

## 📁 Структура данных

### Собранные данные:
```
data/
├── telegram/           # Аудиофайлы от пользователей
├── processed/          # Обработанные данные
└── exports/           # Экспорты для обучения
```

### База данных:
- **users** - пользователи и их статистика
- **audio_submissions** - голосовые записи
- **verifications** - проверки качества
- **user_statistics** - агрегированная статистика

---

## 🚀 Быстрые команды

```bash
# Всё в одной команде (первый запуск)
make setup && cp .env.example .env && echo "Настройте .env и запустите: make start"

# Ежедневный запуск
make start

# Проверка системы
make status && make test

# Остановка
make stop
```

---

## 📞 Поддержка

Если что-то не работает:

1. **Проверьте логи**: `cat logs/telegram_bot.log`
2. **Запустите тесты**: `make test`
3. **Проверьте статус**: `make status`
4. **Пересоздайте .env**: `cp .env.example .env`

---

## 🎉 Готово!

После успешного запуска:
- 🤖 Пользователи могут отправлять голосовые сообщения вашему боту
- 📊 Данные сохраняются в базе данных
- 🎯 Можно запускать обучение модели
- 🌐 API доступен на http://localhost:8000 (если запущен)

**Удачного сбора данных! 🎙️**