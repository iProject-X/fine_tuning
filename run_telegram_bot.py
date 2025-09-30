#!/usr/bin/env python3
"""
Script to run the Uzbek ASR Telegram Bot
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from src.data.collectors.telegram_bot_collector import UzbekASRDataBot

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging configuration"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE', './logs/telegram_bot.log')

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

async def main():
    """Main function to run the bot"""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get configuration from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    database_url = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///uzbek_asr.db')
    storage_path = os.getenv('STORAGE_PATH', './data/telegram')

    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is required")
        logger.error("Please copy .env.example to .env and fill in your bot token")
        sys.exit(1)

    logger.info("Starting Uzbek ASR Data Collection Bot...")
    logger.info(f"Database URL: {database_url}")
    logger.info(f"Storage Path: {storage_path}")

    # Create storage directory
    os.makedirs(storage_path, exist_ok=True)

    # Initialize and run bot
    bot = UzbekASRDataBot(
        token=bot_token,
        database_url=database_url,
        storage_path=storage_path
    )

    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())