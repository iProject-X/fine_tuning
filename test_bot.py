#!/usr/bin/env python3
"""
Test script to verify bot components work correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_database():
    """Test database functionality"""
    print("Testing database...")

    from src.utils.database import DatabaseManager

    # Test with SQLite
    db = DatabaseManager("sqlite+aiosqlite:///test_uzbek_asr.db")

    try:
        # Initialize database
        await db.init_database()
        print("✅ Database tables created successfully")

        # Test user creation
        user = await db.ensure_user_exists(
            telegram_id=123456789,
            username="test_user",
            first_name="Test",
            last_name="User"
        )
        print(f"✅ User created: {user.username}")

        # Test audio submission
        submission_data = {
            'id': 'test-submission-1',
            'user_id': 123456789,
            'file_id': 'test_file_id',
            'file_path': '/tmp/test.ogg',
            'duration': 5.0,
            'language_hint': 'uz',
            'transcription': None,
            'quality_score': 0.8,
            'verification_count': 0,
            'verified': False,
            'timestamp': None,
            'audio_metadata': {'test': True}
        }

        submission = await db.store_audio_submission(submission_data)
        print(f"✅ Audio submission stored: {submission.id}")

        # Test transcription update
        success = await db.update_transcription('test-submission-1', 'Salom dunyo!')
        print(f"✅ Transcription updated: {success}")

        # Test points award
        success = await db.award_points(123456789, 100)
        print(f"✅ Points awarded: {success}")

        # Test user stats
        stats = await db.get_user_stats(123456789)
        if stats:
            print(f"✅ User stats retrieved: {stats.points} points, {stats.audio_count} audio")

        await db.close()
        print("✅ Database test completed successfully")

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

    return True

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")

    try:
        from src.data.collectors.telegram_bot_collector import UzbekASRDataBot, AudioSubmission, UserStats
        print("✅ Bot classes imported successfully")

        from src.utils.database import DatabaseManager, User, AudioSubmission as DBSubmission
        print("✅ Database classes imported successfully")

        from src.data.processors.code_switch_detector import CodeSwitchDetector
        print("✅ Code switch detector imported successfully")

        from src.utils.audio_utils import AudioProcessor
        print("✅ Audio processor imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Running bot component tests...\n")

    # Test imports
    if not test_imports():
        sys.exit(1)

    print()

    # Test database
    if not await test_database():
        sys.exit(1)

    print("\n🎉 All tests passed! Bot is ready to run.")
    print("\nNext steps:")
    print("1. Copy .env.example to .env")
    print("2. Add your TELEGRAM_BOT_TOKEN to .env")
    print("3. Run: python run_telegram_bot.py")

    # Clean up test database
    try:
        os.remove("test_uzbek_asr.db")
        print("\n🧹 Cleaned up test database")
    except:
        pass

if __name__ == "__main__":
    asyncio.run(main())