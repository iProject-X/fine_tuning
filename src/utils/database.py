"""
Database models and manager for Uzbek ASR data collection
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, create_engine
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    """User model for contributors"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    language_code = Column(String(10), nullable=True)
    points = Column(Integer, default=0)
    level = Column(String(50), default='Bronze')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    audio_submissions = relationship("AudioSubmission", back_populates="user")
    verifications = relationship("Verification", back_populates="user")
    user_stats = relationship("UserStatistics", back_populates="user", uselist=False)

class AudioSubmission(Base):
    """Audio submission model"""
    __tablename__ = 'audio_submissions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.telegram_id'), nullable=False)
    file_id = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    duration = Column(Float, nullable=False)
    language_hint = Column(String(10), nullable=True)  # 'uz', 'ru', 'mixed'
    transcription = Column(Text, nullable=True)
    quality_score = Column(Float, default=0.0)
    verification_count = Column(Integer, default=0)
    verified = Column(Boolean, default=False)
    approved = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    audio_metadata = Column(JSON, nullable=True)

    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_verified_approved', 'verified', 'approved'),
        Index('idx_verification_queue', 'verification_count', 'verified'),
    )

    # Relationships
    user = relationship("User", back_populates="audio_submissions")
    verifications = relationship("Verification", back_populates="submission")

class Verification(Base):
    """Verification model for crowd-sourced quality control"""
    __tablename__ = 'verifications'

    id = Column(Integer, primary_key=True)
    submission_id = Column(String(36), ForeignKey('audio_submissions.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.telegram_id'), nullable=False)
    is_correct = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=True)  # 0.0 - 1.0
    comments = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Constraints to prevent duplicate verifications
    __table_args__ = (
        Index('idx_unique_verification', 'submission_id', 'user_id', unique=True),
        Index('idx_submission_timestamp', 'submission_id', 'timestamp'),
    )

    # Relationships
    submission = relationship("AudioSubmission", back_populates="verifications")
    user = relationship("User", back_populates="verifications")

class UserStatistics(Base):
    """Aggregated user statistics"""
    __tablename__ = 'user_statistics'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.telegram_id'), unique=True, nullable=False)
    audio_count = Column(Integer, default=0)
    transcription_count = Column(Integer, default=0)
    verification_count = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    points = Column(Integer, default=0)
    badges = Column(JSON, default=list)  # List of earned badges
    last_activity = Column(DateTime, default=datetime.utcnow)
    streak_days = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="user_stats")

class DatabaseManager:
    """Database manager for handling all database operations"""

    def __init__(self, database_url: str):
        """
        Initialize database manager

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_database(self):
        """Create all database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    async def ensure_user_exists(self, telegram_id: int, username: Optional[str] = None,
                               first_name: Optional[str] = None, last_name: Optional[str] = None,
                               language_code: Optional[str] = None) -> User:
        """Ensure user exists in database, create if not exists"""
        async with self.async_session() as session:
            from sqlalchemy import select
            # Check if user exists
            result = await session.execute(select(User).where(User.telegram_id == telegram_id))
            user = result.scalar_one_or_none()

            if not user:
                # Create new user
                user = User(
                    telegram_id=telegram_id,
                    username=username,
                    first_name=first_name,
                    last_name=last_name,
                    language_code=language_code
                )
                session.add(user)

                # Create user statistics
                user_stats = UserStatistics(user_id=telegram_id)
                session.add(user_stats)

                await session.commit()
                await session.refresh(user)
                logger.info(f"Created new user: {telegram_id}")
            else:
                # Update user info if provided
                if username is not None:
                    user.username = username
                if first_name is not None:
                    user.first_name = first_name
                if last_name is not None:
                    user.last_name = last_name
                if language_code is not None:
                    user.language_code = language_code

                user.updated_at = datetime.utcnow()
                await session.commit()

            return user

    async def store_audio_submission(self, submission_data: Dict[str, Any]) -> AudioSubmission:
        """Store audio submission in database"""
        async with self.async_session() as session:
            submission = AudioSubmission(**submission_data)
            session.add(submission)
            await session.commit()
            await session.refresh(submission)

            # Update user statistics
            await self._update_user_audio_count(session, submission.user_id)

            logger.info(f"Stored audio submission: {submission.id}")
            return submission

    async def update_transcription(self, submission_id: str, transcription: str) -> bool:
        """Update submission with transcription"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(select(AudioSubmission).where(AudioSubmission.id == submission_id))
            submission = result.scalar_one_or_none()
            if not submission:
                return False

            submission.transcription = transcription
            await session.commit()

            # Update user statistics
            await self._update_user_transcription_count(session, submission.user_id)

            logger.info(f"Updated transcription for submission: {submission_id}")
            return True

    async def update_language_hint(self, submission_id: str, language_hint: str) -> bool:
        """Update language hint for submission"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(select(AudioSubmission).where(AudioSubmission.id == submission_id))
            submission = result.scalar_one_or_none()
            if not submission:
                return False

            submission.language_hint = language_hint
            await session.commit()
            return True

    async def award_points(self, user_id: int, points: int) -> bool:
        """Award points to user"""
        async with self.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(select(User).where(User.telegram_id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                return False

            user.points += points
            user.updated_at = datetime.utcnow()

            # Update level based on points
            user.level = self._calculate_level(user.points)

            # Update user statistics
            stats_result = await session.execute(select(UserStatistics).where(UserStatistics.user_id == user_id))
            user_stats = stats_result.scalar_one_or_none()
            if user_stats:
                user_stats.points = user.points
                user_stats.last_activity = datetime.utcnow()

            await session.commit()
            logger.info(f"Awarded {points} points to user {user_id}")
            return True

    async def get_verification_task(self, user_id: int) -> Optional[AudioSubmission]:
        """Get an unverified submission for verification (excluding user's own submissions)"""
        async with self.async_session() as session:
            from sqlalchemy import and_, not_, exists, select

            # Find submissions that need verification
            query = select(AudioSubmission).where(
                and_(
                    AudioSubmission.user_id != user_id,  # Not user's own submission
                    AudioSubmission.transcription.isnot(None),  # Has transcription
                    AudioSubmission.verification_count < 3,  # Needs more verifications
                    AudioSubmission.verified == False,  # Not yet verified
                    # User hasn't verified this submission yet
                    not_(exists().where(
                        and_(
                            Verification.submission_id == AudioSubmission.id,
                            Verification.user_id == user_id
                        )
                    ))
                )
            ).order_by(AudioSubmission.timestamp.asc())

            result = await session.execute(query)
            submission = result.scalar_one_or_none()
            return submission

    async def record_verification(self, submission_id: str, user_id: int,
                                is_correct: bool, confidence: Optional[float] = None) -> bool:
        """Record a verification"""
        async with self.async_session() as session:
            from sqlalchemy import and_, select

            # Check if verification already exists
            existing_query = select(Verification).where(
                and_(
                    Verification.submission_id == submission_id,
                    Verification.user_id == user_id
                )
            )
            existing_result = await session.execute(existing_query)
            existing = existing_result.scalar_one_or_none()

            if existing:
                return False  # Already verified by this user

            # Create verification
            verification = Verification(
                submission_id=submission_id,
                user_id=user_id,
                is_correct=is_correct,
                confidence=confidence
            )
            session.add(verification)

            # Update submission verification count
            submission_result = await session.execute(select(AudioSubmission).where(AudioSubmission.id == submission_id))
            submission = submission_result.scalar_one_or_none()
            if submission:
                submission.verification_count += 1

                # Check if submission should be marked as verified
                if submission.verification_count >= 3:
                    # Calculate verification consensus
                    verifications_query = select(Verification).where(Verification.submission_id == submission_id)
                    verifications_result = await session.execute(verifications_query)
                    verifications = verifications_result.scalars().all()

                    correct_count = sum(1 for v in verifications if v.is_correct)
                    consensus_threshold = len(verifications) * 0.6  # 60% consensus

                    if correct_count >= consensus_threshold:
                        submission.verified = True
                        submission.approved = True
                    else:
                        submission.verified = True
                        submission.approved = False

            await session.commit()

            # Update user verification statistics
            await self._update_user_verification_count(session, user_id)

            logger.info(f"Recorded verification for submission {submission_id} by user {user_id}")
            return True

    async def get_user_stats(self, user_id: int) -> Optional[UserStatistics]:
        """Get user statistics"""
        async with self.async_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            result = await session.execute(
                select(UserStatistics)
                .options(selectinload(UserStatistics.user))
                .where(UserStatistics.user_id == user_id)
            )
            user_stats = result.scalar_one_or_none()
            return user_stats

    async def get_leaderboard(self, limit: int = 10) -> List[UserStatistics]:
        """Get top users by points"""
        async with self.async_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            query = select(UserStatistics).options(
                selectinload(UserStatistics.user)
            ).join(User).where(
                User.is_active == True
            ).order_by(UserStatistics.points.desc()).limit(limit)

            result = await session.execute(query)
            leaderboard = result.scalars().all()
            return leaderboard

    async def _update_user_audio_count(self, session: AsyncSession, user_id: int):
        """Update user's audio submission count"""
        from sqlalchemy import select
        result = await session.execute(select(UserStatistics).where(UserStatistics.user_id == user_id))
        user_stats = result.scalar_one_or_none()
        if user_stats:
            user_stats.audio_count += 1
            user_stats.last_activity = datetime.utcnow()

    async def _update_user_transcription_count(self, session: AsyncSession, user_id: int):
        """Update user's transcription count"""
        from sqlalchemy import select
        result = await session.execute(select(UserStatistics).where(UserStatistics.user_id == user_id))
        user_stats = result.scalar_one_or_none()
        if user_stats:
            user_stats.transcription_count += 1
            user_stats.last_activity = datetime.utcnow()

    async def _update_user_verification_count(self, session: AsyncSession, user_id: int):
        """Update user's verification count"""
        from sqlalchemy import select
        result = await session.execute(select(UserStatistics).where(UserStatistics.user_id == user_id))
        user_stats = result.scalar_one_or_none()
        if user_stats:
            user_stats.verification_count += 1
            user_stats.last_activity = datetime.utcnow()

    def _calculate_level(self, points: int) -> str:
        """Calculate user level based on points"""
        level_thresholds = {
            'Bronze': 0,
            'Silver': 1000,
            'Gold': 5000,
            'Platinum': 15000,
            'Diamond': 50000
        }

        for level, threshold in reversed(list(level_thresholds.items())):
            if points >= threshold:
                return level
        return "Bronze"

    async def close(self):
        """Close database connections"""
        await self.engine.dispose()