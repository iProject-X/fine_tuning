"""
Code-switching detector for Uzbek-Russian mixed speech
Handles detection and segmentation of language switches in text
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class CodeSwitchSegment:
    """Represents a segment of text in a specific language"""
    text: str
    language: str  # 'uz', 'ru', 'mixed', 'unknown'
    start_idx: int
    end_idx: int
    confidence: float
    word_count: int

class CodeSwitchDetector:
    """
    Advanced code-switching detector for Uzbek-Russian text

    Features:
    - Rule-based language detection using linguistic patterns
    - Script-based detection (Cyrillic vs Latin)
    - Context-aware segmentation
    - Confidence scoring
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the code-switch detector

        Args:
            config_path: Path to configuration file with language patterns
        """
        # Uzbek language patterns (Latin script)
        self.uz_patterns = [
            # Pronouns
            r'\b(men|sen|u|biz|siz|ular|mening|sening|uning|bizning|sizning|ularning)\b',

            # Common verbs with suffixes
            r'\b(qil|kel|bor|tur|ol|ber|ko\'r|ur|yoz|o\'qi|ish|yur|o\'yn|sot|ol|tush)\w*\b',

            # Uzbek suffixes
            r'\b\w+(lar|ning|ni|da|ga|dan|dagi|lari|ligini|moqda|yotir|di|gan)\b',

            # Question words
            r'\b(kim|nima|qayer|qachon|qanday|necha|nega|qani)\b',

            # Common Uzbek words
            r'\b(bu|shu|o\'sha|hamma|bari|juda|ko\'p|oz|kam|yangi|eski|katta|kichik)\b',

            # Time expressions
            r'\b(bugun|erta|kech|tong|peshin|oqshom|kun|hafta|oy|yil)\b',

            # Uzbek specific letter combinations
            r"[o']",  # Uzbek apostrophe
            r"\b\w*[qx]\w*\b",  # Words with q or x (common in Uzbek)
        ]

        # Russian language patterns (Cyrillic script)
        self.ru_patterns = [
            # Pronouns
            r'\b(я|ты|он|она|оно|мы|вы|они|меня|тебя|его|её|нас|вас|их)\b',

            # Common verbs
            r'\b(быть|иметь|делать|говорить|знать|видеть|идти|дать|хотеть|мочь|думать)\w*\b',

            # Russian suffixes
            r'\b\w+(ость|ение|ание|ция|ский|ной|ные|ими|ами|ах|ях|ов|ев)\b',

            # Question words
            r'\b(кто|что|где|когда|как|сколько|почему|зачем|откуда|куда)\b',

            # Common Russian words
            r'\b(это|тот|такой|весь|каждый|другой|новый|старый|большой|маленький)\b',

            # Russian prepositions
            r'\b(в|на|за|под|над|при|для|без|через|между|среди|около)\b',

            # Russian particles
            r'\b(не|ни|же|ли|то|ка|бы|б)\b'
        ]

        # Mixed language indicators
        self.mixed_indicators = [
            # Common code-switching patterns
            r'\b(можно|нужно|надо)\s+\w+\b',  # Russian modal + any word
            r'\b\w+\s+(edi|ekan|emas)\b',     # Uzbek auxiliary verbs
        ]

        # Compile patterns for better performance
        self.uz_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.uz_patterns]
        self.ru_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.ru_patterns]
        self.mixed_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.mixed_indicators]

        # Load additional patterns from config if provided
        if config_path:
            self._load_config(config_path)

    def detect_segments(self, text: str, min_segment_length: int = 2) -> List[CodeSwitchSegment]:
        """
        Detect language segments in mixed text

        Args:
            text: Input text to analyze
            min_segment_length: Minimum words per segment

        Returns:
            List of CodeSwitchSegment objects
        """
        if not text.strip():
            return []

        # Clean and tokenize text
        words = self._tokenize_text(text)
        if not words:
            return []

        segments = []
        current_segment_words = []
        current_language = None
        segment_start = 0

        for i, word in enumerate(words):
            detected_lang = self._detect_word_language(word)

            # Check if we need to start a new segment
            if detected_lang != current_language and current_segment_words:
                # Create segment from accumulated words
                if len(current_segment_words) >= min_segment_length:
                    segment_text = ' '.join(current_segment_words)
                    confidence = self._calculate_segment_confidence(
                        current_segment_words, current_language
                    )

                    segments.append(CodeSwitchSegment(
                        text=segment_text,
                        language=current_language,
                        start_idx=segment_start,
                        end_idx=i - 1,
                        confidence=confidence,
                        word_count=len(current_segment_words)
                    ))

                # Start new segment
                current_segment_words = [word]
                current_language = detected_lang
                segment_start = i
            else:
                # Add to current segment
                current_segment_words.append(word)

                # Initialize language if not set
                if current_language is None:
                    current_language = detected_lang

        # Handle final segment
        if current_segment_words and len(current_segment_words) >= min_segment_length:
            segment_text = ' '.join(current_segment_words)
            confidence = self._calculate_segment_confidence(
                current_segment_words, current_language
            )

            segments.append(CodeSwitchSegment(
                text=segment_text,
                language=current_language,
                start_idx=segment_start,
                end_idx=len(words) - 1,
                confidence=confidence,
                word_count=len(current_segment_words)
            ))

        # Post-process segments to merge similar adjacent ones
        segments = self._merge_similar_segments(segments)

        return segments

    def get_switch_points(self, segments: List[CodeSwitchSegment]) -> List[int]:
        """
        Extract language switch points from segments

        Args:
            segments: List of language segments

        Returns:
            List of word indices where language switches occur
        """
        switch_points = []

        for i in range(1, len(segments)):
            if segments[i].language != segments[i-1].language:
                switch_points.append(segments[i].start_idx)

        return switch_points

    def analyze_text_statistics(self, text: str) -> Dict:
        """
        Analyze code-switching statistics for the text

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with analysis results
        """
        segments = self.detect_segments(text)

        if not segments:
            return {
                'total_segments': 0,
                'switch_points': 0,
                'language_distribution': {},
                'avg_segment_length': 0,
                'code_switch_ratio': 0
            }

        # Calculate statistics
        language_counts = Counter(seg.language for seg in segments)
        total_words = sum(seg.word_count for seg in segments)
        switch_points = len(self.get_switch_points(segments))

        return {
            'total_segments': len(segments),
            'switch_points': switch_points,
            'language_distribution': dict(language_counts),
            'avg_segment_length': total_words / len(segments),
            'code_switch_ratio': switch_points / max(len(segments) - 1, 1),
            'total_words': total_words,
            'confidence_scores': [seg.confidence for seg in segments]
        }

    def _detect_word_language(self, word: str) -> str:
        """
        Detect language of a single word

        Args:
            word: Word to analyze

        Returns:
            Language code ('uz', 'ru', 'mixed', 'unknown')
        """
        if not word.strip():
            return 'unknown'

        word_clean = word.lower().strip()

        # Remove punctuation for analysis
        word_alpha = ''.join(c for c in word_clean if c.isalpha() or c in ["'", "'"])

        if not word_alpha:
            return 'unknown'

        # Check script first (most reliable indicator)
        has_cyrillic = bool(re.search(r'[а-яё]', word_alpha))
        has_latin = bool(re.search(r'[a-z]', word_alpha))

        if has_cyrillic and has_latin:
            return 'mixed'
        elif has_cyrillic:
            # Check Russian patterns
            ru_score = sum(1 for pattern in self.ru_regex if pattern.search(word_alpha))
            return 'ru' if ru_score > 0 else 'ru'  # Default to Russian for Cyrillic
        elif has_latin:
            # Check Uzbek patterns
            uz_score = sum(1 for pattern in self.uz_regex if pattern.search(word_alpha))
            return 'uz' if uz_score > 0 else 'uz'  # Default to Uzbek for Latin

        return 'unknown'

    def _calculate_segment_confidence(
        self,
        words: List[str],
        detected_language: str
    ) -> float:
        """
        Calculate confidence score for a language segment

        Args:
            words: List of words in the segment
            detected_language: Detected language for the segment

        Returns:
            Confidence score between 0 and 1
        """
        if not words or detected_language == 'unknown':
            return 0.0

        pattern_matches = 0
        total_words = len(words)

        if detected_language == 'uz':
            patterns = self.uz_regex
        elif detected_language == 'ru':
            patterns = self.ru_regex
        else:
            return 0.5  # Default confidence for mixed/unknown

        # Count pattern matches
        for word in words:
            word_clean = word.lower().strip()
            for pattern in patterns:
                if pattern.search(word_clean):
                    pattern_matches += 1
                    break

        # Calculate base confidence
        confidence = pattern_matches / total_words if total_words > 0 else 0.0

        # Boost confidence for longer segments
        length_boost = min(0.2, total_words * 0.02)
        confidence += length_boost

        # Script consistency bonus
        script_consistency = self._check_script_consistency(words, detected_language)
        confidence += script_consistency * 0.3

        return min(1.0, confidence)

    def _check_script_consistency(self, words: List[str], language: str) -> float:
        """
        Check script consistency within a segment

        Args:
            words: List of words to check
            language: Expected language

        Returns:
            Consistency score between 0 and 1
        """
        if not words:
            return 0.0

        expected_script = 'cyrillic' if language == 'ru' else 'latin'
        consistent_words = 0

        for word in words:
            word_alpha = ''.join(c for c in word.lower() if c.isalpha())
            if not word_alpha:
                continue

            has_cyrillic = bool(re.search(r'[а-яё]', word_alpha))
            has_latin = bool(re.search(r'[a-z]', word_alpha))

            if expected_script == 'cyrillic' and has_cyrillic and not has_latin:
                consistent_words += 1
            elif expected_script == 'latin' and has_latin and not has_cyrillic:
                consistent_words += 1

        return consistent_words / len(words) if words else 0.0

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)

        # Split on whitespace and basic punctuation
        words = re.findall(r"\b\w+(?:['\']\\w+)?\b", text)

        return [w for w in words if w and len(w.strip()) > 0]

    def _merge_similar_segments(
        self,
        segments: List[CodeSwitchSegment]
    ) -> List[CodeSwitchSegment]:
        """
        Merge adjacent segments with the same language

        Args:
            segments: List of segments to merge

        Returns:
            Merged segments
        """
        if len(segments) <= 1:
            return segments

        merged = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            if (current_segment.language == next_segment.language and
                current_segment.confidence > 0.5 and
                next_segment.confidence > 0.5):

                # Merge segments
                merged_text = current_segment.text + ' ' + next_segment.text
                merged_confidence = (
                    current_segment.confidence * current_segment.word_count +
                    next_segment.confidence * next_segment.word_count
                ) / (current_segment.word_count + next_segment.word_count)

                current_segment = CodeSwitchSegment(
                    text=merged_text,
                    language=current_segment.language,
                    start_idx=current_segment.start_idx,
                    end_idx=next_segment.end_idx,
                    confidence=merged_confidence,
                    word_count=current_segment.word_count + next_segment.word_count
                )
            else:
                merged.append(current_segment)
                current_segment = next_segment

        merged.append(current_segment)
        return merged

    def _load_config(self, config_path: str):
        """
        Load additional patterns from configuration file

        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'uzbek_patterns' in config:
                self.uz_patterns.extend(config['uzbek_patterns'])
                self.uz_regex = [re.compile(p, re.IGNORECASE) for p in self.uz_patterns]

            if 'russian_patterns' in config:
                self.ru_patterns.extend(config['russian_patterns'])
                self.ru_regex = [re.compile(p, re.IGNORECASE) for p in self.ru_patterns]

            logger.info(f"Loaded additional patterns from {config_path}")

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")


def main():
    """Example usage of CodeSwitchDetector"""
    detector = CodeSwitchDetector()

    # Test examples
    test_texts = [
        "Men bugun bozorga boraman",  # Pure Uzbek
        "Я сегодня иду на рынок",     # Pure Russian
        "Men bugun рынок ga boraman", # Code-switched
        "Это yomon fikr, можно buni qilmaslik kerak",  # Mixed
        "Bu juda хорошо natija",      # Code-switched
    ]

    for text in test_texts:
        print(f"\nText: {text}")
        segments = detector.detect_segments(text)

        for seg in segments:
            print(f"  [{seg.language}] '{seg.text}' (conf: {seg.confidence:.2f})")

        stats = detector.analyze_text_statistics(text)
        print(f"  Stats: {stats}")


if __name__ == "__main__":
    main()