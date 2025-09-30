"""
Comprehensive evaluation metrics for multilingual ASR system
Includes WER, CER, code-switching accuracy, and language detection metrics
"""

import jiwer
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import editdistance
import logging
from dataclasses import dataclass
import json
import math

from ...data.processors.code_switch_detector import CodeSwitchDetector

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    wer: float
    cer: float
    mer: float
    wil: float
    uzbek_wer: float
    russian_wer: float
    mixed_wer: float
    language_detection_accuracy: float
    code_switch_f1: float
    switch_detection_precision: float
    switch_detection_recall: float
    semantic_similarity: float
    error_analysis: Dict[str, Any]
    language_confusion_matrix: List[List[int]]
    per_sample_results: List[Dict]

class ComprehensiveASREvaluator:
    """
    Advanced evaluation suite for multilingual ASR with code-switching

    Metrics computed:
    - Standard ASR metrics (WER, CER, MER, WIL)
    - Language-specific WER
    - Code-switching detection accuracy
    - Language detection performance
    - Error pattern analysis
    - Confidence calibration metrics
    """

    def __init__(self, languages: List[str] = None):
        """
        Initialize evaluator

        Args:
            languages: List of supported languages
        """
        self.languages = languages or ['uz', 'ru', 'mixed']
        self.code_switch_detector = CodeSwitchDetector()

        # Text normalization for consistent evaluation
        self.jiwer_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemoveEmptyStrings()
        ])

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        language_predictions: Optional[List[str]] = None,
        language_references: Optional[List[str]] = None,
        confidence_scores: Optional[List[float]] = None,
        audio_durations: Optional[List[float]] = None
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of ASR predictions

        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            language_predictions: Predicted languages for each sample
            language_references: Reference languages for each sample
            confidence_scores: Confidence scores for predictions
            audio_durations: Duration of each audio sample in seconds

        Returns:
            EvaluationResult with comprehensive metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        logger.info(f"Evaluating {len(predictions)} samples")

        # Standard ASR metrics
        wer = self._calculate_wer(predictions, references)
        cer = self._calculate_cer(predictions, references)
        mer = self._calculate_mer(predictions, references)
        wil = self._calculate_wil(predictions, references)

        # Language-specific metrics
        language_specific_wer = self._calculate_language_specific_wer(
            predictions, references, language_references
        )

        # Code-switching metrics
        code_switch_metrics = self._evaluate_code_switching(predictions, references)

        # Language detection metrics
        language_detection_metrics = self._evaluate_language_detection(
            language_predictions, language_references
        ) if language_predictions and language_references else {}

        # Error analysis
        error_analysis = self._analyze_errors(predictions, references)

        # Semantic similarity (if available)
        semantic_similarity = self._calculate_semantic_similarity(
            predictions, references
        )

        # Per-sample results
        per_sample_results = self._calculate_per_sample_metrics(
            predictions, references, language_references, confidence_scores, audio_durations
        )

        return EvaluationResult(
            wer=wer,
            cer=cer,
            mer=mer,
            wil=wil,
            uzbek_wer=language_specific_wer.get('uz', 0.0),
            russian_wer=language_specific_wer.get('ru', 0.0),
            mixed_wer=language_specific_wer.get('mixed', 0.0),
            language_detection_accuracy=language_detection_metrics.get('accuracy', 0.0),
            code_switch_f1=code_switch_metrics.get('f1', 0.0),
            switch_detection_precision=code_switch_metrics.get('precision', 0.0),
            switch_detection_recall=code_switch_metrics.get('recall', 0.0),
            semantic_similarity=semantic_similarity,
            error_analysis=error_analysis,
            language_confusion_matrix=language_detection_metrics.get('confusion_matrix', []),
            per_sample_results=per_sample_results
        )

    def _calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate using jiwer"""
        try:
            return jiwer.wer(
                references,
                predictions,
                truth_transform=self.jiwer_transform,
                hypothesis_transform=self.jiwer_transform
            )
        except Exception as e:
            logger.warning(f"WER calculation failed: {e}")
            return 1.0

    def _calculate_cer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Character Error Rate"""
        try:
            return jiwer.cer(
                references,
                predictions,
                truth_transform=self.jiwer_transform,
                hypothesis_transform=self.jiwer_transform
            )
        except Exception as e:
            logger.warning(f"CER calculation failed: {e}")
            return 1.0

    def _calculate_mer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Match Error Rate"""
        try:
            return jiwer.mer(
                references,
                predictions,
                truth_transform=self.jiwer_transform,
                hypothesis_transform=self.jiwer_transform
            )
        except Exception as e:
            logger.warning(f"MER calculation failed: {e}")
            return 1.0

    def _calculate_wil(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Word Information Lost"""
        try:
            return jiwer.wil(
                references,
                predictions,
                truth_transform=self.jiwer_transform,
                hypothesis_transform=self.jiwer_transform
            )
        except Exception as e:
            logger.warning(f"WIL calculation failed: {e}")
            return 1.0

    def _calculate_language_specific_wer(
        self,
        predictions: List[str],
        references: List[str],
        language_references: Optional[List[str]]
    ) -> Dict[str, float]:
        """Calculate WER for each language separately"""
        if not language_references:
            return {}

        language_wer = {}

        for lang in self.languages:
            # Filter samples for this language
            lang_predictions = []
            lang_references = []

            for pred, ref, lang_ref in zip(predictions, references, language_references):
                if lang_ref == lang:
                    lang_predictions.append(pred)
                    lang_references.append(ref)

            if lang_predictions:
                try:
                    wer = jiwer.wer(
                        lang_references,
                        lang_predictions,
                        truth_transform=self.jiwer_transform,
                        hypothesis_transform=self.jiwer_transform
                    )
                    language_wer[lang] = wer
                except Exception as e:
                    logger.warning(f"WER calculation failed for language {lang}: {e}")
                    language_wer[lang] = 1.0
            else:
                language_wer[lang] = 0.0

        return language_wer

    def _evaluate_code_switching(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Evaluate code-switching detection performance"""
        try:
            switch_detection_results = {
                'correct_switches': 0,
                'predicted_switches': 0,
                'reference_switches': 0
            }

            for pred, ref in zip(predictions, references):
                # Detect switches in both prediction and reference
                pred_segments = self.code_switch_detector.detect_segments(pred)
                ref_segments = self.code_switch_detector.detect_segments(ref)

                pred_switches = self._get_switch_points(pred_segments)
                ref_switches = self._get_switch_points(ref_segments)

                switch_detection_results['predicted_switches'] += len(pred_switches)
                switch_detection_results['reference_switches'] += len(ref_switches)

                # Count correct switches (allowing some tolerance)
                for ref_switch in ref_switches:
                    if any(abs(ref_switch - pred_switch) <= 2 for pred_switch in pred_switches):
                        switch_detection_results['correct_switches'] += 1

            # Calculate precision, recall, F1
            correct = switch_detection_results['correct_switches']
            predicted = switch_detection_results['predicted_switches']
            reference = switch_detection_results['reference_switches']

            precision = correct / max(predicted, 1)
            recall = correct / max(reference, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'correct_switches': correct,
                'predicted_switches': predicted,
                'reference_switches': reference
            }

        except Exception as e:
            logger.warning(f"Code-switching evaluation failed: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def _get_switch_points(self, segments) -> List[int]:
        """Extract switch points from language segments"""
        switch_points = []
        for i in range(1, len(segments)):
            if segments[i].language != segments[i-1].language:
                switch_points.append(segments[i].start_idx)
        return switch_points

    def _evaluate_language_detection(
        self,
        language_predictions: List[str],
        language_references: List[str]
    ) -> Dict[str, Any]:
        """Evaluate language detection accuracy"""
        if not language_predictions or not language_references:
            return {}

        # Calculate accuracy
        correct = sum(
            pred == ref for pred, ref in zip(language_predictions, language_references)
        )
        accuracy = correct / len(language_predictions)

        # Create confusion matrix
        lang_to_idx = {lang: i for i, lang in enumerate(self.languages)}
        confusion_matrix = [[0 for _ in self.languages] for _ in self.languages]

        for pred, ref in zip(language_predictions, language_references):
            if pred in lang_to_idx and ref in lang_to_idx:
                pred_idx = lang_to_idx[pred]
                ref_idx = lang_to_idx[ref]
                confusion_matrix[ref_idx][pred_idx] += 1

        # Calculate per-class metrics
        per_class_metrics = {}
        for i, lang in enumerate(self.languages):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[j][i] for j in range(len(self.languages)) if j != i)
            fn = sum(confusion_matrix[i][j] for j in range(len(self.languages)) if j != i)

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            per_class_metrics[lang] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        return {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'per_class_metrics': per_class_metrics,
            'macro_f1': np.mean([metrics['f1'] for metrics in per_class_metrics.values()])
        }

    def _analyze_errors(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        """Detailed error pattern analysis"""
        substitutions = Counter()
        deletions = Counter()
        insertions = Counter()

        total_words = 0
        total_errors = 0

        for pred, ref in zip(predictions, references):
            # Normalize texts
            pred_normalized = self.jiwer_transform(pred)
            ref_normalized = self.jiwer_transform(ref)

            pred_words = pred_normalized.split()
            ref_words = ref_normalized.split()

            total_words += len(ref_words)

            # Calculate edit operations
            operations = self._get_edit_operations(ref_words, pred_words)

            for op_type, ref_word, pred_word in operations:
                total_errors += 1
                if op_type == 'substitute':
                    substitutions[(ref_word, pred_word)] += 1
                elif op_type == 'delete':
                    deletions[ref_word] += 1
                elif op_type == 'insert':
                    insertions[pred_word] += 1

        return {
            'total_words': total_words,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_words, 1),
            'top_substitutions': substitutions.most_common(10),
            'top_deletions': deletions.most_common(10),
            'top_insertions': insertions.most_common(10),
            'substitution_count': sum(substitutions.values()),
            'deletion_count': sum(deletions.values()),
            'insertion_count': sum(insertions.values())
        }

    def _get_edit_operations(self, ref_words: List[str], pred_words: List[str]) -> List[Tuple[str, str, str]]:
        """Get edit operations between reference and prediction"""
        operations = []

        # Use dynamic programming to find optimal alignment
        m, n = len(ref_words), len(pred_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == pred_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )

        # Backtrack to find operations
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i-1] == pred_words[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(('substitute', ref_words[i-1], pred_words[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                operations.append(('delete', ref_words[i-1], ''))
                i -= 1
            else:
                operations.append(('insert', '', pred_words[j-1]))
                j -= 1

        return operations[::-1]

    def _calculate_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Calculate semantic similarity using simple word overlap"""
        try:
            total_similarity = 0
            valid_pairs = 0

            for pred, ref in zip(predictions, references):
                pred_words = set(self.jiwer_transform(pred).split())
                ref_words = set(self.jiwer_transform(ref).split())

                if ref_words:
                    # Jaccard similarity
                    intersection = len(pred_words & ref_words)
                    union = len(pred_words | ref_words)
                    similarity = intersection / max(union, 1)
                    total_similarity += similarity
                    valid_pairs += 1

            return total_similarity / max(valid_pairs, 1)

        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def _calculate_per_sample_metrics(
        self,
        predictions: List[str],
        references: List[str],
        language_references: Optional[List[str]],
        confidence_scores: Optional[List[float]],
        audio_durations: Optional[List[float]]
    ) -> List[Dict]:
        """Calculate metrics for each individual sample"""
        per_sample_results = []

        for i, (pred, ref) in enumerate(predictions, references):
            # Individual WER
            try:
                sample_wer = jiwer.wer(
                    [ref], [pred],
                    truth_transform=self.jiwer_transform,
                    hypothesis_transform=self.jiwer_transform
                )
            except:
                sample_wer = 1.0

            # Individual CER
            try:
                sample_cer = jiwer.cer(
                    [ref], [pred],
                    truth_transform=self.jiwer_transform,
                    hypothesis_transform=self.jiwer_transform
                )
            except:
                sample_cer = 1.0

            result = {
                'index': i,
                'prediction': pred,
                'reference': ref,
                'wer': sample_wer,
                'cer': sample_cer,
                'length_ratio': len(pred) / max(len(ref), 1)
            }

            if language_references:
                result['language'] = language_references[i]

            if confidence_scores:
                result['confidence'] = confidence_scores[i]

            if audio_durations:
                result['audio_duration'] = audio_durations[i]
                # Calculate real-time factor if we have processing time
                result['rtf'] = None  # Would need processing time to calculate

            per_sample_results.append(result)

        return per_sample_results

    def save_results(self, results: EvaluationResult, output_path: str):
        """Save evaluation results to file"""
        results_dict = {
            'summary': {
                'wer': results.wer,
                'cer': results.cer,
                'mer': results.mer,
                'wil': results.wil,
                'language_specific_wer': {
                    'uzbek': results.uzbek_wer,
                    'russian': results.russian_wer,
                    'mixed': results.mixed_wer
                },
                'language_detection_accuracy': results.language_detection_accuracy,
                'code_switching': {
                    'f1': results.code_switch_f1,
                    'precision': results.switch_detection_precision,
                    'recall': results.switch_detection_recall
                },
                'semantic_similarity': results.semantic_similarity
            },
            'error_analysis': results.error_analysis,
            'language_confusion_matrix': results.language_confusion_matrix,
            'per_sample_results': results.per_sample_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {output_path}")


def main():
    """Example usage of ComprehensiveASREvaluator"""
    evaluator = ComprehensiveASREvaluator()

    # Test data
    predictions = [
        "Men bugun bozorga boraman",
        "Я сегодня иду в магазин",
        "Bu juda yomon idea",
        "Men bugun работать ish joyiga boraman"
    ]

    references = [
        "Men bugun bozorga boraman",
        "Я сегодня иду в магазин",
        "Bu juda yomon fikr",
        "Men bugun ishga boraman"
    ]

    language_predictions = ["uz", "ru", "uz", "mixed"]
    language_references = ["uz", "ru", "uz", "mixed"]

    # Evaluate
    results = evaluator.evaluate_batch(
        predictions=predictions,
        references=references,
        language_predictions=language_predictions,
        language_references=language_references
    )

    print(f"WER: {results.wer:.4f}")
    print(f"CER: {results.cer:.4f}")
    print(f"Language detection accuracy: {results.language_detection_accuracy:.4f}")
    print(f"Code-switching F1: {results.code_switch_f1:.4f}")

    # Save results
    evaluator.save_results(results, "evaluation_results.json")


if __name__ == "__main__":
    main()