#!/usr/bin/env python3
"""
Evaluation script for multilingual Uzbek Whisper model
Comprehensive evaluation including WER, CER, code-switching accuracy, and language detection
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import time

import torch
from tqdm import tqdm
from transformers import WhisperProcessor

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.architectures.whisper_multilingual import MultilingualWhisperForConditionalGeneration
from src.evaluation.metrics.comprehensive_evaluator import ComprehensiveASREvaluator
from src.utils.audio_utils import AudioProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate multilingual Uzbek Whisper model")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model"
    )

    parser.add_argument(
        "--test-manifest",
        type=str,
        required=True,
        help="Path to test manifest file"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on (auto, cuda, cpu)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=["uz", "ru", "mixed"],
        help="Languages to evaluate"
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for generation"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()

def load_test_data(manifest_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load test data from manifest file"""
    test_samples = []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_samples and line_num >= max_samples:
                break

            try:
                sample = json.loads(line.strip())
                test_samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num + 1}: {e}")

    logger.info(f"Loaded {len(test_samples)} test samples")
    return test_samples

def load_model_and_processor(model_path: str, device: str):
    """Load model and processor"""
    logger.info(f"Loading model from {model_path}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    try:
        if os.path.isfile(model_path) and (model_path.endswith('.pt') or model_path.endswith('.pth')):
            # Load torchscript model
            model = torch.jit.load(model_path, map_location=device)
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        else:
            # Load Hugging Face model
            model = MultilingualWhisperForConditionalGeneration.from_pretrained(model_path)
            model.to(device)
            processor = WhisperProcessor.from_pretrained(model_path)

        model.eval()
        logger.info(f"Model loaded successfully on {device}")

        return model, processor, device

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def run_inference(
    model,
    processor,
    audio_path: str,
    device: str,
    beam_size: int = 5,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Run inference on a single audio file"""
    try:
        # Load and process audio
        audio_processor = AudioProcessor()
        audio_array = audio_processor.load_audio(audio_path)

        # Prepare input
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Generate transcription
            generated_ids = model.generate(
                inputs['input_features'],
                max_length=448,
                num_beams=beam_size,
                temperature=temperature,
                do_sample=temperature > 0,
                early_stopping=True
            )

            # Decode
            transcription = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # Get model outputs for analysis
            outputs = model(
                input_features=inputs['input_features'],
                return_dict=True
            )

            # Extract language information if available
            language_detected = "unknown"
            confidence = 1.0

            if hasattr(outputs, 'language_probs') and outputs.language_probs is not None:
                language_probs = torch.softmax(outputs.language_probs, dim=-1)
                language_idx = torch.argmax(language_probs, dim=-1).item()
                confidence = language_probs.max().item()

                language_map = {0: 'uz', 1: 'ru', 2: 'mixed'}
                language_detected = language_map.get(language_idx, 'unknown')

        return {
            'transcription': transcription,
            'language_detected': language_detected,
            'confidence': confidence
        }

    except Exception as e:
        logger.error(f"Inference failed for {audio_path}: {e}")
        return {
            'transcription': "",
            'language_detected': "unknown",
            'confidence': 0.0
        }

def evaluate_model(
    model,
    processor,
    test_samples: List[Dict[str, Any]],
    device: str,
    beam_size: int = 5,
    temperature: float = 0.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """Evaluate model on test samples"""
    logger.info("Starting model evaluation...")

    predictions = []
    references = []
    language_predictions = []
    language_references = []
    confidence_scores = []
    audio_durations = []
    processing_times = []

    # Process each sample
    for i, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        start_time = time.time()

        # Run inference
        result = run_inference(
            model=model,
            processor=processor,
            audio_path=sample['audio_path'],
            device=device,
            beam_size=beam_size,
            temperature=temperature
        )

        processing_time = time.time() - start_time

        # Collect results
        predictions.append(result['transcription'])
        references.append(sample['text'])
        language_predictions.append(result['language_detected'])
        language_references.append(sample.get('language', 'unknown'))
        confidence_scores.append(result['confidence'])
        processing_times.append(processing_time)

        # Get audio duration if available
        duration = sample.get('duration', 0.0)
        audio_durations.append(duration)

        if verbose and i < 10:  # Show first 10 samples in verbose mode
            logger.info(f"Sample {i + 1}:")
            logger.info(f"  Reference: {sample['text']}")
            logger.info(f"  Prediction: {result['transcription']}")
            logger.info(f"  Language (ref/pred): {sample.get('language', 'unknown')}/{result['language_detected']}")
            logger.info(f"  Confidence: {result['confidence']:.3f}")
            logger.info(f"  Processing time: {processing_time:.3f}s")

    # Calculate comprehensive metrics
    logger.info("Calculating evaluation metrics...")
    evaluator = ComprehensiveASREvaluator()

    eval_results = evaluator.evaluate_batch(
        predictions=predictions,
        references=references,
        language_predictions=language_predictions,
        language_references=language_references,
        confidence_scores=confidence_scores,
        audio_durations=audio_durations
    )

    # Add performance metrics
    total_audio_duration = sum(audio_durations)
    total_processing_time = sum(processing_times)
    avg_processing_time = total_processing_time / len(processing_times)
    real_time_factor = total_processing_time / max(total_audio_duration, 1)

    performance_metrics = {
        'total_samples': len(test_samples),
        'total_audio_duration': total_audio_duration,
        'total_processing_time': total_processing_time,
        'avg_processing_time': avg_processing_time,
        'real_time_factor': real_time_factor,
        'throughput_samples_per_second': len(test_samples) / total_processing_time
    }

    # Combine all results
    final_results = {
        'evaluation_metrics': {
            'wer': eval_results.wer,
            'cer': eval_results.cer,
            'mer': eval_results.mer,
            'wil': eval_results.wil,
            'language_specific_wer': {
                'uzbek': eval_results.uzbek_wer,
                'russian': eval_results.russian_wer,
                'mixed': eval_results.mixed_wer
            },
            'language_detection_accuracy': eval_results.language_detection_accuracy,
            'code_switching': {
                'f1': eval_results.code_switch_f1,
                'precision': eval_results.switch_detection_precision,
                'recall': eval_results.switch_detection_recall
            },
            'semantic_similarity': eval_results.semantic_similarity
        },
        'performance_metrics': performance_metrics,
        'error_analysis': eval_results.error_analysis,
        'language_confusion_matrix': eval_results.language_confusion_matrix,
        'evaluation_config': {
            'beam_size': beam_size,
            'temperature': temperature,
            'device': device,
            'model_path': None  # Will be set by caller
        }
    }

    return final_results

def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    logger.info("Starting model evaluation...")
    logger.info(f"Arguments: {args}")

    try:
        # Load test data
        test_samples = load_test_data(args.test_manifest, args.max_samples)

        if not test_samples:
            logger.error("No test samples loaded")
            sys.exit(1)

        # Load model and processor
        model, processor, device = load_model_and_processor(args.model_path, args.device)

        # Run evaluation
        results = evaluate_model(
            model=model,
            processor=processor,
            test_samples=test_samples,
            device=device,
            beam_size=args.beam_size,
            temperature=args.temperature,
            verbose=args.verbose
        )

        # Add model path to results
        results['evaluation_config']['model_path'] = args.model_path

        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {output_path}")

        # Print summary
        metrics = results['evaluation_metrics']
        performance = results['performance_metrics']

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"WER: {metrics['wer']:.4f}")
        print(f"CER: {metrics['cer']:.4f}")
        print(f"Language Detection Accuracy: {metrics['language_detection_accuracy']:.4f}")
        print(f"Code-Switching F1: {metrics['code_switching']['f1']:.4f}")

        print(f"\nLanguage-specific WER:")
        for lang, wer in metrics['language_specific_wer'].items():
            print(f"  {lang}: {wer:.4f}")

        print(f"\nPerformance:")
        print(f"  Samples: {performance['total_samples']}")
        print(f"  Total audio: {performance['total_audio_duration']:.1f}s")
        print(f"  Processing time: {performance['total_processing_time']:.1f}s")
        print(f"  Real-time factor: {performance['real_time_factor']:.2f}x")
        print(f"  Throughput: {performance['throughput_samples_per_second']:.1f} samples/s")

        print("\n" + "="*60)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()