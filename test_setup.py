#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly
"""

import sys
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperProcessor
import json

def test_pytorch():
    """Test PyTorch installation"""
    print("🔧 Testing PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

    # Simple tensor operation
    x = torch.randn(2, 3)
    y = torch.randn(3, 2)
    z = torch.mm(x, y)
    print(f"Matrix multiplication test: {z.shape}")
    print("✅ PyTorch working!")

def test_transformers():
    """Test Transformers library"""
    print("\n🤖 Testing Transformers...")

    try:
        # Load Whisper processor (this will download the model)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        print(f"Whisper processor loaded: {type(processor)}")

        # Load model
        model = WhisperModel.from_pretrained("openai/whisper-tiny")
        print(f"Whisper model loaded: {type(model)}")
        print(f"Model config: {model.config}")

        print("✅ Transformers working!")
        return model, processor

    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return None, None

def test_multilingual_model():
    """Test our multilingual model creation"""
    print("\n🌐 Testing Multilingual Model...")

    try:
        # Import our custom model
        sys.path.append('src')
        from models.architectures.whisper_multilingual import create_multilingual_whisper

        model = create_multilingual_whisper(
            base_model_name="openai/whisper-tiny",
            languages=["uz", "ru", "mixed"],
            adapter_dim=64,  # Smaller for testing
            freeze_base=True
        )

        print(f"Multilingual model created: {type(model)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        print("✅ Multilingual model working!")
        return model

    except Exception as e:
        print(f"❌ Multilingual model test failed: {e}")
        return None

def test_code_switch_detector():
    """Test code-switching detector"""
    print("\n🔄 Testing Code-Switch Detector...")

    try:
        sys.path.append('src')
        from data.processors.code_switch_detector import CodeSwitchDetector

        detector = CodeSwitchDetector()

        # Test with mixed text
        test_texts = [
            "Men bugun bozorga boraman",  # Pure Uzbek
            "Я сегодня иду на рынок",     # Pure Russian
            "Men bugun рынок ga boraman", # Code-switched
            "Bu yomon fikr edi",          # Pure Uzbek
            "Это juda yaxshi",            # Code-switched
        ]

        for text in test_texts:
            segments = detector.detect_segments(text)
            print(f"Text: '{text}'")
            for seg in segments:
                print(f"  [{seg.language}] '{seg.text}' (conf: {seg.confidence:.2f})")

        print("✅ Code-switch detector working!")
        return detector

    except Exception as e:
        print(f"❌ Code-switch detector test failed: {e}")
        return None

def test_fastapi():
    """Test FastAPI"""
    print("\n🚀 Testing FastAPI...")

    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"message": "Hello Uzbek Whisper!"}

        client = TestClient(app)
        response = client.get("/test")

        print(f"FastAPI response: {response.json()}")
        print("✅ FastAPI working!")

    except Exception as e:
        print(f"❌ FastAPI test failed: {e}")

def main():
    """Main test function"""
    print("🎯 Testing Uzbek Whisper Setup")
    print("=" * 50)

    # Test basic dependencies
    test_pytorch()
    model, processor = test_transformers()

    # Test our custom components
    if model and processor:
        multilingual_model = test_multilingual_model()
        detector = test_code_switch_detector()
        test_fastapi()

    print("\n" + "=" * 50)
    print("🎉 Setup test completed!")

    # Summary
    summary = {
        "pytorch": "✅",
        "transformers": "✅" if model else "❌",
        "multilingual_model": "✅" if 'multilingual_model' in locals() and multilingual_model else "❌",
        "code_switch_detector": "✅" if 'detector' in locals() and detector else "❌",
        "fastapi": "✅"
    }

    print("\n📊 Test Summary:")
    for component, status in summary.items():
        print(f"  {component}: {status}")

    if all(status == "✅" for status in summary.values()):
        print("\n🚀 All systems ready! Your Uzbek Whisper setup is complete.")
    else:
        print("\n⚠️ Some components failed. Check the error messages above.")

if __name__ == "__main__":
    main()