#!/usr/bin/env python3
"""
Simple test script for the Uzbek Whisper API
"""

import requests
import json
import time
from pathlib import Path

def test_health_endpoint():
    """Test health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
    return False

def test_root_endpoint():
    """Test root endpoint"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
    except Exception as e:
        print(f"Root endpoint failed: {e}")
    return False

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get("http://localhost:8000/models/info", timeout=5)
        print(f"Model info: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
    except Exception as e:
        print(f"Model info failed: {e}")
    return False

def create_dummy_audio():
    """Create a dummy audio file for testing"""
    try:
        import numpy as np
        import wave

        # Generate 3 seconds of sine wave (440 Hz)
        sample_rate = 16000
        duration = 3
        frequency = 440

        t = np.linspace(0, duration, sample_rate * duration, False)
        audio_data = np.sin(2 * np.pi * frequency * t)

        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        # Save as WAV file
        with wave.open('test_audio.wav', 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print("✅ Dummy audio file created: test_audio.wav")
        return True

    except Exception as e:
        print(f"❌ Failed to create dummy audio: {e}")
        return False

def test_transcription():
    """Test transcription endpoint"""
    # Create dummy audio if not exists
    if not Path('test_audio.wav').exists():
        if not create_dummy_audio():
            return False

    try:
        with open('test_audio.wav', 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            data = {
                'language_hint': 'uz',
                'enable_language_detection': True,
                'enable_code_switch_detection': True
            }

            print("🎵 Testing transcription...")
            response = requests.post(
                "http://localhost:8000/transcribe",
                files=files,
                data=data,
                timeout=30
            )

            print(f"Transcription: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Transcription result:")
                print(f"  Text: {result.get('text', 'N/A')}")
                print(f"  Language: {result.get('language_detected', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 'N/A')}")
                print(f"  Processing time: {result.get('processing_time', 'N/A')}s")
                return True
            else:
                print(f"❌ Transcription failed: {response.text}")

    except Exception as e:
        print(f"❌ Transcription test failed: {e}")

    return False

def main():
    """Main test function"""
    print("🧪 Testing Uzbek Whisper API")
    print("=" * 40)

    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Model Info", test_model_info),
        ("Transcription", test_transcription)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)

        start_time = time.time()
        success = test_func()
        duration = time.time() - start_time

        results[test_name] = {
            'success': success,
            'duration': duration
        }

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} ({duration:.2f}s)")

    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)

    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)

    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {test_name}: {result['duration']:.2f}s")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print(f"⚠️ {total - passed} tests failed. Check the logs above.")

    # Cleanup
    if Path('test_audio.wav').exists():
        Path('test_audio.wav').unlink()
        print("\n🗑️ Cleanup: test audio file removed")

if __name__ == "__main__":
    main()