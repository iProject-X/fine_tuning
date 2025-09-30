
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load your fine-tuned model
processor = WhisperProcessor.from_pretrained("./models/whisper-uz-basic")
model = WhisperForConditionalGeneration.from_pretrained("./models/whisper-uz-basic", local_files_only=True)

# Use for inference
# audio_input = ...  # your audio data
# result = model.generate(audio_input)
print("Uzbek Whisper model loaded successfully!")
