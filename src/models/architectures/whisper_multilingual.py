"""
Multilingual Whisper model with language-specific adapters
Optimized for Uzbek-Russian code-switching scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
from dataclasses import dataclass

from transformers import WhisperModel, WhisperConfig, WhisperProcessor
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder, WhisperDecoder, WhisperPreTrainedModel
)

@dataclass
class MultilingualWhisperConfig:
    """Configuration for multilingual Whisper model"""
    base_model_name: str = "openai/whisper-base"
    languages: List[str] = None
    adapter_dim: int = 256
    adapter_dropout: float = 0.1
    language_adapter_layers: List[int] = None  # Which encoder layers to add adapters
    fusion_type: str = "add"  # "add", "concat", "attention"
    language_detection_layers: int = 2
    freeze_base_model: bool = False

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["uz", "ru", "mixed"]
        if self.language_adapter_layers is None:
            self.language_adapter_layers = [2, 4, 6]  # Every other layer

class LanguageSpecificAdapter(nn.Module):
    """
    Bottleneck adapter for language-specific fine-tuning

    Based on "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al.)
    """

    def __init__(self, d_model: int, adapter_dim: int, dropout: float = 0.1):
        super().__init__()

        self.down_project = nn.Linear(d_model, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize with small weights
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.normal_(self.up_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter transformation

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Transformed tensor with same shape
        """
        residual = x

        # Bottleneck transformation
        adapted = self.down_project(x)
        adapted = self.activation(adapted)
        adapted = self.dropout(adapted)
        adapted = self.up_project(adapted)

        # Residual connection and layer norm
        output = self.layer_norm(residual + adapted)

        return output

class LanguageDetectionHead(nn.Module):
    """
    Neural language detection head for determining language mix
    """

    def __init__(self, d_model: int, num_languages: int, num_layers: int = 2):
        super().__init__()

        layers = []
        current_dim = d_model

        for i in range(num_layers - 1):
            hidden_dim = d_model // (2 ** (i + 1))
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_languages))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect language distribution

        Args:
            x: Hidden states (batch_size, seq_len, d_model)

        Returns:
            Language probabilities (batch_size, num_languages)
        """
        # Global average pooling across sequence length
        pooled = x.mean(dim=1)

        # Language classification
        logits = self.classifier(pooled)

        return logits

class MultilingualWhisperEncoder(WhisperEncoder):
    """
    Whisper encoder with language-specific adapters
    """

    def __init__(self, config, multilingual_config: MultilingualWhisperConfig):
        super().__init__(config)

        self.multilingual_config = multilingual_config

        # Language-specific adapters
        self.language_adapters = nn.ModuleDict()
        for language in multilingual_config.languages:
            for layer_idx in multilingual_config.language_adapter_layers:
                adapter_name = f"{language}_layer_{layer_idx}"
                self.language_adapters[adapter_name] = LanguageSpecificAdapter(
                    d_model=config.d_model,
                    adapter_dim=multilingual_config.adapter_dim,
                    dropout=multilingual_config.adapter_dropout
                )

        # Language detection head
        self.language_detector = LanguageDetectionHead(
            d_model=config.d_model,
            num_languages=len(multilingual_config.languages),
            num_layers=multilingual_config.language_detection_layers
        )

        # Fusion mechanism for combining adapter outputs
        if multilingual_config.fusion_type == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        elif multilingual_config.fusion_type == "concat":
            self.fusion_projection = nn.Linear(
                config.d_model * len(multilingual_config.languages),
                config.d_model
            )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        language_hints: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with language-aware processing
        """
        # Standard Whisper encoder forward pass
        encoder_outputs = super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # We need hidden states for adapters
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state
        all_hidden_states = encoder_outputs.hidden_states

        # Language detection
        language_logits = self.language_detector(hidden_states)
        language_probs = F.softmax(language_logits, dim=-1)

        # Apply language-specific adapters to specified layers
        adapter_outputs = {}

        for language in self.multilingual_config.languages:
            language_output = hidden_states

            for layer_idx in self.multilingual_config.language_adapter_layers:
                if layer_idx < len(all_hidden_states):
                    adapter_name = f"{language}_layer_{layer_idx}"
                    if adapter_name in self.language_adapters:
                        layer_hidden = all_hidden_states[layer_idx]
                        adapted_hidden = self.language_adapters[adapter_name](layer_hidden)

                        # Use the final layer's adapted output
                        if layer_idx == max(self.multilingual_config.language_adapter_layers):
                            language_output = adapted_hidden

            adapter_outputs[language] = language_output

        # Fusion of adapter outputs
        fused_output = self._fuse_adapter_outputs(
            adapter_outputs, language_probs, hidden_states
        )

        # Update encoder outputs
        encoder_outputs.last_hidden_state = fused_output
        encoder_outputs.language_logits = language_logits
        encoder_outputs.language_probs = language_probs

        return encoder_outputs

    def _fuse_adapter_outputs(
        self,
        adapter_outputs: Dict[str, torch.Tensor],
        language_probs: torch.Tensor,
        original_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse outputs from language-specific adapters
        """
        if self.multilingual_config.fusion_type == "add":
            # Weighted sum based on language probabilities
            fused = torch.zeros_like(original_hidden)

            for i, language in enumerate(self.multilingual_config.languages):
                if language in adapter_outputs:
                    weight = language_probs[:, i].unsqueeze(-1).unsqueeze(-1)
                    fused += weight * adapter_outputs[language]

            return fused

        elif self.multilingual_config.fusion_type == "concat":
            # Concatenate and project
            concat_outputs = []
            for language in self.multilingual_config.languages:
                if language in adapter_outputs:
                    concat_outputs.append(adapter_outputs[language])
                else:
                    concat_outputs.append(original_hidden)

            concatenated = torch.cat(concat_outputs, dim=-1)
            fused = self.fusion_projection(concatenated)

            return fused

        elif self.multilingual_config.fusion_type == "attention":
            # Attention-based fusion
            adapter_stack = torch.stack([
                adapter_outputs.get(lang, original_hidden)
                for lang in self.multilingual_config.languages
            ], dim=1)  # (batch, num_langs, seq_len, d_model)

            batch_size, num_langs, seq_len, d_model = adapter_stack.shape
            adapter_stack = adapter_stack.view(batch_size, num_langs * seq_len, d_model)

            query = original_hidden  # (batch, seq_len, d_model)
            key_value = adapter_stack

            fused, _ = self.fusion_attention(query, key_value, key_value)

            return fused

        else:
            return original_hidden

class MultilingualWhisperModel(WhisperPreTrainedModel):
    """
    Multilingual Whisper model with language adapters and code-switching support
    """

    def __init__(self, config, multilingual_config: MultilingualWhisperConfig):
        super().__init__(config)

        self.multilingual_config = multilingual_config

        # Load base Whisper model components
        base_model = WhisperModel.from_pretrained(multilingual_config.base_model_name)

        # Replace encoder with multilingual version
        self.encoder = MultilingualWhisperEncoder(config, multilingual_config)

        # Copy weights from base model encoder
        self._copy_encoder_weights(base_model.encoder)

        # Use standard decoder (could be extended with adapters too)
        self.decoder = base_model.decoder

        # Freeze base model if specified
        if multilingual_config.freeze_base_model:
            self._freeze_base_model()

        # Initialize new parameters
        self.post_init()

    def _copy_encoder_weights(self, base_encoder):
        """Copy weights from base encoder to multilingual encoder"""
        # Copy all matching parameters
        base_state_dict = base_encoder.state_dict()
        current_state_dict = self.encoder.state_dict()

        for name, param in base_state_dict.items():
            if name in current_state_dict and current_state_dict[name].shape == param.shape:
                current_state_dict[name].copy_(param)

    def _freeze_base_model(self):
        """Freeze base Whisper parameters, keeping only adapters trainable"""
        # Freeze encoder base parameters
        for name, param in self.encoder.named_parameters():
            if not any(adapter_key in name for adapter_key in ['language_adapters', 'language_detector', 'fusion']):
                param.requires_grad = False

        # Freeze decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        language_hints: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with language-aware processing
        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                language_hints=language_hints,
            )

        # Decoder forward pass
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Combine outputs
        if return_dict:
            from transformers.modeling_outputs import Seq2SeqModelOutput
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                # Add language detection outputs
                language_logits=getattr(encoder_outputs, 'language_logits', None),
                language_probs=getattr(encoder_outputs, 'language_probs', None),
            )

        return decoder_outputs + encoder_outputs

class MultilingualWhisperForConditionalGeneration(WhisperPreTrainedModel):
    """
    Multilingual Whisper model for conditional generation with language detection
    """

    def __init__(self, config, multilingual_config: MultilingualWhisperConfig):
        super().__init__(config)

        self.model = MultilingualWhisperModel(config, multilingual_config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Language detection loss weight
        self.language_loss_weight = 0.1

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        language_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        language_hints: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with loss computation
        """
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            language_hints=language_hints,
        )

        lm_logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # Language modeling loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # Language detection loss
            if language_labels is not None and hasattr(outputs, 'language_logits'):
                language_loss_fct = nn.CrossEntropyLoss()
                language_loss = language_loss_fct(
                    outputs.language_logits.view(-1, len(self.model.multilingual_config.languages)),
                    language_labels.view(-1)
                )
                loss += self.language_loss_weight * language_loss

        if return_dict:
            from transformers.modeling_outputs import Seq2SeqLMOutput
            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
                language_logits=getattr(outputs, 'language_logits', None),
                language_probs=getattr(outputs, 'language_probs', None),
            )

        return (loss, lm_logits) + outputs[1:]

    def generate(
        self,
        input_features: torch.Tensor,
        language_detection: bool = True,
        **generation_kwargs
    ):
        """
        Generate with optional language detection
        """
        if language_detection:
            # First pass to detect language
            with torch.no_grad():
                encoder_outputs = self.model.encoder(input_features, return_dict=True)
                if hasattr(encoder_outputs, 'language_probs'):
                    language_probs = encoder_outputs.language_probs
                    detected_language_idx = torch.argmax(language_probs, dim=-1)

                    # You could use this to set generation parameters
                    # For example, different generation configs for different languages

        return super().generate(input_features, **generation_kwargs)


def create_multilingual_whisper(
    base_model_name: str = "openai/whisper-base",
    languages: List[str] = None,
    adapter_dim: int = 256,
    freeze_base: bool = True
) -> MultilingualWhisperForConditionalGeneration:
    """
    Create a multilingual Whisper model with specified configuration

    Args:
        base_model_name: Base Whisper model to use
        languages: List of language codes
        adapter_dim: Dimension of adapter layers
        freeze_base: Whether to freeze base model parameters

    Returns:
        Configured multilingual Whisper model
    """
    if languages is None:
        languages = ["uz", "ru", "mixed"]

    # Load base config
    base_config = WhisperConfig.from_pretrained(base_model_name)

    # Create multilingual config
    multilingual_config = MultilingualWhisperConfig(
        base_model_name=base_model_name,
        languages=languages,
        adapter_dim=adapter_dim,
        freeze_base_model=freeze_base
    )

    # Create model
    model = MultilingualWhisperForConditionalGeneration(base_config, multilingual_config)

    return model


if __name__ == "__main__":
    # Example usage
    model = create_multilingual_whisper(
        base_model_name="openai/whisper-base",
        languages=["uz", "ru", "mixed"],
        adapter_dim=256,
        freeze_base=True
    )

    # Test forward pass
    batch_size = 2
    n_mels = 80
    sequence_length = 3000

    input_features = torch.randn(batch_size, n_mels, sequence_length)
    decoder_input_ids = torch.randint(0, 51865, (batch_size, 10))

    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        print(f"Logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'language_logits'):
            print(f"Language logits shape: {outputs.language_logits.shape}")
            print(f"Language probabilities: {torch.softmax(outputs.language_logits, dim=-1)}")