"""
Text Generator

PURPOSE:
Main class for text generation using trained GPT model.
Combines model loading, tokenization, and decoding strategies.

WHAT THIS FILE DOES:
1. Load trained model from checkpoint
2. Generate text autoregressively (token by token)
3. Apply sampling strategies and repetition control
4. Convert tokens back to text

GENERATION PROCESS:
1. Start with prompt tokens
2. Loop:
   a. Model forward pass → logits
   b. Apply repetition penalty
   c. Sample next token
   d. Append to sequence
   e. Check stopping criteria
3. Decode tokens to text

PACKAGES USED:
- torch: PyTorch
- sentencepiece: Tokenizer

FILES FROM THIS PROJECT:
- model/architecture/transformer.py: GPT model
- model/configs/model_config.py: Model config
- inference/sampling.py: Sampling strategies
- inference/decoding_utils.py: Repetition control
- utils/checkpoints.py: Load checkpoints
"""

import torch
import sentencepiece as spm
from pathlib import Path
from typing import List, Optional

from llm_med.model.architecture import GPTTransformer
from llm_med.model.configs.model_config import ModelConfig
from llm_med.inference.generation_config import GenerationConfig
from llm_med.inference.sampling import sample_next_token
from llm_med.inference.decoding_utils import (
    apply_repetition_penalty,
    block_ngram_repeats,
    should_stop_generation,
)


class TextGenerator:
    """
    Text generation with trained GPT model.

    This is your interface for inference.
    """

    def __init__(
        self, model: GPTTransformer, tokenizer: spm.SentencePieceProcessor, device: str = "cuda"
    ):
        """
        Args:
            model: Trained GPT model
            tokenizer: SentencePiece tokenizer
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, tokenizer_path: Path, device: str = "cuda"):
        """
        Load generator from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer .model file
            device: Device to load on

        Returns:
            TextGenerator instance
        """
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model from config
        model_config_dict = checkpoint["model_config"]
        model_config = ModelConfig(**model_config_dict)
        model = GPTTransformer(model_config)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        print(f"Model loaded (step {checkpoint['step']})")
        print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")

        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_path}")
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))

        return cls(model, tokenizer, device)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode_as_ids(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode_ids(token_ids)

    def generate(
        self, prompt: str, gen_config: GenerationConfig = None, verbose: bool = False
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            gen_config: Generation configuration
            verbose: Print generation progress

        Returns:
            Generated text

        Process:
        1. Encode prompt to tokens
        2. Generate tokens autoregressively
        3. Decode back to text
        """
        if gen_config is None:
            gen_config = GenerationConfig()

        # Encode prompt
        prompt_tokens = self.encode(prompt)

        if verbose:
            print(f"Prompt: {prompt}")
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Generating...")

        # Generate tokens
        generated_tokens = self.generate_tokens(
            prompt_tokens=prompt_tokens, gen_config=gen_config, verbose=verbose
        )

        # Decode to text
        generated_text = self.decode(generated_tokens)

        if verbose:
            print(f"\nGenerated {len(generated_tokens)} tokens")
            print(f"Output: {generated_text}")

        return generated_text

    def generate_tokens(
        self, prompt_tokens: List[int], gen_config: GenerationConfig, verbose: bool = False
    ) -> List[int]:
        """
        Generate tokens autoregressively.

        Args:
            prompt_tokens: Input token IDs
            gen_config: Generation config
            verbose: Print progress

        Returns:
            List of generated token IDs (including prompt)
        """
        # Start with prompt
        generated = prompt_tokens.copy()

        # Generation loop
        with torch.no_grad():
            for step in range(gen_config.max_length):
                # Check stopping criteria
                if should_stop_generation(
                    generated_tokens=generated,
                    stop_tokens=gen_config.stop_tokens,
                    max_length=gen_config.max_length,
                    min_length=gen_config.min_length,
                ):
                    break

                # Prepare input (last max_seq_len tokens)
                max_seq_len = self.model.config.max_seq_len
                input_ids = generated[-max_seq_len:]
                input_tensor = torch.tensor([input_ids], device=self.device)

                # Forward pass
                logits = self.model(input_tensor)

                # Get logits for last position
                next_token_logits = logits[0, -1, :]  # [vocab_size]

                # Apply repetition penalty
                if gen_config.repetition_penalty != 1.0:
                    generated_tensor = torch.tensor([generated], device=self.device)
                    next_token_logits = apply_repetition_penalty(
                        next_token_logits.unsqueeze(0),
                        generated_tensor,
                        gen_config.repetition_penalty,
                    ).squeeze(0)

                # Block n-gram repeats
                if gen_config.no_repeat_ngram_size > 0:
                    next_token_logits = block_ngram_repeats(
                        next_token_logits, generated, gen_config.no_repeat_ngram_size
                    )

                # Sample next token
                next_token = sample_next_token(
                    next_token_logits.unsqueeze(0),
                    temperature=gen_config.temperature,
                    top_k=gen_config.top_k,
                    top_p=gen_config.top_p,
                )

                next_token_id = next_token.item()
                generated.append(next_token_id)

                if verbose and step % 10 == 0:
                    partial_text = self.decode(generated)
                    print(f"Step {step}: {partial_text}")

        return generated

    def generate_batch(self, prompts: List[str], gen_config: GenerationConfig = None) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            gen_config: Generation config

        Returns:
            List of generated texts
        """
        if gen_config is None:
            gen_config = GenerationConfig()

        results = []
        for prompt in prompts:
            output = self.generate(prompt, gen_config, verbose=False)
            results.append(output)

        return results
