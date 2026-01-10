"""
Text Formatter for Causal Language Modeling

Converts structured Q&A pairs into causal text format suitable for
next-token prediction training.

Design decisions explained:
- Simple format preserves question-answer structure
- Special tokens ([Q], [A]) help model learn task boundaries
- Newlines create clear separation for tokenizer
- No complex templating - reduces failure modes
"""

from typing import List
from dataclasses import dataclass


@dataclass
class FormatConfig:
    """Configuration for text formatting."""

    use_special_tokens: bool = True  # Use [Q] and [A] markers
    add_separator: bool = True  # Add newline between Q and A
    add_end_token: bool = True  # Add end-of-text marker
    question_prefix: str = "Q: "
    answer_prefix: str = "A: "
    separator: str = "\n"
    end_token: str = "\n\n"  # Double newline = document boundary


class CausalTextFormatter:
    """
    Formats Q&A pairs for causal language modeling.

    Follows Open-Closed Principle: easy to extend with new formats
    without modifying existing code.
    """

    def __init__(self, config: FormatConfig = None):
        """
        Initialize formatter.

        Args:
            config: Formatting configuration
        """
        self.config = config or FormatConfig()

    def format_single_pair(self, question: str, answer: str) -> str:
        """
        Format a single Q&A pair.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            Formatted text string

        Example:
            >>> formatter = CausalTextFormatter()
            >>> formatter.format_single_pair("What is cancer?", "Cancer is...")
            "Q: What is cancer?\\nA: Cancer is...\\n\\n"
        """
        parts = []

        # Add question
        if self.config.use_special_tokens:
            parts.append(f"{self.config.question_prefix}{question}")
        else:
            parts.append(question)

        # Add separator
        if self.config.add_separator:
            parts.append(self.config.separator)

        # Add answer
        if self.config.use_special_tokens:
            parts.append(f"{self.config.answer_prefix}{answer}")
        else:
            parts.append(answer)

        # Add end token
        if self.config.add_end_token:
            parts.append(self.config.end_token)

        return "".join(parts)

    def format_batch(self, qa_pairs: List[tuple]) -> str:
        """
        Format multiple Q&A pairs into a single text corpus.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Single formatted text string
        """
        formatted_pairs = [self.format_single_pair(q, a) for q, a in qa_pairs]
        return "".join(formatted_pairs)

    def format_from_structured(self, qa_objects: List) -> str:
        """
        Format from structured QAPair objects.

        Args:
            qa_objects: List of QAPair objects (from parser)

        Returns:
            Formatted text corpus
        """
        pairs = [(obj.question, obj.answer) for obj in qa_objects]
        return self.format_batch(pairs)


class MinimalFormatter(CausalTextFormatter):
    """
    Minimal format without special tokens.
    Useful for baseline comparisons.
    """

    def __init__(self):
        config = FormatConfig(
            use_special_tokens=False,
            add_separator=True,
            add_end_token=True,
            separator="\n",
            end_token="\n\n",
        )
        super().__init__(config)


class StructuredFormatter(CausalTextFormatter):
    """
    More structured format with explicit markers.
    Better for instruction-tuning style training.
    """

    def __init__(self):
        config = FormatConfig(
            use_special_tokens=True,
            add_separator=True,
            add_end_token=True,
            question_prefix="### Question: ",
            answer_prefix="### Answer: ",
            separator="\n\n",
            end_token="\n\n---\n\n",
        )
        super().__init__(config)
