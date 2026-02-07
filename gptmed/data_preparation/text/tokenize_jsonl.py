"""
Parallel Tokenization Pipeline

Takes preprocessed JSONL and generates tokens for model training.
Supports parallel processing with ThreadPoolExecutor for efficient batch processing.

Supports multiple tokenization methods:
1. Hugging Face Transformers (AutoTokenizer)
2. SentencePiece
3. Built-in Tokenizer from data_preparation module

Parallel Processing:
- Automatically scales based on dataset size
- ThreadPoolExecutor for I/O-bound tokenization
- Separate outputs for individual documents and merged files

Usage:
    python3 tokenize_jsonl.py \
        --input-file ./output/full_preprocessed.jsonl \
        --method huggingface \
        --model gpt2 \
        --workers 4 \
        --output-dir ./output/tokens
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Adjust path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gptmed.data_preparation.text import Tokenizer as CustomTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimplifiedTokenizedRecord:
    """Simplified tokenized record - only final tokens"""
    filename: str
    tokens: List[int]  # Token IDs
    token_count: int
    word_count: int  # Number of words in original text
    character_count: int  # Number of characters in original text
    tokenizer_method: str
    status: str


class ParallelJSONLTokenizer:
    """Parallel tokenization for preprocessed JSONL files"""
    
    def __init__(
        self,
        input_file: str,
        output_dir: str = None,
        method: str = 'huggingface',
        model_name: Optional[str] = None,
        workers: Optional[int] = None,
    ):
        """
        Initialize parallel tokenizer
        
        Args:
            input_file: Input preprocessed JSONL file
            output_dir: Output directory for tokens
            method: Tokenization method (custom/huggingface/sentencepiece)
            model_name: Model name for huggingface/sentencepiece
            workers: Number of worker threads (auto-detect if None)
        """
        self.input_file = Path(input_file)
        
        if output_dir is None:
            output_dir = self.input_file.parent / 'tokens'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.method = method
        self.model_name = model_name
        
        # Auto-detect workers based on CPU count
        if workers is None:
            workers = max(2, min(4, os.cpu_count() or 2))
        self.workers = workers
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize tokenizer based on method
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer based on method"""
        if self.method == 'custom':
            self.tokenizer = CustomTokenizer(mode='word')
            self.logger.info("Initialized custom tokenizer (word mode)")
        
        elif self.method == 'huggingface':
            try:
                from transformers import AutoTokenizer
                if self.model_name is None:
                    self.model_name = 'gpt2'
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.logger.info(f"Initialized Hugging Face tokenizer ({self.model_name})")
            except ImportError:
                raise ImportError("Install transformers: pip install transformers")
        
        elif self.method == 'sentencepiece':
            try:
                import sentencepiece as spm
                if self.model_name is None:
                    raise ValueError("SentencePiece requires model_name parameter")
                self.tokenizer = spm.SentencePieceProcessor(model_file=self.model_name)
                self.logger.info(f"Initialized SentencePiece tokenizer ({self.model_name})")
            except ImportError:
                raise ImportError("Install sentencepiece: pip install sentencepiece")
        
        else:
            raise ValueError(f"Unknown tokenization method: {self.method}")
    
    def load_jsonl(self) -> List[Dict[str, Any]]:
        """Load preprocessed JSONL"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        records = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Line {line_num}: Invalid JSON - {str(e)}")
        
        self.logger.info(f"Loaded {len(records)} records from {self.input_file.name}")
        return records
    
    def _tokenize_with_custom(self, text: str) -> List[str]:
        """Tokenize using custom tokenizer"""
        return self.tokenizer.tokenize(text)
    
    def _tokenize_with_huggingface(self, text: str) -> List[int]:
        """Tokenize using Hugging Face tokenizer - returns token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def _tokenize_with_sentencepiece(self, text: str) -> List[int]:
        """Tokenize using SentencePiece - returns token IDs"""
        return self.tokenizer.encode(text)
    
    def tokenize_record(self, record: Dict[str, Any], index: int = 0) -> Tuple[SimplifiedTokenizedRecord, str]:
        """
        Tokenize a single record
        
        Returns:
            Tuple of (TokenizedRecord, filename_clean)
        """
        try:
            filename = record.get('filename', f'document_{index}')
            text = record.get('text', '')
            
            # Calculate text statistics
            word_count = len(text.split()) if text else 0
            character_count = len(text) if text else 0
            
            if not text or not isinstance(text, str) or len(text.strip()) < 10:
                return (
                    SimplifiedTokenizedRecord(
                        filename=filename,
                        tokens=[],
                        token_count=0,
                        word_count=word_count,
                        character_count=character_count,
                        tokenizer_method=self.method,
                        status='empty_or_invalid_text'
                    ),
                    filename
                )
            
            # Tokenize based on method
            if self.method == 'custom':
                tokens = self._tokenize_with_custom(text)
                # Convert to integers if needed
                if tokens and isinstance(tokens[0], str):
                    tokens = list(range(len(tokens)))  # Fallback
            elif self.method == 'huggingface':
                tokens = self._tokenize_with_huggingface(text)
            elif self.method == 'sentencepiece':
                tokens = self._tokenize_with_sentencepiece(text)
            else:
                tokens = []
            
            return (
                SimplifiedTokenizedRecord(
                    filename=filename,
                    tokens=tokens,
                    token_count=len(tokens),
                    word_count=word_count,
                    character_count=character_count,
                    tokenizer_method=self.method,
                    status='success'
                ),
                filename
            )
            
        except Exception as e:
            self.logger.error(f"Error tokenizing {record.get('filename', 'unknown')}: {str(e)}")
            text = record.get('text', '')
            word_count = len(text.split()) if text else 0
            character_count = len(text) if text else 0
            return (
                SimplifiedTokenizedRecord(
                    filename=record.get('filename', f'document_{index}'),
                    tokens=[],
                    token_count=0,
                    word_count=word_count,
                    character_count=character_count,
                    tokenizer_method=self.method,
                    status=f'error: {str(e)}'
                ),
                record.get('filename', f'document_{index}')
            )
    
    def save_merged_jsonl(self, records: List[SimplifiedTokenizedRecord]) -> bool:
        """Save all tokenized records to merged file"""
        try:
            output_file = self.output_dir / 'merged_tokens.jsonl'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    json.dump(asdict(record), f, ensure_ascii=False)
                    f.write('\n')
            
            self.logger.info(f"✓ Saved {len(records)} merged tokens to {output_file.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving merged JSONL: {str(e)}")
            return False
    
    def save_token_summary(self, records: List[SimplifiedTokenizedRecord]) -> bool:
        """Save token statistics summary to JSON file"""
        try:
            successful = [r for r in records if r.status == 'success']
            
            # Calculate statistics
            total_tokens = sum(r.token_count for r in successful)
            total_words = sum(r.word_count for r in successful)
            total_characters = sum(r.character_count for r in successful)
            avg_tokens_per_file = total_tokens / len(successful) if successful else 0
            avg_words_per_file = total_words / len(successful) if successful else 0
            avg_chars_per_file = total_characters / len(successful) if successful else 0
            
            # Build per-file summary
            file_summaries = []
            for record in records:
                file_summaries.append({
                    'filename': record.filename,
                    'tokens': record.token_count,
                    'words': record.word_count,
                    'characters': record.character_count,
                    'status': record.status,
                })
            
            # Overall statistics
            summary = {
                'metadata': {
                    'tokenizer_method': self.method,
                    'tokenizer_model': self.model_name,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                },
                'overall_statistics': {
                    'total_files': len(records),
                    'successfully_processed': len(successful),
                    'failed': len(records) - len(successful),
                    'total_tokens': total_tokens,
                    'total_words': total_words,
                    'total_characters': total_characters,
                    'average_tokens_per_file': round(avg_tokens_per_file, 2),
                    'average_words_per_file': round(avg_words_per_file, 2),
                    'average_characters_per_file': round(avg_chars_per_file, 2),
                },
                'per_file_summary': file_summaries,
            }
            
            # Save to JSON
            output_file = self.output_dir / 'token_stats.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✓ Saved token summary to {output_file.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving token summary: {str(e)}")
            return False
    
    def process(self) -> Dict[str, Any]:
        """Main parallel tokenization pipeline"""
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info("Parallel Tokenization Pipeline")
        self.logger.info("="*70)
        self.logger.info(f"Input file: {self.input_file}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Method: {self.method}")
        self.logger.info(f"Workers: {self.workers}")
        if self.model_name:
            self.logger.info(f"Model: {self.model_name}")
        self.logger.info("="*70)
        
        # Load records
        records = self.load_jsonl()
        
        if not records:
            self.logger.warning("No records to tokenize")
            return {'status': 'failure', 'total_records': 0}
        
        # Parallel tokenization
        self.logger.info(f"\nTokenizing {len(records)} records with {self.workers} workers...")
        tokenized_records = []
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.tokenize_record, record, idx): idx
                for idx, record in enumerate(records)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    tokenized, filename = future.result()
                    tokenized_records.append(tokenized)
                    
                    completed += 1
                    if completed % max(1, len(records) // 10) == 0:
                        self.logger.info(f"Progress: {completed}/{len(records)} records tokenized")
                
                except Exception as e:
                    self.logger.error(f"Error processing record: {str(e)}")
        
        # Save merged file
        self.logger.info(f"\nSaving merged tokenized records...")
        saved = self.save_merged_jsonl(tokenized_records)
        
        # Save summary statistics
        self.logger.info(f"Saving token summary statistics...")
        summary_saved = self.save_token_summary(tokenized_records)
        
        # Calculate statistics
        successful = [r for r in tokenized_records if r.status == 'success']
        failed = [r for r in tokenized_records if r.status != 'success']
        
        total_time = time.time() - start_time
        
        # Statistics
        total_tokens = sum(r.token_count for r in successful)
        avg_tokens_per_file = total_tokens / len(successful) if successful else 0
        
        # Print summary
        self.logger.info(f"\n" + "="*70)
        self.logger.info(f"Tokenization Summary")
        self.logger.info(f"="*70)
        self.logger.info(f"Total records: {len(records)}")
        self.logger.info(f"Successfully tokenized: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")
        self.logger.info(f"\nToken Statistics:")
        self.logger.info(f"  Total tokens: {total_tokens:,}")
        self.logger.info(f"  Average tokens per file: {avg_tokens_per_file:.1f}")
        self.logger.info(f"\nOutput File:")
        self.logger.info(f"  {self.output_dir}/merged_tokens.jsonl")
        self.logger.info(f"\nTotal time: {total_time:.2f}s")
        self.logger.info(f"Throughput: {len(records)/total_time:.2f} records/sec")
        self.logger.info(f"="*70)
        
        return {
            'status': 'success' if saved else 'failure',
            'input_file': str(self.input_file),
            'output_dir': str(self.output_dir),
            'total_records': len(records),
            'successful': len(successful),
            'failed': len(failed),
            'total_tokens': total_tokens,
            'average_tokens_per_record': avg_tokens_per_file,
            'tokenizer_method': self.method,
            'workers_used': self.workers,
            'total_time': total_time,
            'throughput': len(records) / total_time,
        }




if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parallel tokenize preprocessed JSONL files'
    )
    parser.add_argument(
        '--input-file',
        default='./output/full_preprocessed.jsonl',
        help='Input preprocessed JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for tokens (auto-generated if not specified)'
    )
    parser.add_argument(
        '--method',
        default='huggingface',
        choices=['custom', 'huggingface', 'sentencepiece'],
        help='Tokenization method (default: huggingface)'
    )
    parser.add_argument(
        '--model',
        default='gpt2',
        help='Model name for huggingface/sentencepiece (default: gpt2)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker threads (auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    tokenizer = ParallelJSONLTokenizer(
        input_file=args.input_file,
        output_dir=args.output_dir,
        method=args.method,
        model_name=args.model,
        workers=args.workers,
    )
    
    result = tokenizer.process()
    sys.exit(0 if result['status'] == 'success' else 1)
