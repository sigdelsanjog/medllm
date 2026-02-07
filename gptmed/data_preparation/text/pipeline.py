"""
Complete PDF → Tokens → Vocabulary Pipeline

Orchestrates the full preprocessing pipeline:
1. Extract text from PDFs (in-memory)
2. Preprocess text (in-memory)
3. Tokenize (saves merged_tokens.jsonl and token_stats.json)
4. Build Vocabulary (creates vocab.json, token_counts.json, and vocab_info.json)

This is the main entry point for generating training data with complete tokenization and vocabulary information.

Usage:
    python3 pipeline.py \
        --input-dir ./pdfs \
        --output-dir ./output \
        --tokenizer-method huggingface \
        --tokenizer-model gpt2 \
        --workers 4
"""

import sys
import json
import logging
import time
import torch
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict
import argparse
import importlib.util

# Adjust path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load batch_pdf_to_jsonl
spec1 = importlib.util.spec_from_file_location("batch_pdf_to_jsonl", Path(__file__).parent / "batch_pdf_to_jsonl.py")
batch_module = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(batch_module)
PDFBatchProcessor = batch_module.PDFBatchProcessor
PDFRecord = batch_module.PDFRecord

# Load preprocess_jsonl
spec2 = importlib.util.spec_from_file_location("preprocess_jsonl", Path(__file__).parent / "preprocess_jsonl.py")
preprocess_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(preprocess_module)
ComprehensiveJSONLPreprocessor = preprocess_module.ComprehensiveJSONLPreprocessor
FullPreprocessedRecord = preprocess_module.FullPreprocessedRecord

# Load tokenize_jsonl
spec3 = importlib.util.spec_from_file_location("tokenize_jsonl", Path(__file__).parent / "tokenize_jsonl.py")
tokenize_module = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(tokenize_module)
ParallelJSONLTokenizer = tokenize_module.ParallelJSONLTokenizer
SimplifiedTokenizedRecord = tokenize_module.SimplifiedTokenizedRecord

# Load build_vocabulary
spec4 = importlib.util.spec_from_file_location("build_vocabulary", Path(__file__).parent / "build_vocabulary.py")
vocab_module = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(vocab_module)
VocabularyBuilder = vocab_module.VocabularyBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndPipeline:
    """Complete PDF to training data pipeline"""
    
    def __init__(
        self,
        input_dir: str = "./pdfs",
        output_dir: str = "./output",
        tokenizer_method: str = "huggingface",
        tokenizer_model: str = "gpt2",
        workers: int = 10,
        case_mode: str = "lower",
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        device: str = "gpu",
    ):
        """
        Initialize pipeline
        
        Args:
            input_dir: Directory containing PDFs
            output_dir: Output directory for final results
            tokenizer_method: Tokenization method (huggingface/custom/sentencepiece)
            tokenizer_model: Tokenizer model name
            workers: Number of parallel workers
            case_mode: Case normalization (lower/upper/title/sentence)
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            device: Processing device (gpu/cpu, default: gpu)
        """
        # Initialize logger FIRST (before device configuration)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tokenizer_method = tokenizer_method
        self.tokenizer_model = tokenizer_model
        self.workers = workers
        self.case_mode = case_mode
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure device (GPU/CPU) - NOW logger is available
        self.device = None
        self._configure_device(device)
    
    def _configure_device(self, device: str):
        """
        Configure and validate GPU/CPU device
        
        Args:
            device: Device preference (gpu/cpu)
        """
        device = device.lower()
        
        if device == "gpu":
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"  Available GPUs: {gpu_count}")
                self.logger.info(f"  CUDA Version: {torch.version.cuda}")
                self.device = "cuda"
            else:
                self.logger.warning("GPU requested but not available. Falling back to CPU.")
                self.device = "cpu"
        elif device == "cpu":
            self.logger.info("Using CPU for processing")
            self.device = "cpu"
        else:
            self.logger.warning(f"Unknown device '{device}'. Using GPU if available, else CPU.")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def step1_extract_pdfs(self) -> List[PDFRecord]:
        """Step 1: Extract text from PDFs in-memory"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 1: PDF EXTRACTION")
        self.logger.info("="*70)
        self.logger.info(f"Processing Device: {self.device.upper()}")
        self.logger.info("="*70)
        
        processor = PDFBatchProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            max_workers=self.workers,
        )
        
        result = processor.process()
        records = result.get('records', [])
        
        self.logger.info(f"\n✓ Extracted {len(records)} PDF records")
        return records
    
    def step2_preprocess_text(self, records: List[PDFRecord]) -> List[FullPreprocessedRecord]:
        """Step 2: Preprocess text in-memory"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 2: TEXT PREPROCESSING")
        self.logger.info("="*70)
        
        # Create a dummy temp file just to initialize the preprocessor
        # We'll manually preprocess records using its methods
        temp_input = self.output_dir / "_temp_input.jsonl"
        with open(temp_input, 'w') as f:
            f.write("{}\n")
        
        try:
            processor = ComprehensiveJSONLPreprocessor(
                input_file=str(temp_input),
                output_file=None,
                case_mode=self.case_mode,
                remove_stopwords=self.remove_stopwords,
                remove_punctuation=self.remove_punctuation,
            )
        finally:
            temp_input.unlink()
        
        preprocessed_records = []
        
        # Convert PDFRecord to dict format expected by preprocess_record
        for record in records:
            record_dict = {
                'filename': record.filename,
                'text': record.text,
                'word_count': record.word_count,
            }
            
            # Preprocess the record
            preprocessed = processor.preprocess_record(record_dict)
            preprocessed_records.append(preprocessed)
        
        self.logger.info(f"✓ Preprocessed {len(preprocessed_records)} records")
        
        # Save preprocessed output
        output_file = self.output_dir / "full_preprocessed.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in preprocessed_records:
                json.dump(asdict(record), f, ensure_ascii=False)
                f.write('\n')
        
        self.logger.info(f"✓ Saved: {output_file.name}")
        
        return preprocessed_records
    
    def step3_tokenize(self, preprocessed_records: List[FullPreprocessedRecord]) -> Dict[str, Any]:
        """Step 3: Tokenize preprocessed text"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 3: TOKENIZATION")
        self.logger.info("="*70)
        
        # Save preprocessed records to temporary JSONL for tokenizer to consume
        temp_file = self.output_dir / "_temp_preprocessed.jsonl"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for record in preprocessed_records:
                json.dump(asdict(record), f, ensure_ascii=False)
                f.write('\n')
        
        # Initialize tokenizer
        tokens_dir = self.output_dir / "tokens"
        tokenizer = ParallelJSONLTokenizer(
            input_file=str(temp_file),
            output_dir=str(tokens_dir),
            method=self.tokenizer_method,
            model_name=self.tokenizer_model,
            workers=self.workers,
        )
        
        # Tokenize
        result = tokenizer.process()
        
        # Clean up temporary file
        temp_file.unlink()
        
        return result
    
    def step4_build_vocabulary(self, tokenization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Build vocabulary from tokenized data"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 4: VOCABULARY BUILDING")
        self.logger.info("="*70)
        
        try:
            # Get path to merged tokens
            tokens_dir = self.output_dir / "tokens"
            merged_tokens_file = tokens_dir / "merged_tokens.jsonl"
            
            if not merged_tokens_file.exists():
                self.logger.warning(f"Merged tokens file not found: {merged_tokens_file}")
                return {'status': 'failure', 'message': 'Merged tokens file not found'}
            
            # Build vocabulary
            builder = VocabularyBuilder(str(merged_tokens_file), str(tokens_dir))
            builder.build()
            builder.save()
            builder.print_summary()
            
            # Prepare vocabulary summary
            vocab_info = {
                'total_unique_tokens': len(builder.id_to_token),
                'total_token_instances': sum(builder.token_frequency.values()),
                'token_id_range': [
                    int(min(builder.token_frequency.keys())),
                    int(max(builder.token_frequency.keys()))
                ] if builder.token_frequency else [0, 0],
                'gpt2_vocab_enabled': bool(builder.gpt2_vocab),
                'top_tokens': [
                    {
                        'token_id': token_id,
                        'frequency': freq,
                        'label': builder.id_to_token[token_id]
                    }
                    for token_id, freq in builder.token_frequency.most_common(10)
                ]
            }
            
            return {
                'status': 'success',
                'vocabulary_info': vocab_info,
                'vocab_file': str(tokens_dir / 'vocab.json'),
                'token_counts_file': str(tokens_dir / 'token_counts.json'),
                'vocab_info_file': str(tokens_dir / 'vocab_info.json'),
            }
            
        except Exception as e:
            self.logger.error(f"Error building vocabulary: {str(e)}")
            return {'status': 'failure', 'message': str(e)}
    
    def run(self) -> Dict[str, Any]:
        """Execute full pipeline"""
        start_time = time.time()
        
        self.logger.info("\n")
        self.logger.info("╔" + "="*68 + "╗")
        self.logger.info("║" + " "*15 + "END-TO-END PDF → TOKENS PIPELINE" + " "*21 + "║")
        self.logger.info("╚" + "="*68 + "╝")
        
        try:
            # Step 1: Extract PDFs
            pdf_records = self.step1_extract_pdfs()
            
            if not pdf_records:
                self.logger.error("No PDF records extracted. Exiting.")
                return {'status': 'failure', 'message': 'No PDFs extracted'}
            
            # Step 2: Preprocess
            preprocessed_records = self.step2_preprocess_text(pdf_records)
            
            if not preprocessed_records:
                self.logger.error("No records preprocessed. Exiting.")
                return {'status': 'failure', 'message': 'Preprocessing failed'}
            
            # Step 3: Tokenize
            tokenization_result = self.step3_tokenize(preprocessed_records)
            
            if tokenization_result.get('status') != 'success':
                self.logger.error("Tokenization failed. Exiting.")
                return {'status': 'failure', 'message': 'Tokenization failed'}
            
            # Step 4: Build Vocabulary
            vocabulary_result = self.step4_build_vocabulary(tokenization_result)
            
            total_time = time.time() - start_time
            
            # Final summary
            self.logger.info("\n" + "="*70)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"\nFinal Outputs:")
            self.logger.info(f"  1. {self.output_dir}/full_preprocessed.jsonl (cleaned text)")
            self.logger.info(f"  2. {self.output_dir}/tokens/merged_tokens.jsonl (training tokens)")
            self.logger.info(f"  3. {self.output_dir}/tokens/token_stats.json (token & text summary)")
            self.logger.info(f"  4. {self.output_dir}/tokens/vocab.json (vocabulary mapping)")
            self.logger.info(f"  5. {self.output_dir}/tokens/token_counts.json (token frequencies)")
            self.logger.info(f"  6. {self.output_dir}/tokens/vocab_info.json (vocabulary metadata)")
            
            self.logger.info(f"\nToken Summary Statistics:")
            if tokenization_result.get('status') == 'success':
                self.logger.info(f"  Total tokens: {tokenization_result.get('total_tokens', 0):,}")
                self.logger.info(f"  Total records: {tokenization_result.get('total_records', 0)}")
                self.logger.info(f"  Average tokens per file: {tokenization_result.get('average_tokens_per_record', 0):.1f}")
            
            self.logger.info(f"\nVocabulary Statistics:")
            if vocabulary_result.get('status') == 'success':
                vocab_info = vocabulary_result.get('vocabulary_info', {})
                self.logger.info(f"  Total unique tokens: {vocab_info.get('total_unique_tokens', 0):,}")
                self.logger.info(f"  Total token instances: {vocab_info.get('total_token_instances', 0):,}")
                token_range = vocab_info.get('token_id_range', [0, 0])
                self.logger.info(f"  Token ID range: {token_range[0]} - {token_range[1]}")
                self.logger.info(f"  GPT2 vocabulary enabled: {vocab_info.get('gpt2_vocab_enabled', False)}")
            
            self.logger.info(f"\nTotal Time: {total_time:.2f}s")
            self.logger.info(f"="*70 + "\n")
            
            return {
                'status': 'success',
                'total_time': total_time,
                'pdf_extraction': {
                    'records': len(pdf_records),
                },
                'preprocessing': {
                    'records': len(preprocessed_records),
                },
                'tokenization': tokenization_result,
                'vocabulary': vocabulary_result,
                'output_dir': str(self.output_dir),
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'status': 'failure', 'error': str(e)}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Complete PDF to training tokens pipeline'
    )
    parser.add_argument(
        '--input-dir',
        default='./pdfs',
        help='Input directory containing PDFs (default: ./pdfs)'
    )
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory for final results (default: ./output)'
    )
    parser.add_argument(
        '--tokenizer-method',
        default='huggingface',
        choices=['huggingface', 'custom', 'sentencepiece'],
        help='Tokenization method (default: huggingface)'
    )
    parser.add_argument(
        '--tokenizer-model',
        default='gpt2',
        help='Tokenizer model name (default: gpt2)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--case-mode',
        default='lower',
        choices=['lower', 'upper', 'title', 'sentence'],
        help='Case normalization mode (default: lower)'
    )
    parser.add_argument(
        '--remove-stopwords',
        action='store_true',
        help='Remove common stopwords during preprocessing'
    )
    parser.add_argument(
        '--remove-punctuation',
        action='store_true',
        help='Remove punctuation during preprocessing'
    )
    parser.add_argument(
        '--device',
        default='gpu',
        choices=['gpu', 'cpu'],
        help='Processing device - gpu or cpu (default: gpu)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = EndToEndPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_method=args.tokenizer_method,
        tokenizer_model=args.tokenizer_model,
        workers=args.workers,
        case_mode=args.case_mode,
        remove_stopwords=args.remove_stopwords,
        remove_punctuation=args.remove_punctuation,
        device=args.device,
    )
    
    result = pipeline.run()
    
    # Exit with appropriate code
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
