"""
Test GPTMed Generation API

This script tests the generation function after training.
"""

import sys
from pathlib import Path

# Add gptmed to path to use local version
sys.path.insert(0, str(Path(__file__).parent))

import gptmed

print("="*60)
print("Testing GPTMed Generation")
print("="*60)

# Use the trained model
checkpoint = "model/checkpoints/best_model.pt"
tokenizer = "../medllm/tokenizer/medquad_tokenizer.model"
prompts = [
    "What is diabetes?",
    "What causes high blood pressure?",
    "How to treat fever?"
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {prompt}")
    print('='*60)
    
    try:
        answer = gptmed.generate(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=150,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            device="cuda"
        )
        
        print(f"\n📝 Generated:")
        print(answer)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("✅ Generation test complete!")
print('='*60)
