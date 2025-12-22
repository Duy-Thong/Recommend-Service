"""
Test Word2Vec recommender without actual model.
"""

import sys
sys.path.insert(0, '.')

from recommend_service.services.evaluation.baseline_methods import Word2VecRecommender

# Test 1: Check if class can be instantiated
print("Test 1: Creating Word2VecRecommender instance...")
try:
    recommender = Word2VecRecommender(model_path="dummy_path.bin")
    print("[OK] Instance created successfully")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
    sys.exit(1)

# Test 2: Check tokenization
print("\nTest 2: Testing tokenization...")
try:
    text = "Nhan Vien Kinh Doanh - Sales Executive"
    tokens = recommender._tokenize(text)
    print(f"Input: {text}")
    print(f"Tokens: {tokens}")
    print(f"[OK] Tokenization works")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
    sys.exit(1)

# Test 3: Check if gensim import works
print("\nTest 3: Checking gensim import...")
try:
    from gensim.models import KeyedVectors
    print("[OK] gensim is installed")
except ImportError:
    print("[FAIL] gensim not installed - need to install: pip install gensim")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
print("\nWord2VecRecommender is ready to use.")
print("Just need to provide a real Word2Vec model file.")
