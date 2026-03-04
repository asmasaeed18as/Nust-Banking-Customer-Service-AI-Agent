import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

chunks = json.load(open('data/processed/bank_knowledge_chunks.json'))
meta   = json.load(open('data/vector_store/bank_metadata.json'))
index  = faiss.read_index('data/vector_store/bank_faiss_index.bin')

print('=== EMBEDDING SANITY CHECK ===')
model = SentenceTransformer('all-MiniLM-L6-v2')
tests = [
    'How do I transfer money?',
    'What is the daily transfer limit?',
    'Reset my MPIN password',
]
embs = model.encode(tests, normalize_embeddings=True, convert_to_numpy=True)
print('Shape:', embs.shape)
for e in embs:
    print('  norm:', round(float(np.linalg.norm(e)), 6))
print('Sim Q1-Q2 (transfer pair):', round(float(np.dot(embs[0], embs[1])), 4), ' HIGH expected')
print('Sim Q1-Q3 (diff topic):   ', round(float(np.dot(embs[0], embs[2])), 4), ' LOWER expected')

print()
print('=== RETRIEVAL TESTS ===')
queries = [
    'What is the daily funds transfer limit?',
    'How to reset mobile banking password?',
    'Can I do international transactions?',
    'What are home remittance services?',
    'Is biometric login supported?',
]
for q in queries:
    qe = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.search(qe, k=3)
    print('Query:', q)
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        chunk = meta[idx]
        score = round(1 - dist/2, 3)
        src   = chunk['source']
        question = chunk['question'][:70]
        answer   = chunk['answer'][:90]
        print(f'  #{rank+1} score={score} src={src}')
        print(f'       Q: {question}')
        print(f'       A: {answer}')
    print()
