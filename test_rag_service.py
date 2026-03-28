import time
from app.services.rag_service import answer_question

questions = [
    "What are common symptoms of parasites in dogs?",
    "When should I take my dog to the vet for vomiting?",
    "What are signs of dehydration in dogs?"
]

for i, q in enumerate(questions, 1):
    print(f"\n[{i}] start")
    t0 = time.time()
    ans = answer_question(q)
    t1 = time.time()
    print(f"[{i}] elapsed: {t1 - t0:.2f}s")
    print(ans[:300], "...")