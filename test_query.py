import os
import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.rag import ask, get_vectordb

def main():
    question = "bảo hiểm y tế là gì"
    print(f"Question: {question}")
    answer, docs = ask(question)
    
    print("\n--- RETRIEVED DOCS ---")
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] Source: {doc.metadata.get('source')} - Page: {doc.metadata.get('page')}")
        print(doc.page_content)
        
    print("\n--- ANSWER ---")
    print(answer)

if __name__ == "__main__":
    main()
