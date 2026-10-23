# Quick terminal test for the records RAG (no ESP32 needed).
#   python test_rag.py            -> retrieval only (no API key needed)
#   python test_rag.py --llm      -> full RAG + Groq answers
#   python test_rag.py --chat     -> interactive chat
import asyncio
import sys

from rag_engine import build_context

QUESTIONS = [
    "What medicines do I take in the morning?",
    "Am I allergic to anything?",
    "When is my next appointment?",
    "How was my sugar report last time?",
    "Can I take ibuprofen for my knee pain?",
    "Who is my emergency contact?",
    "What surgery did I have before?",
]


async def main():
    use_llm = "--llm" in sys.argv or "--chat" in sys.argv
    if use_llm:
        from llm_agent import load_llm, generate_response
        load_llm()

    if "--chat" in sys.argv:
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit", ""):
                break
            print("Bot:", await generate_response(q))
        return

    for q in QUESTIONS:
        print("\n" + "=" * 70 + f"\nQ: {q}")
        if use_llm:
            print("A:", await generate_response(q))
        else:
            print(build_context("patient_001", q, k=3))


asyncio.run(main())