# Quick terminal test for the records RAG (no ESP32 needed).
#   python test_rag.py            -> retrieval only (no API key needed)
#   python test_rag.py --llm      -> full RAG + LLM answers
#   python test_rag.py --chat     -> interactive chat
import asyncio
import sys

try:
    from rag_engine import build_context
except ImportError:
    from backend.rag_engine import build_context

QUESTIONS = [
    "What medicines do I take in the morning?",
    "Am I allergic to anything?",
    "When is my next appointment?",
    "How was my sugar report last time?",
    "Can I take ibuprofen for my knee pain?",
    "Who is my emergency contact?",
    "What surgery did I have before?",
]


async def main_async():
    use_chat = "--chat" in sys.argv
    from llm_agent import load_llm, generate_response
    load_llm()

    if use_chat:
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", ""):
                break
            print("Bot:", await generate_response(q))
        return

    for q in QUESTIONS:
        print("\n" + "=" * 70 + f"\nQ: {q}")
        print("A:", await generate_response(q))


def main_sync():
    for q in QUESTIONS:
        print("\n" + "=" * 70 + f"\nQ: {q}")
        print(build_context("patient_001", q, k=3))


if __name__ == "__main__":
    if "--llm" in sys.argv or "--chat" in sys.argv:
        asyncio.run(main_async())
    else:
        main_sync()
