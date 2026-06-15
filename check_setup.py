"""Setup checker — verifies both LLM providers and Firestore before running.

Usage (venv active):  python check_setup.py

Run this after setting up a new machine. If everything prints OK,
`streamlit run app.py` will work.
"""

import sys

import config
from chatbots import ask_claude, ask_gemini
from storage import Conversation, delete_conversation, save_conversation


def main() -> int:
    print("=" * 60)
    print("DOUBLECHAT SETUP CHECK")
    print("=" * 60)

    # 1. Configuration
    problems = config.validate()
    if problems:
        for problem in problems:
            print(f"  CONFIG ERROR: {problem}")
        return 1
    print(f"  Config OK  (project={config.GCP_PROJECT}, "
          f"location={config.GCP_LOCATION})")
    print()

    exit_code = 0

    # 2. One real call per LLM provider
    for name, ask in [("Gemini", ask_gemini), ("Claude", ask_claude)]:
        result = ask("Reply with the single word: OK")
        if result.ok:
            print(f"  {name:<9} OK   model={result.model}  "
                  f"latency={result.latency_s}s  "
                  f"tokens={result.input_tokens}/{result.output_tokens}")
        else:
            print(f"  {name:<9} FAIL {result.error}")
            exit_code = 1

    # 3. Firestore round-trip (write then delete a throwaway doc)
    try:
        conv = Conversation(
            id=None,
            title="__setup_check__",
            model="n/a",
            provider="n/a",
            messages=[{"role": "user", "content": "ping"}],
        )
        conv_id, write_ms = save_conversation(conv)
        _, delete_ms = delete_conversation(conv_id)
        print(f"  {'Firestore':<9} OK   write={write_ms}ms  delete={delete_ms}ms")
    except Exception as e:  # noqa: BLE001
        print(f"  {'Firestore':<9} FAIL {type(e).__name__}: {e}")
        exit_code = 1

    print()
    print("=" * 60)
    print("All good — run: streamlit run app.py" if exit_code == 0
          else "Fix the failures above, then re-run this script.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
