import sys
import io
from app import run_agent

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "What is a Platform Event?"
    print(run_agent(question, []))
