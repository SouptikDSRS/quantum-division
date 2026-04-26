from src.llm_optimizer import run_ai_optimizer
from src.division_engine import DivisionEngine


def main():
    engine = DivisionEngine()

    n = 16
    target_accuracy = 0.99

    output = run_ai_optimizer(n, target_accuracy, engine)

    print("\n=== FINAL RESULT ===")
    print(output["result"])

    print("\n=== AI ANALYSIS ===")
    print(output["analysis"])


if __name__ == "__main__":
    main()