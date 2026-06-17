"""Example: solve a math problem with OnaAI-2.0.

Run:
    python examples/solve_math.py
"""

from onaai import ReasoningEngine


def main() -> None:
    engine = ReasoningEngine.from_default()

    problem = (
        "What is the remainder when 7^100 is divided by 13? "
        "Give the final answer in \\boxed{}."
    )

    result = engine.solve(problem)

    print("Problem:")
    print(" ", problem)
    print("\nReasoning (truncated to 500 chars):")
    print(" ", (result.reasoning or "<none>")[:500])
    print("\nFinal answer:")
    print(" ", result.answer)


if __name__ == "__main__":
    main()
