"""Example demonstrating the hallucination detection framework."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from verifier import RAGHallucinationVerifier


def example_1_simple():
    """Simple example with clean claims."""
    context = """
    Python 3.12 introduced new features like structural pattern matching and improved 
    type hints. The release date was October 2, 2023. It became the latest stable 
    version as of late 2023.
    """
    
    answer = """
    Python 3.12 uses structural pattern matching and improved type hints. 
    It was released on October 2, 2023.
    """
    
    verifier = RAGHallucinationVerifier(
        context=context,
        answer=answer
    )
    result = verifier.verify()
    print(f"Answer label: {result.answer_label}")
    for claim in result.claim_results:
        print(f"  [{claim['final_label']}] {claim['claim_text']}")


def example_2_hallucinated():
    """Example with hallucinated claims."""
    context = """
    Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. 
    Elon Musk joined in 2004 and became CEO in 2008.
    """
    
    answer = """
    Tesla was founded by Elon Musk in 2004, and he remains the CEO until today.
    """
    
    verifier = RAGHallucinationVerifier(
        context=context,
        answer=answer
    )
    result = verifier.verify()
    print(f"Answer label: {result.answer_label}")
    for claim in result.claim_results:
        print(f"  [{claim['final_label']}] {claim['claim_text']}")


def example_3_contradiction():
    """Example with contradictory claims."""
    context = """
    The United States has 50 states. The capital city is Washington, D.C. 
    It is located on the East Coast.
    """
    
    answer = """
    The US has 48 states and the capital is on the West Coast.
    """
    
    verifier = RAGHallucinationVerifier(
        context=context,
        answer=answer
    )
    result = verifier.verify()
    print(f"Answer label: {result.answer_label}")
    for claim in result.claim_results:
        print(f"  [{claim['final_label']}] {claim['claim_text']}")


if __name__ == "__main__":
    print("Example 1: Simple Entailment")
    print("-" * 70)
    example_1_simple()
    
    print("\n\nExample 2: Hallucinated Claim")
    print("-" * 70)
    example_2_hallucinated()
    
    print("\n\nExample 3: Contradictory Claim")
    print("-" * 70)
    example_3_contradiction()
