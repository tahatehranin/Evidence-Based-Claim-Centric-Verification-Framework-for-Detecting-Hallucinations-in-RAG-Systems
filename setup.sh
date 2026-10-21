#!/bin/bash

# Evidence-Based Claim-Centric Verification Framework for Hallucination Detection in RAG Systems
# Installation and setup script

set -e

echo "========================================"
echo "Setting up RAG Hallucination Detection"
echo "========================================"

# Create venv
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demo:"
echo "  python -m examples.demo"
echo ""
echo "To use in your code:"
echo "  from verifier import RAGHallucinationVerifier"
echo "  verifier = RAGHallucinationVerifier(context='...', answer='...')"
echo "  result = verifier.verify()"
