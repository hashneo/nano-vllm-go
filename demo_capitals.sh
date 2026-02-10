#!/bin/bash
# Demo script showing GPT-2 answering capital city questions

set -e

MODEL_DIR="./models/gpt2-small"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
  echo "Model not found at $MODEL_DIR"
  echo "Downloading GPT-2 Small (124M parameters)..."
  echo ""
  python3 scripts/download_model.py --model gpt2 --output "$MODEL_DIR"
  echo ""
  echo "✓ Model downloaded!"
  echo ""
fi

# Build if needed
if [ ! -f "./bin/ask-gpt2" ]; then
  echo "Building ask-gpt2..."
  make ask-gpt2
  echo ""
fi

echo "GPT-2 Capital City Quiz"
echo "======================="
echo ""

for country in "France" "Italy" "Germany" "Japan" "Spain" "England" "Russia" "China"; do
  echo -n "Q: What is the capital of $country? A: "
  answer=$(./bin/ask-gpt2 "The capital city of $country is" 2>&1 | \
           grep "Answer only:" | \
           sed 's/Answer only: //' | \
           awk '{print $1}' | \
           sed 's/\.$//')
  echo "$answer"
done
