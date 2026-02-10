#!/bin/bash
# Demo script showing GPT-2 answering capital city questions

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
