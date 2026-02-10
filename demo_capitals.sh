#!/bin/bash
echo "GPT-2 Capital City Quiz"
echo "======================="
echo ""

for country in "France" "Italy" "Germany" "Japan" "Spain" "England" "Russia" "China"; do
  echo -n "Q: What is the capital of $country? A: "
  answer=$(go run ask_gpt2.go "The capital city of $country is" 2>&1 | \
           grep "Answer only:" | \
           sed 's/Answer only: //' | \
           awk '{print $1}' | \
           sed 's/\.$//')
  echo "$answer"
done
