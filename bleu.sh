#!/bin/bash
for filename in predictions/*.txt; do
  python bleu.py $filename
  sleep 5
done