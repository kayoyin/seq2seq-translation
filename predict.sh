#!/bin/bash
for filename in models/*.pt; do
  python predict.py $(basename "$filename" .pt)
  sleep 5
done