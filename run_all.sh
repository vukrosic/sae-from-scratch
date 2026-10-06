#!/bin/sh
# Runs every step in order and prints what the video shows. About three minutes.
set -e
cd "$(dirname "$0")"
for f in step1_model step2_neurons step3_superposition step4_sparse_autoencoder step5_sae_on_model step6_features step7_steering step8_how_many; do
  echo "== $f"
  python3 $f.py
done
