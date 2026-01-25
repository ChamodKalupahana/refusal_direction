#!/usr/bin/env python3
"""Plot PC1 character length results from JSON data."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results
with open("pc1_character_length_results_10x.json", "r") as f:
    data = json.load(f)

# Extract data
multipliers = data["multipliers"]
by_category = data["summary"]["by_category"]

# Prepare data for plotting
top_chars = [by_category["top"][str(m)]["avg_char_length"] for m in multipliers]
top_words = [by_category["top"][str(m)]["avg_word_count"] for m in multipliers]
bottom_chars = [by_category["bottom"][str(m)]["avg_char_length"] for m in multipliers]
bottom_words = [by_category["bottom"][str(m)]["avg_word_count"] for m in multipliers]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot character length
ax1.plot(multipliers, top_chars, 'o-', label='TOP prompts (high PC1)', color='#2ecc71', linewidth=2, markersize=8)
ax1.plot(multipliers, bottom_chars, 's-', label='BOTTOM prompts (low PC1)', color='#e74c3c', linewidth=2, markersize=8)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax1.set_xlabel('PC1 Multiplier', fontsize=12)
ax1.set_ylabel('Average Character Length', fontsize=12)
ax1.set_title('Character Length vs PC1 Multiplier', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot word count
ax2.plot(multipliers, top_words, 'o-', label='TOP prompts (high PC1)', color='#2ecc71', linewidth=2, markersize=8)
ax2.plot(multipliers, bottom_words, 's-', label='BOTTOM prompts (low PC1)', color='#e74c3c', linewidth=2, markersize=8)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax2.set_xlabel('PC1 Multiplier', fontsize=12)
ax2.set_ylabel('Average Word Count', fontsize=12)
ax2.set_title('Word Count vs PC1 Multiplier', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pc1_character_length_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved to pc1_character_length_plot.png")
