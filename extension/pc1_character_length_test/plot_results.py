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
results = data["results"]

# Compute mean and std for each category/multiplier from individual results
def compute_stats(results, category, multipliers):
    """Compute mean and std for char_length and word_count."""
    char_means, char_stds = [], []
    word_means, word_stds = [], []
    
    for m in multipliers:
        m_str = str(float(m))
        chars = []
        words = []
        for r in results:
            if r["category"] == category:
                chars.append(r["multiplier_results"][m_str]["char_length"])
                words.append(r["multiplier_results"][m_str]["word_count"])
        
        char_means.append(np.mean(chars))
        char_stds.append(np.std(chars) / np.sqrt(len(chars)))  # SEM
        word_means.append(np.mean(words))
        word_stds.append(np.std(words) / np.sqrt(len(words)))  # SEM
    
    return char_means, char_stds, word_means, word_stds

# Get stats for top and bottom prompts
top_chars, top_chars_std, top_words, top_words_std = compute_stats(results, "top", multipliers)
bottom_chars, bottom_chars_std, bottom_words, bottom_words_std = compute_stats(results, "bottom", multipliers)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot character length with error bars
ax1.errorbar(multipliers, top_chars, yerr=top_chars_std, fmt='o-', label='TOP prompts (high PC1)', 
             color='#2ecc71', linewidth=2, markersize=8, capsize=4)
ax1.errorbar(multipliers, bottom_chars, yerr=bottom_chars_std, fmt='s-', label='BOTTOM prompts (low PC1)', 
             color='#e74c3c', linewidth=2, markersize=8, capsize=4)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax1.set_xlabel('PC1 Multiplier', fontsize=12)
ax1.set_ylabel('Average Character Length', fontsize=12)
ax1.set_title('Character Length vs PC1 Multiplier', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot word count with error bars
ax2.errorbar(multipliers, top_words, yerr=top_words_std, fmt='o-', label='TOP prompts (high PC1)', 
             color='#2ecc71', linewidth=2, markersize=8, capsize=4)
ax2.errorbar(multipliers, bottom_words, yerr=bottom_words_std, fmt='s-', label='BOTTOM prompts (low PC1)', 
             color='#e74c3c', linewidth=2, markersize=8, capsize=4)
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
