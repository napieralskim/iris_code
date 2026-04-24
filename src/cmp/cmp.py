#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from   daug import daug_strip_mask
import itertools
import numpy as np
from   numpy.typing import NDArray
import matplotlib.pyplot as plt
import re


def parse_filename(filename: str) -> str | None:
    match = re.match(r"([a-z]+)([lr])(\d+)", filename)
    if match:
        name, side, _ = match.groups()
        return f"{name}_{side}"
    return None

def load_code(path) -> NDArray[np.bool]:
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        matrix = [[char == '1' for char in line.strip()] for line in lines]
    return np.array(matrix, dtype=np.bool)

# mask: 1 <=> probably the iris, 0 <=> probably the eyelid
def hamming_distance(code_a: NDArray[np.bool], code_b: NDArray[np.bool], mask: NDArray[np.bool]):
    diffs = (code_a ^ code_b) & mask
    return np.sum(diffs) / np.sum(mask)

def hamming_distance_with_rotation(
        code_a: NDArray[np.bool],
        code_b: NDArray[np.bool],
        mask: NDArray[np.bool],
        max_shift=5):
    dist_min = 1.0
    for shift in range(-max_shift, max_shift + 1):
        code_b_shifted = np.roll(code_b, shift, axis=1)
        mask_shifted = np.roll(mask, shift, axis=1)
        mask_combo = mask & mask_shifted
        mask_combo_count_1 = np.sum(mask_combo)
        if mask_combo_count_1 == 0:
            continue
        dist = hamming_distance(code_a, code_b_shifted, mask_combo)
        dist_min = min(dist_min, dist)
    return dist_min

def run_analysis(output_dir="output"):
    filenames = [f for f in os.listdir(output_dir) if not f.startswith('.')]
    codes = {}
    
    for filename in filenames:
        label = parse_filename(filename)
        if label:
            codes[filename] = {
                'label': label,
                'data': load_code(os.path.join(output_dir, filename))
            }

    # TODO this assumes all codes have the same format
    # (including same width and the fact that "imaginary bits" follow "the real ones")
    iris_width = codes[filenames[0]]['data'].shape[1] // 2
    mask = np.stack([daug_strip_mask(i, iris_width) for i in range(8)])
    mask = np.hstack((mask, mask))

    intra_dists = []
    inter_dists = []

    for f1, f2 in itertools.combinations(codes.keys(), 2):
        code_a = codes[f1]['data']
        code_b = codes[f2]['data']
        dist = hamming_distance(code_a, code_b, mask) # TODO configurable with rotation
        
        if codes[f1]['label'] == codes[f2]['label']:
            intra_dists.append(dist)
        else:
            inter_dists.append(dist)
            
    return intra_dists, inter_dists

intra, inter = run_analysis()

plt.figure(figsize=(10, 6))
plt.hist(intra, bins=30, alpha=0.6, label='To samo oko (Intra-class)', color='green', density=True)
plt.hist(inter, bins=30, alpha=0.6, label='Różne oczy (Inter-class)', color='red', density=True)

# Teoretyczny próg Daugmana
plt.axvline(x=0.32, color='black', linestyle='--', label='Typowy próg (0.32)')

plt.title("Rozkład odległości Hamminga")
plt.xlabel("Hamming Distance")
plt.ylabel("Gęstość prawdopodobieństwa")
plt.legend()
plt.show()