#!/usr/bin/env python3
"""
Script to count unique BabelNet IDs in the first column of Chinese_expandnet.senses.txt
"""

def count_unique_babelnet_ids(filename):
    unique_ids = set()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Split the line by tab
            parts = line.strip().split('\t')

            # Check if there is at least 1 column
            if len(parts) >= 1:
                # The first column (index 0) contains the BabelNet ID
                babelnet_id = parts[0]
                unique_ids.add(babelnet_id)

    return unique_ids

if __name__ == "__main__":
    filename = "ExpandNet/Chinese_expandnet.senses.txt"

    unique_ids = count_unique_babelnet_ids(filename)

    print(f"Number of unique BabelNet IDs in the first column: {len(unique_ids)}")
    print(f"\nTotal unique IDs: {len(unique_ids)}")
