#!/usr/bin/env python3
"""
Count TRUE and FALSE values in the "Link in Dictionary?" column of Annotations_Chinese.tsv
"""

import csv
from collections import Counter

# File path
tsv_file = "Annotations_Chinese.tsv"

# Initialize counter
link_counter = Counter()

# Read the TSV file
with open(tsv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')

    for row in reader:
        link_value = row.get('Link in Dictionary?', '')
        if link_value is not None:
            link_value = link_value.strip()
        else:
            link_value = ''
        if link_value:  # Only count non-empty values
            link_counter[link_value] += 1

# Print results
print("Count of values in 'Link in Dictionary?' column:")
print("=" * 50)
for value, count in sorted(link_counter.items()):
    print(f"{value}: {count}")

print("=" * 50)
print(f"Total entries counted: {sum(link_counter.values())}")

# Print TRUE and FALSE specifically
if 'True' in link_counter:
    print(f"\nTRUE count: {link_counter['True']}")
if 'False' in link_counter:
    print(f"FALSE count: {link_counter['False']}")
