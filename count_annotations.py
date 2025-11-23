#!/usr/bin/env python3
"""
Script to count manual annotation results from Annotations_Chinese.tsv
"""

import csv


def count_annotations(file_path):
    """
    Count annotation results from TSV file.

    Args:
        file_path: Path to the TSV file

    Returns:
        tuple: (same_meaning_count, translation_equivalents_only_count)
    """
    same_meaning_count = 0
    translation_equivalents_only_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            same_meaning = row.get('same_meaning', '') or ''
            translation_equivalents = row.get('translation_equivalents', '') or ''

            same_meaning = same_meaning.strip()
            translation_equivalents = translation_equivalents.strip()

            # Count rows where same_meaning is 1
            if same_meaning == '1':
                same_meaning_count += 1

            # Count rows where same_meaning is 0 but translation_equivalents is 1
            if same_meaning == '0' and translation_equivalents == '1':
                translation_equivalents_only_count += 1

    return same_meaning_count, translation_equivalents_only_count


if __name__ == '__main__':
    file_path = 'Annotations_Chinese.tsv'

    same_meaning_count, translation_equivalents_only_count = count_annotations(file_path)

    print("Annotation Results:")
    print("=" * 50)
    print(f"Number of 1s in 'same_meaning' column: {same_meaning_count}")
    print(f"Number where 'same_meaning'=0 but 'translation_equivalents'=1: {translation_equivalents_only_count}")
    print("=" * 50)
