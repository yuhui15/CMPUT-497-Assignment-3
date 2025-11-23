#!/usr/bin/env python3
"""
Script to compare Annotations_Chinese_old.tsv and Annotations_Chinese.tsv
to identify which instance entries were removed.
"""

def parse_instances(filename, has_header=False):
    """Parse instance entries from the annotation file."""
    instances = {}

    with open(filename, 'r', encoding='utf-8') as f:
        if has_header:
            next(f)  # Skip header line

        for line_num, line in enumerate(f, start=2 if has_header else 1):
            parts = line.strip().split('\t')

            if len(parts) < 2:
                continue

            # Check if this is an instance entry
            # Old format: column 2 is "instance"
            # New format: column 1 starts with "d" (like d000.s000.s000)
            is_instance_old = len(parts) >= 6 and parts[1] == "instance"
            is_instance_new = len(parts) >= 1 and parts[0].startswith('d') and '.' in parts[0]

            if is_instance_old:
                # Old format: sentence_id, type, lemma, pos, raw_text, instance_id, target_token, ...
                sentence_id = parts[0]
                instance_id = parts[5]
                lemma = parts[2]
                target_token = parts[6] if len(parts) > 6 else ""

                instances[instance_id] = {
                    'sentence_id': sentence_id,
                    'lemma': lemma,
                    'target_token': target_token,
                    'line_num': line_num,
                    'full_line': line.strip()
                }
            elif is_instance_new:
                # New format: Token ID, Source Token, Source Lemma, Source POS, Translated Token, Translated Lemma, Synset ID, ...
                token_id = parts[0]
                source_token = parts[1] if len(parts) > 1 else ""
                translated_token = parts[4] if len(parts) > 4 else ""
                synset_id = parts[6] if len(parts) > 6 else ""

                instances[token_id] = {
                    'token_id': token_id,
                    'source_token': source_token,
                    'translated_token': translated_token,
                    'synset_id': synset_id,
                    'line_num': line_num,
                    'full_line': line.strip()
                }

    return instances

def main():
    print("Parsing Annotations_Chinese_old.tsv...")
    old_instances = parse_instances("Annotations_Chinese_old.tsv", has_header=False)

    print("Parsing Annotations_Chinese.tsv...")
    new_instances = parse_instances("Annotations_Chinese.tsv", has_header=True)

    print(f"\nTotal instances in old file: {len(old_instances)}")
    print(f"Total instances in new file: {len(new_instances)}")
    print(f"Difference: {len(old_instances) - len(new_instances)}")

    # Find instances in old but not in new
    old_ids = set(old_instances.keys())
    new_ids = set(new_instances.keys())

    # The IDs are different formats, so let's analyze the instances differently
    # Let's look at instances that might be missing based on content

    print("\n" + "="*80)
    print("INSTANCES IN OLD FILE:")
    print("="*80)

    # Group by whether they have target_token
    with_translation = []
    without_translation = []

    for instance_id, data in old_instances.items():
        if data['target_token'].strip():
            with_translation.append(data)
        else:
            without_translation.append(data)

    print(f"\nInstances WITH Chinese translation: {len(with_translation)}")
    print(f"Instances WITHOUT Chinese translation: {len(without_translation)}")

    print("\n" + "="*80)
    print("SAMPLE INSTANCES WITHOUT TRANSLATION (first 20):")
    print("="*80)
    for i, data in enumerate(without_translation[:20], 1):
        print(f"{i}. Line {data['line_num']}: {data['sentence_id']} - {data['lemma']}")

    print("\n" + "="*80)
    print("INSTANCES IN NEW FILE:")
    print("="*80)

    # Check new file instances
    with_synset = []
    without_synset = []

    for token_id, data in new_instances.items():
        if data['synset_id'].strip():
            with_synset.append(data)
        else:
            without_synset.append(data)

    print(f"\nInstances WITH Synset ID: {len(with_synset)}")
    print(f"Instances WITHOUT Synset ID: {len(without_synset)}")

    # Check lemma distribution in old file
    print("\n" + "="*80)
    print("ANALYSIS OF MISSING INSTANCES:")
    print("="*80)

    # Count instances without translation that might have been removed
    print(f"\nLikely removed instances (no translation in old file): {len(without_translation)}")
    print(f"Actual difference in count: {len(old_instances) - len(new_instances)}")

    if len(without_translation) > 0:
        print("\nAll instances without translation in old file:")
        for i, data in enumerate(without_translation, 1):
            print(f"{i}. Line {data['line_num']}: {data['sentence_id']} | "
                  f"Lemma: {data['lemma']} | Instance ID: {data.get('full_line', '').split()[5] if len(data.get('full_line', '').split()) > 5 else 'N/A'}")

if __name__ == "__main__":
    main()
