#!/usr/bin/env python3
"""
Convert expandnet_step3_project.out.tsv to Chinese_expandnet.senses.txt format.
The output format is: synset_id<TAB>translation
"""

import argparse
import pandas as pd


def convert_tsv_to_senses(input_tsv, output_txt):
    """
    Convert TSV file to senses.txt format.

    Args:
        input_tsv: Path to input TSV file (expandnet_step3_project.out.tsv)
        output_txt: Path to output text file (Chinese_expandnet.senses.txt)
    """
    # Read the TSV file
    # Assuming the file has no header and two columns: synset_id and translation
    df = pd.read_csv(input_tsv, sep='\t', header=None, names=['synset_id', 'translation'])

    # Remove any rows with missing values
    df = df.dropna()

    # Write to output file in the same format
    with open(output_txt, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['synset_id']}\t{row['translation']}\n")

    print(f"Conversion complete!")
    print(f"Input file:  {input_tsv}")
    print(f"Output file: {output_txt}")
    print(f"Total entries: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert expandnet_step3_project.out.tsv to Chinese_expandnet.senses.txt"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='expandnet_step3_project.out.tsv',
        help='Input TSV file (default: expandnet_step3_project.out.tsv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Chinese_expandnet.senses.txt',
        help='Output text file (default: Chinese_expandnet.senses.txt)'
    )

    args = parser.parse_args()

    convert_tsv_to_senses(args.input, args.output)


if __name__ == '__main__':
    main()
