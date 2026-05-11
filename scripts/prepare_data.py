"""
Data Preparation Script for LARFT

This script converts a JSONL dataset into the Parquet format required by
the verl training framework.

Expected input JSONL format (one JSON object per line):
{
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Please write an article about AI in approximately 500 words."}
    ],
    "reward_model": {
        "ground_truth": 500,       # target word count
        "style": "chat"
    },
    "data_source": "length_control"
}

Output: Parquet files for train/test splits.

Usage:
    python scripts/prepare_data.py \
        --input_file data/raw/your_data.jsonl \
        --output_dir data/ \
        --test_ratio 0.1 \
        --seed 42
"""

import argparse
import json
import os
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_jsonl(file_path: str) -> list:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    print(f"Loaded {len(data)} samples from {file_path}")
    return data


def validate_sample(sample: dict, idx: int) -> bool:
    """Validate that a sample has the required fields."""
    required_keys = ["prompt", "reward_model", "data_source"]
    for key in required_keys:
        if key not in sample:
            print(f"Warning: Sample {idx} missing key '{key}', skipping.")
            return False

    if "ground_truth" not in sample.get("reward_model", {}):
        print(f"Warning: Sample {idx} missing 'reward_model.ground_truth', skipping.")
        return False

    if not isinstance(sample["prompt"], list):
        print(f"Warning: Sample {idx} 'prompt' should be a list of messages, skipping.")
        return False

    return True


def create_parquet(samples: list, output_path: str):
    """Convert samples to a Parquet file."""
    records = []
    for sample in samples:
        records.append({
            "prompt": json.dumps(sample["prompt"], ensure_ascii=False),
            "reward_model": json.dumps(sample["reward_model"], ensure_ascii=False),
            "data_source": sample["data_source"],
        })

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    print(f"Saved {len(records)} samples to {output_path}")


def create_sample_data(output_dir: str, num_train: int = 1000, num_test: int = 100):
    """
    Generate sample training data for demonstration purposes.

    This creates synthetic length-control prompts with random target word counts.
    """
    print("Generating sample data for demonstration...")

    templates = [
        "Please write a response about the topic of '{topic}' in approximately {count} words.",
        "Write exactly {count} words about '{topic}'.",
        "Compose a {count}-word essay on the subject of '{topic}'.",
        "In about {count} words, discuss '{topic}'.",
        "Generate a text of approximately {count} words about '{topic}'.",
    ]

    topics = [
        "artificial intelligence", "climate change", "space exploration",
        "renewable energy", "machine learning", "quantum computing",
        "education reform", "healthcare innovation", "digital privacy",
        "sustainable development", "ocean conservation", "urban planning",
        "food security", "mental health", "cybersecurity",
    ]

    def make_sample(seed_val):
        rng = random.Random(seed_val)
        topic = rng.choice(topics)
        count = rng.choice([50, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000])
        template = rng.choice(templates)
        user_msg = template.format(topic=topic, count=count)

        return {
            "prompt": [
                {"role": "system", "content": "You are a helpful writing assistant. Follow the user's length requirement precisely."},
                {"role": "user", "content": user_msg},
            ],
            "reward_model": {
                "ground_truth": count,
                "style": "chat",
            },
            "data_source": "length_control",
        }

    train_samples = [make_sample(i) for i in range(num_train)]
    test_samples = [make_sample(num_train + i) for i in range(num_test)]

    os.makedirs(output_dir, exist_ok=True)
    create_parquet(train_samples, os.path.join(output_dir, "train.parquet"))
    create_parquet(test_samples, os.path.join(output_dir, "test.parquet"))

    # Also save as JSONL for reference
    jsonl_path = os.path.join(output_dir, "sample_data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in train_samples[:10]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved 10 sample records to {jsonl_path} for reference.")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LARFT")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to input JSONL file. If not provided, generates sample data.")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory for Parquet files.")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of data to use for testing (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split.")
    parser.add_argument("--generate_sample", action="store_true",
                        help="Generate sample data for demonstration.")
    parser.add_argument("--num_train", type=int, default=1000,
                        help="Number of training samples to generate (with --generate_sample).")
    parser.add_argument("--num_test", type=int, default=100,
                        help="Number of test samples to generate (with --generate_sample).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.generate_sample or args.input_file is None:
        create_sample_data(args.output_dir, args.num_train, args.num_test)
        return

    # Load and validate data
    raw_data = load_jsonl(args.input_file)
    valid_data = [s for i, s in enumerate(raw_data) if validate_sample(s, i)]
    print(f"Validated {len(valid_data)}/{len(raw_data)} samples.")

    if len(valid_data) == 0:
        print("Error: No valid samples found. Exiting.")
        return

    # Split into train/test
    random.seed(args.seed)
    random.shuffle(valid_data)
    split_idx = max(1, int(len(valid_data) * (1 - args.test_ratio)))
    train_data = valid_data[:split_idx]
    test_data = valid_data[split_idx:]

    print(f"Split: {len(train_data)} train, {len(test_data)} test")

    # Save as Parquet
    create_parquet(train_data, os.path.join(args.output_dir, "train.parquet"))
    create_parquet(test_data, os.path.join(args.output_dir, "test.parquet"))

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
