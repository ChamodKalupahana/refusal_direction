import os
import json
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.load_dataset import load_dataset_split

OUTPUT_PATH = os.path.join(REPO_ROOT, "dataset", "splits", "harmful_shortest.json")
MAX_WORDS = 18
NUM_PROMPTS = 100


def load_harmful_examples():
    examples = []
    for split in ("train", "val", "test"):
        examples.extend(load_dataset_split(harmtype="harmful", split=split, instructions_only=False))
    return examples


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    examples = load_harmful_examples()
    lengths = [(word_count(example["instruction"]), idx, example) for idx, example in enumerate(examples)]
    lengths = [item for item in lengths if item[0] < MAX_WORDS]
    lengths.sort(key=lambda x: (x[0], x[1]))
    shortest = [example for _, _, example in lengths[:NUM_PROMPTS]]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(shortest, f, indent=4)

    print(f"Saved {len(shortest)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
