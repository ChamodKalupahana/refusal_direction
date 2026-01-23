import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.load_dataset import load_dataset_split

def main():
    print("Loading datasets to check size...")
    harmful_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    
    print(f"Harmful Train Size: {len(harmful_train)}")
    print(f"Harmless Train Size: {len(harmless_train)}")
    print(f"Max N_INST_TRAIN: {min(len(harmful_train), len(harmless_train))}")

if __name__ == "__main__":
    main()
