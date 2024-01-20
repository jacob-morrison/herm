import argparse
from datasets import load_dataset
import random
import os

print("Filtering subset of RM benchmark dataset")

def print_pretty(row):
    print("prompt:", row["prompt"])
    print("- - - - - - - - - - - - -")
    print("chosen:", row["chosen"])
    print("- - - - - - - - - - - - -")
    print("rejected:", row["rejected"])

# Set up argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('subset', type=str, help='Subset to use')
parser.add_argument('start_idx', type=int, nargs='?', default=0, help='Start index for processing the dataset (default: 0)')

# Parse arguments
args = parser.parse_args()

# Define the subsets
subsets = [
    "alpacaeval-easy", # 0
    "alpacaeval-length", # 1
    "alpacaeval-hard", # 2
    "mt-bench-easy", # 3
    "mt-bench-med", # 4
    "mt-bench-hard", # 5
    "refusals-dangerous", # 6
    "refusals-offensive", # 7
    "llmbar-natural", # 8
    "llmbar-adver-neighbor",
    "llmbar-adver-GPTInst",
    "llmbar-adver-GPTOut",
    "llmbar-adver-manual",
    "XSTest"
]
assert args.subset in subsets, f"Subset given {args.subset} not found in list of subsets"

# Get subset and start index from arguments
start_idx = args.start_idx

# Load dataset
dataset = load_dataset("ai2-rlhf-collab/rm-benchmark-dev", split="train")

# Filter dataset based on subset
dataset = dataset.filter(lambda x: x["subset"] == args.subset)

# Seed for reproducibility
random.seed(0)
indices = list(range(len(dataset)))
random.shuffle(indices)
print(indices)

# Loop over the dataset
for idx in indices[start_idx:]:
    os.system('clear')
    row = dataset[idx]
    print(idx, "=====================")
    print_pretty(row)
    input("Press Enter to continue...")