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

# # Load dataset
# dataset = load_dataset("ai2-rlhf-collab/rm-benchmark-dev", split="train")

# # Filter dataset based on subset
# dataset = dataset.filter(lambda x: x["subset"] == args.subset)

if args.subset == 'refusals-dangerous':
    file_path = 'f_out-dolphin-dangerous.jsonl'
    subset = 'dangerous'
elif args.subset == 'refusals-offensive':
    file_path = 'f_out-dolphin-offensive.jsonl'
    subset = 'offensive'
else:
    print('Only doing refusals atm')
    quit()

dataset = []
with open(file_path) as f_in:
    import json
    for line in f_in.readlines():
        data = json.loads(line)
        dataset.append(data)

# Seed for reproducibility
random.seed(0)
indices = list(range(len(dataset)))
random.shuffle(indices)
print(indices)

# Loop over the dataset
with open(f'filtered-refusals-dolphin-{subset}.jsonl', 'a') as f_out:
    i = indices.index(start_idx)
    while i < len(indices):
    # for idx in indices[start_idx:]:
        idx = indices[i]
        row = dataset[idx]
        print(idx, "=====================")
        print_pretty(row)
        inp = input("Press k to keep and f to filter...")
        if inp == 'k':
            row['filtered'] = False
            i += 1
            json.dump(row, f_out)
            f_out.write('\n')
            os.system('clear')
        elif inp == 'f':
            row['filtered'] = True
            i += 1
            json.dump(row, f_out)
            f_out.write('\n')
            os.system('clear')
        else:
            os.system('clear')
            print(f'Invalid input: {inp}')