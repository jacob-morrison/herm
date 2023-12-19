import json

categories = set()
with open('all_prompts_train_with_gpt-4-1106-preview_responses_explicit_refusal.jsonl') as refusals_in:
    with open('f_out-vllm.jsonl') as zephyr_in:
        with open('filtered_refusals_pairs.jsonl', 'w') as f_out:
            refusals = list(refusals_in.readlines())
            zephyr_responses = list(zephyr_in.readlines())
            for ref_dict, zephyr_dict in zip(refusals, zephyr_responses):
                data = json.loads(ref_dict)
                if data["is_refusal"]:
                    categories.add(data["category"])
                    f_out.writelines(zephyr_dict)

print(categories)