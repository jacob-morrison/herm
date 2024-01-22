import json

model = 'dolphin'

categories = set()
with open('all_prompts_train_with_gpt-4-1106-preview_responses_explicit_refusal.jsonl') as refusals_in:
    with open(f'f_out-{model}-vllm.jsonl') as model_in:
        with open(f'filtered_refusals_pairs-{model}.jsonl', 'w') as f_out:
            refusals = list(refusals_in.readlines())
            responses = list(model_in.readlines())
            for ref_dict, response_dict in zip(refusals, responses):
                data = json.loads(ref_dict)
                if data["is_refusal"] and data["category"] in ['Dangerous or sensitive topics', 'Triggers for offensive language']:
                    categories.add(data["category"])
                    f_out.writelines(response_dict)

print(categories)