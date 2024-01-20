from vllm import LLM, SamplingParams

from transformers import pipeline
import torch
import json

pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device=0)

max_response_length = 0

datas = []
full_prompts = []
responses = []
with open('all_prompts_train_with_gpt-4-1106-preview_responses_explicit_refusal.jsonl') as f_in:
    with open('f_out-mistral-vllm.jsonl', 'w') as f_out:
        with open('f_out-mistral-dangerous.jsonl', 'w') as f_out_dangerous:
            with open('f_out-mistral-offensive.jsonl', 'w') as f_out_offensive:
                i = 0
                for line in f_in.readlines():
                    data = json.loads(line)
                    prompt = data["prompt"]
                    refusal = data["response"]
                    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
                    messages = [
                        {
                            "role": "system",
                            "content": "", # TODO: do we want a system prompt?
                        },
                        {"role": "user", "content": prompt},
                    ]
                    full_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    full_prompts.append(full_prompt)
                    datas.append(data)
                    i += 1

                llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=1024,
                )
                outputs = llm.generate(full_prompts, sampling_params)

                unk_count = 0
                for data, output in zip(datas, outputs):
                    new_response = output.outputs[0].text
                    if "id" not in data:
                        id = f"extra_id_{unk_count}"
                        unk_count += 1
                    else:
                        id = data["id"]
                    out_data = {
                        "prompt": data["prompt"], # the instruction given in the various test sets.
                        "chosen": data["response"], # the response from the better model or the better rated prompt.
                        "chosen_model": "unknown_model", # where applicable
                        "rejected": new_response, # the response with the lower score or from word model.
                        "rejected_model": "HuggingFaceH4/zephyr-7b-beta", # where applicable
                        "subset": data["category"], # Will be from the various evaluation scores assigned to the model.
                        "id": id, # an incremented id for every prompt in the benchmark.
                    }
                    responses.append(out_data)
                    json.dump(out_data, f_out)
                    f_out.write('\n')
                    if data['category'] == 'Dangerous or sensitive topics':
                        json.dump(out_data, f_out_dangerous)
                        f_out_dangerous.write('\n')
                    elif data['category'] == 'Triggers for offensive language':
                        json.dump(out_data, f_out_offensive)
                        f_out_offensive.write('\n')