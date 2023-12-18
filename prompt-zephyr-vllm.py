from vllm import LLM, SamplingParams

from transformers import pipeline
import torch
import json

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device=0)

max_response_length = 0

datas = []
full_prompts = []
responses = []
with open('all_prompts_train_with_gpt-4-1106-preview_responses_explicit_refusal.jsonl') as f_in:
    with open('f_out-vllm.jsonl', 'w') as f_out:
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
            if i == 100:
                break

        llm = LLM(model="HuggingFaceH4/zephyr-7b-beta")
        sampling_params = SamplingParams(
            n=1,
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
        )
        outputs = llm.generate(full_prompts, sampling_params)

        for data, output in zip(datas, outputs):
            new_response = output.outputs[0].text
            out_data = {
                "prompt": data["prompt"], # the instruction given in the various test sets.
                "chosen": data["response"], # the response from the better model or the better rated prompt.
                "chosen_model": "unknown_model", # where applicable
                "rejected": new_response, # the response with the lower score or from word model.
                "rejected_model": "HuggingFaceH4/zephyr-7b-beta", # where applicable
                "subset": data["category"], # Will be from the various evaluation scores assigned to the model.
                "id": data["id"], # an incremented id for every prompt in the benchmark.
            }
            responses.append(out_data)
            json.dump(out_data, f_out)
            f_out.write('\n')
            if i % 10 == 0:
                print(i)