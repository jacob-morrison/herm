from transformers import pipeline
import torch
import json

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

max_response_length = 0

with open('all_prompts_train_with_gpt-4-1106-preview_responses_explicit_refusal.jsonl') as f_in:
    for line in f_in.readlines():
        data = json.loads(line)
        response = data["response"]
        if len(response) > max_response_length:
            max_response_length = len(response)

print(max_response_length)


    # # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "", # TODO: do we want a system prompt?
    #     },
    #     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    # ]
    # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # print(outputs[0]["generated_text"])