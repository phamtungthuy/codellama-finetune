import copy
import datasets

def preprocess_dataset(dataset, tokenizer):
#     dataset = datasets.load_dataset(dataset_id, split=split)
    
    prompt = (
        f"### Câu hỏi:\n{{instruction}}\n\n### Trả lời:\n"
    )
    
    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(instruction=sample["question"]),
            "message": sample["answer"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    # mx = 0

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False, max_length=500, truncation=True)
        message = tokenizer.encode(sample["message"] +  tokenizer.eos_token, max_length=800, truncation=True, add_special_tokens=False)
        max_length = 801 - len(prompt) - len(message)
        # mx = max(mx, len(prompt) + len(message))
        pad = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True)

        sample = {
            "input_ids": prompt + message + pad,
            "attention_mask" : [1] * (len(prompt) + len(message) + len(pad)),
            "labels": [-100] * len(prompt) + message + [-100] * len(pad),
            }

        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    # print(mx)
    return dataset