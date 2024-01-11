import torch
from transformers import (
    AutoModelForCausalLM, 
    CodeLlamaTokenizer,
    default_data_collator, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from utils.tokenizer import preprocess_dataset
from contextlib import nullcontext
import argparse
from datasets import load_dataset
import bitsandbytes as bnb
import wandb
wandb.login()


def find_linear_layers(model):
    """ find linear layers in given transformer model """
    lora_module_names = set()
    for name, module in model.named_modules():
        # 4 bits for qlora
        if isinstance(module, bnb.nn.Linear4bit): 
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training
    )

    target_modules = find_linear_layers(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules
    )

    # prepare int-8 model for training
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

def finetune(model_name, dataset_id):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtyp=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
        quantization_config=bnb_config, 
    )

    model.config.use_cache = False
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                            trust_remote_code=True,
                                            )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    train_dataset = load_dataset(dataset_id, split="train")
    eval_dataset = load_dataset(dataset_id, split="validation")
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer)
    model, lora_config = create_peft_config(model)

    training_arguments = TrainingArguments(
        output_dir="trained-model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2, # 4
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=10,
        learning_rate=2e-4,
        group_by_length=True,
        logging_strategy="steps",
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.model.push_to_hub("phamtungthuy/test")
    trainer.tokenizer.push_to_hub("phamtungthuy/test")
    trainer.model.save_pretrained("./trained-model")
    trainer.tokenizer.save_pretrained("./trained-model")
if __name__ == '__main__':
    finetune(model_name="vinai/PhoGPT-7B5-Instruct", dataset_id="phamtungthuy/cauhoiphapluat_400tokenanswer")
