import torch
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import PeftModel
from datasets import load_dataset
import time
model_base = "vinai/PhoGPT-7B5-Instruct"
model_peft="phamtungthuy/law-model-version2"
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtyp=torch.bfloat16
    )
    
tokenizer = AutoTokenizer.from_pretrained(model_base)

model = AutoModelForCausalLM.from_pretrained(
    model_base,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,
    quantization_config=bnb_config, 
)

model = PeftModel.from_pretrained(model, model_peft)
model = model.merge_and_unload()
model.save_pretrained("/trained-model")