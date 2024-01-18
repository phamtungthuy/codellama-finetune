import torch
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel
from datasets import load_dataset
import time
model_path = "vinai/PhoGPT-7B5-Instruct"  
model_peft = "phamtungthuy/law-model-version2"
config = AutoConfig.from_pretrained(model_path)  
config.init_device = "cuda"
# config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!
  
model = AutoModelForCausalLM.from_pretrained(  
    model_path, config=config, torch_dtype=torch.bfloat16
)
# If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
model = PeftModel.from_pretrained(model,
                                  model_peft,
                                 torch_dtype=torch.bfloat16,
                                 device_map="auto")
model = model.merge_and_unload()
model.push_to_hub("phamtungthuy/peft-model-bin", use_temp_dir=False, safe_serialization=False)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  

