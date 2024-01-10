# Thay thế 'checkpoint_path' bằng đường dẫn đến thư mục chứa checkpoint của bạncheckpoint_path = 'training_article/best_model'
model_name="phamtungthuy/trained-model"
# Giờ bạn có thể sử dụng 'model' và 'tokenizer' để dự đoán hoặc tiếp tục đào tạo

import streamlit as st
import time
import torch
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoTokenizer
)
    
PROMPT_TEMPLATE = "### Câu hỏi:\n{instruction}\n\n### Trả lời:"


st.title("LAWENGINEERING CHATBOT")

if "model" not in st.session_state:

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtyp=torch.bfloat16
    )
    print("importing model ...")
    st.session_state["model"] = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
        quantization_config=bnb_config, 
    )
    st.session_state["model"].config.use_cache = False
    st.session_state["model"].eval()
    
    st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(model_name, 
                                                                  trust_remote_code=True)
    st.session_state["tokenizer"].pad_token = st.session_state["tokenizer"].eos_token
    st.session_state["tokenizer"].padding_side = "right"
    pass
    
if "message" not in st.session_state:
    st.session_state.messages =[]

if prompt := st.chat_input("Hãy nhập vào yêu cầu?"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    st.markdown("Response is rendering...")
    
    input_prompt = PROMPT_TEMPLATE.format_map(
        {"instruction": prompt}
    )
    inputs = st.session_state["tokenizer"](input_prompt, return_tensors="pt").to("cuda")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": prompt
        }
    )
    with st.chat_message("assistant"):
        full_res = ""
        holder = st.empty()

        for _ in range(len(input_prompt)):
            input_len = inputs["input_ids"].shape[1]
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                output = st.session_state["model"].generate(
                    inputs = inputs["input_ids"].to("cuda"),
                    attention_mask=inputs["attention_mask"].to("cuda"),
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,
                    max_new_tokens=1,
                    eos_token_id=st.session_state["tokenizer"].eos_token_id,
                    pad_token_id=st.session_state["tokenizer"].pad_token_id
                )
            if output[0][-1] == st.session_state["tokenizer"].eos_token_id:
                break
            response = st.session_state["tokenizer"].batch_decode(output[0][input_len:], skip_special_tokens=True)[:1]
            full_res += ''.join(response)
            holder.markdown(full_res+ "▌")
            inputs = {
                "input_ids": output,
                "attention_mask": torch.ones(1, len(output[0]))
            }
        holder.markdown(full_res)
        # for word in prompt.split():
        #     full_res += word + " "
        #     time.sleep(0.1) 
        #     holder.markdown(full_res + "▌")
        # holder.markdown(full_res)
        
            
        
        #     st.markdown("Laweng chatbot: {}".format(prompt))
        
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_res
            }
        )