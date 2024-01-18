from vllm import LLM, SamplingParams
from time import time
import json
time0 = time()
llm = LLM(model="phamtungthuy/merged-peft-model", tokenizer="vinai/PhoGPT-7B5-Instruct")
print("Time to generate LLM", time() - time0)

PROMPT_TEMPLATE = "### Câu hỏi:\n{instruction}\n\n### Trả lời:" 
instruction = "Trách nhiệm của người giữ và ghi sổ kế toán đối với Công ty chứng khoán được quy định như thế nào? Chào mọi người, em có một vấn đề hi vọng được các anh chị trong Thư ký luật giải đáp. Em đang học về nghiệp vụ kế toán, kiểm toán. Vì em mong muốn làm trong Công ty chứng khoán nên cũng tìm hiểu về các quy định pháp luật liên quan tới kế toán áp dụng đối với Công ty chứng khoán. Em muốn hỏi: Trách nhiệm của người giữ và ghi sổ kế toán đối với Công ty chứng khoán được quy định như thế nào? Và văn bản pháp luật nào quy định về điều này? Mong Ban biên tập Thư Ký Luật trả lời giúp em. Xin cám ơn!" 
input_prompt = [PROMPT_TEMPLATE.format_map(  
    {"instruction": instruction}  
)]
time1 = time()
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400)

outputs = llm.generate(input_prompt, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    with open("./data.txt", "w") as file:
        file.write(json.dumps(generated_text, ensure_ascii=False))
print("Total generation:", time() - time1)
