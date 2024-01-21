import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline,
    TextStreamer
)


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from time import time

model_name = "phamtungthuy/law-model-version2"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True,
                                         )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

streamer = TextStreamer(tokenizer)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    max_new_tokens=200,
    do_sample=True,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id,  
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=1
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
prompt_template = """
Dựa vào văn bản sau đây:\n{context}\nHãy trả lời câu hỏi: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

loader = TextLoader("./vbpl/vbpl.txt",
                    encoding="utf8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
retriever = vectordb.as_retriever()

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )
time1 = time()
for chunk in rag_chain.stream("Bản cáo bạch là gì?"):
    print(chunk, flush=False)
    
print("Total generation time:", time() - time1)