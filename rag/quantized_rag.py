import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline
)
from dotenv import load_dotenv
import os
from transformers import BitsAndBytesConfig, TextStreamer


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from elasticsearch import Elasticsearch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from time import time
load_dotenv()

es = Elasticsearch(
    [{"host": "112.137.129.158", "port": 9200}],
    http_auth=(os.getenv("ELASTIC_USERNAME"), os.getenv("ELASTIC_PASSWORD"))
)
if es.ping():
    info = es.info()
    print("Kết nối đến Elasticsearch thành công.")
    print("Phiên bản Elasticsearch hiện tại:", info["version"]["number"])
else:
    print("Không thể kết nối đến Elasticsearch.")


model_name = "phamtungthuy/law-model-version2"

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
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
    do_sample=True,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

def inference_rag(question):
    data = es.search(index="question", 
                body={
                    "track_total_hits": True,
                    "query": {
                        "match_phrase": {
                            "question": question
                        }
     
                    }
                }, size=1)
    law_ids = []
    for item in data["hits"]["hits"]:
        law_ids += item["_source"]["relevant"].split(",")
    print(law_ids)
    documents = []
    for law_id in law_ids:
        response = es.search(index="law_v2",
                            body={
                            "track_total_hits": True,
                            "query": {
                                "match_phrase": {
                                    "law_id": law_id
                                }
                            }
                        }, size=1)
        text = ""
        for item in response["hits"]["hits"]:
            source = item["_source"]
            for article in source["articles"]:
                text += article["title"] + "\n"
                text += "\n".join(article["text"]) + "\n\n"
        documents.append(Document(page_content=text, metadata={"source": "local"}))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    db = FAISS.from_documents(all_splits, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    retriever = db.as_retriever()
    
    prompt_template = """
    Dựa vào văn bản sau đây:\n{context}\nHãy trả lời câu hỏi: {question}
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )
    time1 = time()
    for chunk in rag_chain.stream(question):
        print(chunk)
        
    print("Total generation time: ", time() - time1)
inference_rag("Điều kiện an toàn về phòng cháy và chữa cháy của quán Karaoke?")