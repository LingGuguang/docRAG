import re, json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
import torch
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        ret = json.load(f)
    return ret


def save_json(path, dic):
    with open(path,"w", encoding='utf-8') as f: 
        f.write(json.dumps(dic,ensure_ascii=False, indent=2)) 

def save_txt(path, dic):
    with open(path, 'w+', encoding='utf-8') as f:
        f.write(dic)

def init_model(path):
    # Load model directly
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def baichuan_wrapper(func) -> List[dict]:
    def wrapper(*args, **kwargs):
        system, question = func(*args, **kwargs)
        ret = [
            {"role": "system", "content": system},
            {"role": "user", "content": question}]
        return ret
    return wrapper

@baichuan_wrapper
def RAG_template_for_baichuan(query: str, informations: str) -> List[dict]:
    system = f'你是专业的中文系学生，你需要回答关于文学作品故事情节的问题。为了回答问题，你将看到一些文学作品的文字片段。你需要根据片段，总结与问题有关的信息，然后用流畅的语言回答问题。并不是每个文字片段都存在有效信息，因此当你无法根据现有信息回答问题时，请诚实地回答"基于当前材料，我无法回答问题"。'
    question  = f"""问题:{query}? \n\n文字片段: {informations}\n\n你的回答:"""
    return system, question

@baichuan_wrapper
def hypothetical_answer_template(query: str) -> str:
    system = '你是著名的网络小说作者。你的观众向你提出了一个与小说剧情有关的问题。你需要编写一段文风符合下述小说类型的小说片段，要求小说片段中的信息足以解答观众的问题。'
    question = f'问题：{query}?\n\n小说类型：都市玄幻、穿越\n\n小说片段：'
    return system, question

@baichuan_wrapper
def additional_query_template(query: str) -> str:
    system = '你是著名的网络小说作者。你的观众向你提出了一个与小说剧情有关的问题。你需要编写一段文风符合下述小说类型的小说片段，要求小说片段中的信息足以解答观众的问题。'
    question = f'问题：{query}?\n\n小说类型：都市玄幻、穿越\n\n小说片段：'
    return system, question

class baichuan2LLM(LLM):
    model: AutoModelForCausalLM = None 
    tokenizer: AutoTokenizer = None
    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model = self.model.eval()

    def _call(self, prompt:str, stream:bool=False,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any):
        # print('prompt:', prompt)
        messages = [
            {"role": "user", "content": prompt}
        ]
         # 重写调用函数
        response= self.model.chat(self.tokenizer, messages)
        print("response_from_call:",response)
        return response