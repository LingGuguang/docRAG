
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from typing import cast 
import torch
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForSequenceClassification
from utils.get_prompt import Sui_prompt_setting, intent_recognize_prompt

from langchain.chains.conversation.base import LLMChain



class bceEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, path : str):
        # Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)

    def __call__(self, docs: Documents):
        input = {k: v for k, v in self.tokenizer(list(docs), padding=True, truncation=True, return_tensors="pt").items()}
        output = self.model(**input, return_dict=True)
        last =  output.last_hidden_state[:, 0]
        return cast(
            Embeddings, 
            last.tolist()
            )
    
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
        return response

    @property
    def _llm_type(self) -> str:
        return "baichuan2_LLM"


class QwenLLMChat(LLM):
    tokenizer : AutoTokenizer = None
    model : AutoModelForCausalLM = None
    
    def __init__(self, model_path: str):
        super().__init__()
        nf4_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          trust_remote_code=True, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        # self.model = self.model.eval()

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # response = self.model.chat(messages, self.tokenizer)
        
        return response
    
    @property
    def _llm_type(self) -> str:
        return "qwen1.5_LLM"

class myChain:
    llm_chain = None

    def __init__(self, llm, prompt, memory=None):
        self.llm_chain = LLMChain(
            llm=llm, 
            prompt=prompt,
            memory=memory,
            return_final_only=True,
            verbose=True,
            output_key="output",# 设置输出内容的占位符，这使得该输出可以直接被后续链中的template直接使用。
            # input_key="input", # 设置conversation的新输入的占位符
        )

    def invoke(self, query: str, is_output: bool=False, stream: bool=False):
        if stream:
            position = 0
            for response in self.llm_chain.invoke(input=query, stream=stream):
                if is_output:
                    print(response[position:], end='', flush=True)
                position = len(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            return response
        else:
            response = self.llm_chain.invoke(input=query, return_only_outputs=True)
            if is_output:
                print(response['output'])
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return response['output']

    
class bceRerankFunction:
    def __init__(self, path : str):
        # Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

    def run(self, docs):
        input = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt").items()
        input = {k:v for k,v in input}
        scores = self.model(**input, return_dict=True).logits.view(-1,).float()
        scores = torch.sigmoid(scores)
        return scores
 
class Sui(LLM):
    model: AutoModelForCausalLM = None 
    tokenizer: AutoTokenizer = None
    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model = self.model.eval()
        self.system_prompt = Sui_prompt_setting()

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
        return response