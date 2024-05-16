# Load model directly
from utils.basic_utils import * 
import torch
import chromadb 
import os, sys
from llm import bceEmbeddingFunction, bceRerankFunction, myChain, baichuan2LLM, QwenLLMChat
from langchain_core.documents import Document
from argparser import main_argparser
from text_search import BM25Model
from utils.get_prompt import intent_recognize_prompt, Sui_prompt_setting
from utils.get_memory import Sui_Memory
from utils.intent_clear import basic_intention_filter
from init_info import InitInfo

from langchain_chroma import Chroma

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class docRAG(InitInfo):
    parser = main_argparser()

    ### 设定文本
    chromadb_name_for_save = "chaos_history"
    dataset_dirname = 'dataset'
    docs_path = f'{dataset_dirname}/史上第一混乱碎片版.txt'

    current_path = os.path.dirname(sys.path[0])
    embedding_model_path = "models/bce-embedding-base_v1"
    # model_path = os.path.join(current_path, "LLModel/baichuan2-7B-chat")
    model_name = "qwen1.5-14B-chat"
    model_path = os.path.join(current_path, f"LLModel/{model_name}")
    rerank_model_path = os.path.join(current_path, "LLModel/bce-reranker-base_v1")
    chroma_path = "dataset/"

    RERANK_TOP_K = 3
    BM25_NUMS = 3
    RETRIEVEL_NUMS = 3    

    ### 自定义本地embedding_function
    print("bce embedding is running...")    
    bce_embedding_function = bceEmbeddingFunction(embedding_model_path)

    ### 可以用client设计pipeline
    print("get chromadb...") 
    chroma_client = chromadb.PersistentClient(chroma_path)
    persist_directory = chroma_path
    chroma_collection = Chroma(persist_directory=persist_directory, collection_name=chromadb_name_for_save, embedding_function=bce_embedding_function)

    docs = read_text(docs_path, split_line=True)
    reranker = bceRerankFunction(rerank_model_path)

    if model_name == "baichuan2-7B-chat":
        llm = baichuan2LLM(model_path)
    elif model_name == "qwen1.5-14B-chat":
        llm = QwenLLMChat(model_path)
    else:
        raise ValueError(f"wrong model named {model_name}")
    memory = Sui_Memory
    
    def run(self, query: str) -> str: 
        # 入口
        # intention recognition
        intentChain = myChain(llm=self.llm,
                            prompt=intent_recognize_prompt(),
                            # memory=self.memory
                            )
        # intention filter
        curr_intent = basic_intention_filter(intentChain.invoke(query), self.intent_set)
        print("意图：", curr_intent)
        # intention string
        curr_intent = self.intent_set[curr_intent]

        if curr_intent == "查询":
            # 准备好格式
            retrievel_docs_with_score = {
                "embedding": [],
                "bm25": []
            }
            print('查询')
            retrievel_docs_with_score['embedding'] = self.chroma_collection.similarity_search_with_score(query_texts=query, 
                                                                                 n_results=self.RETRIEVEL_NUMS) # 这里面塞query、embedding，都行，反正embedding_function已经给了
            bm25 = BM25Model(self.docs)
            bm25_docs, bm25_scores = self.retrievel_BM25(query, bm25, n_results=self.BM25_NUMS, score=True)
            retrievel_docs_with_score['bm25'] = [(doc, score) for doc, score in zip(bm25_docs, bm25_scores)]
            
            # if self.parser.hypocritical_answer:
            #     hypothetical_answer = self.hypothetical_answer_generation(query)
            #     retrievel_docs += self.retrievel(f'{query} {hypothetical_answer}')
            #     # print(retrievel_docs)
            # if int(self.parser.additional_query):
            #     additional_query = self.additional_query_generation(query, int(self.parser.additional_query))
            #     retrievel_docs += self.retrievel(additional_query)
            #     # print(retrievel_docs)

            # 格式标准化
            retrievel_docs_with_score = self._standard_retrievel_docs_with_score(retrievel_docs_with_score)

            # 检测是否拒识。
            retrievel_docs_with_score = self._refuze_recognize_pre(retrievel_docs_with_score)

            self._prep_for_rerank(query, retrievel_docs_with_score)

            text_docs = [[query, doc] for doc in list(set(retrievel_docs_with_score))]
            rerank_score = self.reranker.run(text_docs)
            
            rerank_docs = sorted([(query_and_doc[1], score) for query_and_doc, score in zip(text_docs, rerank_score)], key=lambda x: x[1], reverse=True)
            rerank_topk_docs = [doc for doc, score in rerank_docs[:self.RERANK_TOP_K]]
            rerank_concat_docs = '\n'.join(rerank_topk_docs)

            chatChain = myChain(llm=self.llm,
                            prompt=Sui_prompt_setting(intent=curr_intent, rag_text=rerank_concat_docs),
                            memory=self.memory)

            response = chatChain.invoke(query, 
                                        is_output=False,
                                        stream=False,)
        else:
            chatChain = myChain(llm=self.llm,
                            prompt=Sui_prompt_setting(intent=curr_intent),
                            memory=self.memory)

            response = chatChain.invoke(query, 
                                        is_output=False,
                                        stream=False,)

        print('memory:', self.memory)
        return response
    
    

    ### BM25
    def retrievel_BM25(self, query: str, model: BM25Model, n_results: int=3, score: bool=True) -> List[str]:
        ret = model.topk(query, k=n_results, score=score)
        return ret
    
    # def model_check(func):
    #     def wrapper(self, *args, **kwargs):
    #         if not self.model or not self.tokenizer:
    #             self.model, self.tokenizer = init_model(self.model_path)
    #         return func(self, *args, **kwargs)
    #     return wrapper

    # 有时候query与文章并不相似，我们希望通过LLM生成一个伪答案，我们期望这个伪答案与真正的答案长得有一点像，这样就能在向量数据库里找到真正的答案了。

    async def hypothetical_answer_generation(self, query: str) -> str:
        message = hypothetical_answer_template(query)
        ret = self.model.chat(self.tokenizer, message)
        return ret 

    # 你还可以生成多个表述不同的问题

    async def additional_query_generation(self, query: str, query_nums:int=1) -> str:
        message = additional_query_template(query, query_nums)
        ret = self.model.chat(self.tokenizer, message)
        return ret
    



rag = docRAG()
while True:
    # try:
    #     query = input("input query: ")
    #     if query.strip() == "exit":
    #         break
    #     response = rag.run(query)
    #     print("response:", response + '\n')
    # except:
    #     print('wrong token.')
    query = input("input query: ")
    if query.strip() == "exit":
        break
    response = rag.run(query)
    print("response:", response + '\n')
