from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from utils.prompt import SUI_SETTING, INTENT_PROMPT, SUI_INPUT, INTENT_INPUT

def intent_recognize_prompt():
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTENT_PROMPT),
        MessagesPlaceholder(variable_name='history', optional=True),
        HumanMessagePromptTemplate.from_template(INTENT_INPUT)
    ])
    return chat_prompt
"""intent: str, rag_text: str="""""
def Sui_prompt_setting():
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SUI_SETTING),
        MessagesPlaceholder(variable_name='history', optional=True),
        HumanMessagePromptTemplate.from_template(SUI_INPUT)
    ])
    # chat_prompt = chat_prompt.partial(intent=intent, rag_text=rag_text)

    return chat_prompt
