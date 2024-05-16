from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,

)
from utils.prompt import SUI_CHAT_PROMPT, SUI_INTENTION_PROMPT, INTENT_RECOG_PROMPT, SOFT_REJECTION_PROMPT, ACCEPT_PROMPT

def intent_recognize_prompt():
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTENT_RECOG_PROMPT),
        # MessagesPlaceholder(variable_name='history', optional=False),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    return chat_prompt

def Sui_prompt_setting(intent: str, rag_text: str="", is_reject: bool=None, is_accept: bool=None):
    if is_reject:
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SUI_CHAT_PROMPT),
            MessagesPlaceholder(variable_name='history', optional=False),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chat_prompt = chat_prompt.partial(intent=intent, rag_text=rag_text)
    elif is_accept:
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SUI_INTENTION_PROMPT),
            MessagesPlaceholder(variable_name='history', optional=False),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chat_prompt = chat_prompt.partial(intent=intent, rag_text=rag_text, soft_rejection_or_accept=ACCEPT_PROMPT)
    else:
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SUI_INTENTION_PROMPT),
            MessagesPlaceholder(variable_name='history', optional=False),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chat_prompt = chat_prompt.partial(intent=intent, rag_text=rag_text, soft_rejection_or_accept=SOFT_REJECTION_PROMPT)

    return chat_prompt
