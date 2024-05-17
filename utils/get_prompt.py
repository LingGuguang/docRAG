from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,

)
from utils.prompt import (SUI_CHAT_PROMPT, SUI_INTENTION_PROMPT, INTENT_RECOG_PROMPT, SOFT_REJECTION_PROMPT, ACCEPT_PROMPT,
                          ENHANCE_ANSWER, ENHANCE_QUERY)

def intent_recognize_prompt():
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTENT_RECOG_PROMPT),
        # MessagesPlaceholder(variable_name='history', optional=False),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    return chat_prompt

def Sui_prompt_setting(status: str, intent: str=None, rag_text: str=None):
    status_set = ['chat', 'soft_reject', 'accept']
    if status not in status_set:
        raise ValueError(f'Wrong status. We got {status}, but we only accept {status_set}')

    if status == "chat":
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SUI_CHAT_PROMPT),
            MessagesPlaceholder(variable_name='history', optional=False),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chat_prompt = chat_prompt.partial(intent=intent, rag_text=rag_text)
    elif status == "accept":
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

def enhance_answer_prompt(query: str):
    chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(ENHANCE_ANSWER),
            # MessagesPlaceholder(variable_name='history', optional=False),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
    return chat_prompt

def enhance_query_prompt(query: str, nums: int=1):
    chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(ENHANCE_QUERY),
            # MessagesPlaceholder(variable_name='history', optional=False),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
    response_schema = [ResponseSchema(name="rewrite", description="重写后的问题")]
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions = output_parser.get_format_instructions()
    
    chat_prompt = chat_prompt.partial(nums=nums, format_instructions=format_instructions)
    return chat_prompt, output_parser