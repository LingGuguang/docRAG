
from langchain.memory import ConversationBufferMemory

Sui_Memory = ConversationBufferMemory(
        human_prefix="观众",
        ai_prefix="岁己",
        return_messages=True
        )