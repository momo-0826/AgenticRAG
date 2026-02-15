from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    question: str
    documents: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    generation: str
    retry_count: int
    needs_retry: bool