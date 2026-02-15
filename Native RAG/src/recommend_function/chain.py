from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.model import get_llm
from src.utils.retriever import get_retriever
from src.recommend_function.prompts import RECOMMEND_PROMPT

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def create_recommend_chain():
    llm = get_llm()
    retriever = get_retriever()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RECOMMEND_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain