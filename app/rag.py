from loguru import logger
from .util import get_google_llm
from .chromadb import get_chromadb_retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from .model import AgentState


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def rag_handler(state: AgentState):
    logger.debug("-->rag<--")
    message = state["messages"][0]
    logger.debug(f"rag received message: {message}")

    retriever = get_chromadb_retriever()
    llm = get_google_llm()
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"""
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    resp = chain.invoke(message)
    logger.debug(f"response from rag llm call: {resp}")
    return {"messages": [resp]}
