from loguru import logger
from .model import AgentState


def router_handler(state: AgentState):
    logger.debug("-->router<--")
    message = state["messages"][-1]
    logger.debug(f"router received message: {message}")

    if "usa" in message.lower():
        return "RAG CALL"
    elif "websearch" in message.lower():
        return "TAVILY CALL"
    else:
        return "LLM CALL"


def retry_handler(state: AgentState):
    logger.debug("-->retry<--")
    message = state["messages"][-1]
    logger.debug(f"retry received message: {message}")
    return message.lower()
