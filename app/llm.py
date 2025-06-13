from loguru import logger
from .util import get_google_llm
from .model import AgentState


def llm_handler(state: AgentState):
    logger.debug("-->llm<--")
    message = state["messages"][0]
    logger.debug(f"llm received message: {message}")

    llm = get_google_llm()

    resp = llm.invoke(
        "Answer the follow question with you knowledge of the real world. Following is the user question: "
        + message
    )
    logger.debug(f"response from rag llm call: {resp}")
    return {"messages": [resp]}
