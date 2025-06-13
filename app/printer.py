from loguru import logger
from .util import get_google_llm
from .model import AgentState


def printer_handler(state: AgentState):
    logger.debug("-->printer<--")
    question = state["messages"][0]
    answer = state["messages"][-1]
    logger.debug(f"Question: {question}\nAnswer: {answer}")
