from loguru import logger
from .config import settings
from .model import AgentState
from langchain_tavily import TavilySearch
import os


def tavily_handler(state: AgentState):
    logger.debug("-->tavily<--")
    message = state["messages"][0]
    logger.debug(f"tavily received message: {message}")

    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    tavily = TavilySearch(max_results=3, topic="general")
    resp = tavily.invoke({"query": message})
    logger.debug(f"tavily search response: {resp}")
    r = resp["results"][0]

    return {
        "messages": [f"title:{r['title']}, url: {r['url']}, content: {r['content']}"]
    }
