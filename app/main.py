from loguru import logger
from langgraph.graph import StateGraph, START, END
from .model import AgentState
from .supervisor import supervisor_handler
from .router import router_handler, retry_handler
from .tavily import tavily_handler
from .rag import rag_handler
from .llm import llm_handler
from .validator import validator_handler
from .printer import printer_handler
from .util import create_graph_image


def main():
    logger.debug("starting...")

    # state = {"messages": ["what is the stock price of nvidia today?"]}

    # state = {"messages": ["what is the GDP of USA?"]}

    # state = {"messages": ["what is the value of 4 * 5?"]}

    state = {"messages": ["What is the most popular sport, cricket or tennis?"]}

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_handler)
    graph.add_node("llm", llm_handler)
    graph.add_node("rag", rag_handler)
    graph.add_node("tavily", tavily_handler)
    graph.add_node("validator", validator_handler)
    graph.add_node("printer", printer_handler)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        router_handler,
        {"LLM CALL": "llm", "RAG CALL": "rag", "TAVILY CALL": "tavily"},
    )
    graph.add_edge("llm", "validator")
    graph.add_edge("rag", "validator")
    graph.add_edge("tavily", "validator")

    graph.add_conditional_edges(
        "validator",
        retry_handler,
        {"invalid": "supervisor", "valid": "printer"},
    )

    graph.add_edge("validator", "printer")
    graph.add_edge("printer", END)

    chain = graph.compile()

    # create_graph_image(chain)

    resp = chain.invoke(state)
    logger.debug(f"Final Response: {resp}")

    logger.debug("completed...")


if __name__ == "__main__":
    main()
