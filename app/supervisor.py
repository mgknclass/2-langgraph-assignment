from loguru import logger
from .util import get_google_llm
from .model import TopicSelectionParser, AgentState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser


def supervisor_handler(state: AgentState):
    logger.debug("-->supervisor<--")
    message = state["messages"][0]
    logger.debug(f"supervisor received message: {message}")

    parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)
    template = """Your task is to classify the given user query into one of the following categories: [USA,NotRelated,WebSearch].
    Only respond with the category name and nothing else.

    User query: {question}
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = get_google_llm()

    chain = prompt | llm | parser
    resp = chain.invoke({"question": message})
    logger.debug(f"supervisor llm response: {resp}")

    return {"messages": [resp.topic]}
