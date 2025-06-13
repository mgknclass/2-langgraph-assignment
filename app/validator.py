from loguru import logger
from .util import get_google_llm
from .model import AgentState, ValidationParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


def validator_handler(state: AgentState):
    logger.debug("-->validator<--")
    question = state["messages"][0]
    answer = state["messages"][-1]
    logger.debug(f"validator received question: {question}")
    logger.debug(f"validator received answer: {answer}")

    parser = PydanticOutputParser(pydantic_object=ValidationParser)
    template = """Your task is to take given user query and answer and validate if the answer is valid answer for the question.    
    Only respond with the valid or invalid
    User query: {question}
    Answer: {answer}
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = get_google_llm()

    chain = prompt | llm | parser
    resp = chain.invoke({"question": question, "answer": answer})
    logger.debug(f"supervisor llm response: {resp}")

    return {"messages": [resp.validation_status]}
