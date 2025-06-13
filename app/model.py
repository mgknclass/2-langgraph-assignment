from typing_extensions import Annotated
from typing import Sequence, TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class TopicSelectionParser(BaseModel):
    topic: str = Field(description="Topic Selection")
    reason: str = Field(description="Reason for Topic Selection")


class ValidationParser(BaseModel):
    validation_status: str = Field(description="Validation Parser")
    reason: str = Field(description="Reason for Validation message")
