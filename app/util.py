from langchain_google_genai import ChatGoogleGenerativeAI
from .config import settings


def get_google_llm():
    return ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model=settings.llm_model_name,
        temperature=0,
    )


def create_graph_image(graph):
    png_image = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_image)
