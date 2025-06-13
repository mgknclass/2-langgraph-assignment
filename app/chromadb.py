from loguru import logger
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import settings


def get_split_documents():
    dir_loader = DirectoryLoader(
        path=settings.data_path, glob="./*.txt", loader_cls=TextLoader
    )
    docs = dir_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return splitter.split_documents(docs)


def get_chromadb_retriever():
    docs = get_split_documents()
    db = Chroma.from_documents(
        documents=docs,
        embedding=HuggingFaceEmbeddings(model_name=settings.embedding_model_name),
    )
    return db.as_retriever(search_kwargs={"k": 3})
