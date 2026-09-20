import base64
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "faiss_index"

PROMPT = ChatPromptTemplate.from_template(
    "You are a shopping assistant. Using ONLY the product context below, "
    "answer the user's question and recommend specific products by name. "
    "If nothing in the context fits, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def format_docs(docs) -> str:
    lines = []
    for doc in docs:
        meta = doc.metadata
        lines.append(f"- {meta.get('name')} (${meta.get('price')}): {doc.page_content}")
    return "\n".join(lines)


def load_retriever(k: int = 4):
    index_file = INDEX_PATH / "index.faiss"
    metadata_file = INDEX_PATH / "index.pkl"
    if not index_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Run `python build_index.py` from the project folder first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_chain():
    retriever = load_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def caption_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Turn an uploaded product photo into a text description for retrieval."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    b64 = base64.b64encode(image_bytes).decode()
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this product in one or two sentences: category, color, style, notable features.",
            },
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64}"},
        ],
    }
    return llm.invoke([message]).content
