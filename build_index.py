"""
Run this once (and again whenever the catalog changes) to build the FAISS index.

    python build_index.py
"""
import json
import base64
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "products.json"
INDEX_PATH = PROJECT_ROOT / "faiss_index"


def caption_image_url(vision_llm: ChatGoogleGenerativeAI, image_url: str) -> str:
    """Fetch a product image and ask Gemini Vision to describe it."""
    resp = requests.get(image_url, timeout=15)
    resp.raise_for_status()
    b64 = base64.b64encode(resp.content).decode()
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Describe this product for a shopping catalog in one sentence: "
                    "category, color, material, and style."
                ),
            },
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
        ],
    }
    return vision_llm.invoke([message]).content


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Product catalog not found at {DATA_PATH}")

    products = json.loads(DATA_PATH.read_text())
    vision_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    texts, metadatas = [], []
    for product in products:
        print(f"Captioning {product['name']}...")
        try:
            caption = caption_image_url(vision_llm, product["image_url"])
        except Exception as exc:  # noqa: BLE001
            print(f"  image captioning failed ({exc}), using text description only")
            caption = ""

        combined_text = f"{product['name']}. {product['description']} {caption}".strip()
        texts.append(combined_text)
        metadatas.append(
            {
                "id": product["id"],
                "name": product["name"],
                "price": product["price"],
                "category": product["category"],
                "image_url": product["image_url"],
            }
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(INDEX_PATH)
    print(f"\nSaved FAISS index with {len(texts)} products to ./{INDEX_PATH}")


if __name__ == "__main__":
    main()
