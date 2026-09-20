"""Multimodal catalog retrieval and grounded answer generation.

The retrieval path follows the project design:

1. Turn an uploaded image into a text description with Gemini Vision.
2. Fuse the text query and image description into one retrieval query.
3. Retrieve a broad candidate set from FAISS.
4. Apply catalog filters and rerank with vector similarity, lexical overlap,
   and metadata matches.
5. Give only the reranked catalog context to the answer model.
"""

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "faiss_index"
CATALOG_PATH = PROJECT_ROOT / "products.json"


@dataclass(frozen=True)
class SearchFilters:
    """User-controlled constraints applied before final reranking."""

    category: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    in_stock_only: bool = False


@dataclass
class ProductMatch:
    """A product plus transparent retrieval scores for debugging/UI use."""

    product: dict[str, Any]
    vector_score: float
    lexical_score: float
    metadata_score: float
    final_score: float


@dataclass
class VisualAnalysis:
    """Lens-like evidence extracted from an uploaded product image."""

    category: str = "unknown"
    brand: str = "unknown"
    model: str = "unknown"
    colors: list[str] | None = None
    materials: list[str] | None = None
    features: list[str] | None = None
    visible_specs: list[str] | None = None
    ocr_text: list[str] | None = None
    keywords: list[str] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "colors",
            "materials",
            "features",
            "visible_specs",
            "ocr_text",
            "keywords",
        ):
            if getattr(self, field_name) is None:
                setattr(self, field_name, [])

    @classmethod
    def from_response(cls, raw: str) -> "VisualAnalysis":
        """Parse Gemini's JSON response with a safe text fallback."""
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            payload = json.loads(cleaned[start : end + 1]) if start >= 0 else {}
        except json.JSONDecodeError:
            payload = {}

        def value(key: str, default: Any = "unknown") -> Any:
            return payload.get(key, default) if isinstance(payload, dict) else default

        def list_value(key: str) -> list[str]:
            item = value(key, [])
            if isinstance(item, list):
                return [str(entry).strip() for entry in item if str(entry).strip()]
            if item:
                return [part.strip() for part in re.split(r"[,;]", str(item)) if part.strip()]
            return []

        if not payload:
            return cls(description=raw, keywords=sorted(_tokens(raw)))
        return cls(
            category=str(value("category")),
            brand=str(value("brand")),
            model=str(value("model")),
            colors=list_value("colors"),
            materials=list_value("materials"),
            features=list_value("features"),
            visible_specs=list_value("visible_specs"),
            ocr_text=list_value("ocr_text"),
            keywords=list_value("keywords"),
            description=str(value("description", "")),
        )

    @property
    def search_text(self) -> str:
        parts = [
            self.category,
            self.brand,
            self.model,
            self.description,
            " ".join(self.colors or []),
            " ".join(self.materials or []),
            " ".join(self.features or []),
            " ".join(self.visible_specs or []),
            " ".join(self.ocr_text or []),
            " ".join(self.keywords or []),
        ]
        return ". ".join(part for part in parts if part and part != "unknown")

    @property
    def summary(self) -> str:
        return (
            f"Category: {self.category}; Brand/model: {self.brand} {self.model}; "
            f"Colors: {', '.join(self.colors or []) or 'unknown'}; "
            f"Features: {', '.join(self.features or []) or 'unknown'}; "
            f"OCR: {', '.join(self.ocr_text or []) or 'none'}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "brand": self.brand,
            "model": self.model,
            "colors": self.colors,
            "materials": self.materials,
            "features": self.features,
            "visible_specs": self.visible_specs,
            "ocr_text": self.ocr_text,
            "keywords": self.keywords,
            "description": self.description,
        }


STOP_WORDS = {
    "a", "an", "and", "for", "from", "i", "in", "is", "me", "of",
    "on", "or", "please", "show", "the", "to", "with",
}
PRICE_NUMBER = r"(?:[$₹€£]\s*)?\d+(?:,\d{3})*(?:\.\d+)?"
CATEGORY_ALIASES = {
    "tops": {"top", "shirt", "t-shirt", "tee", "blouse", "apparel", "clothing", "sleeve"},
    "footwear": {"shoe", "sneaker", "loafer", "boot", "sandal", "footwear"},
    "outerwear": {"jacket", "coat", "parka", "outerwear", "hood", "winter"},
    "electronics": {"laptop", "computer", "keyboard", "phone", "tablet", "monitor", "electronic"},
    "kitchen": {"coffee", "mug", "kitchen", "dripper", "cookware"},
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def _number_value(value: str) -> float:
    return float(re.sub(r"[^0-9.]", "", value))


def _visual_category(image_description: str) -> str | None:
    image_tokens = _tokens(image_description)
    for category, aliases in CATEGORY_ALIASES.items():
        if image_tokens & aliases:
            return category
    return None


def extract_price_constraints(query: str) -> tuple[float | None, float | None]:
    """Extract a natural-language price range from a shopping request.

    Examples:
    - ``under $30`` -> ``(None, 30)``
    - ``between $20 and $40`` -> ``(20, 40)``
    - ``around $25`` -> approximately ``(20, 30)``
    """
    text = query.lower().replace(",", "")

    range_match = re.search(
        rf"(?:between|from)\s*({PRICE_NUMBER})\s*(?:and|to|-)\s*({PRICE_NUMBER})",
        text,
    )
    if not range_match:
        range_match = re.search(
            rf"({PRICE_NUMBER})\s*(?:to|-)\s*({PRICE_NUMBER})", text
        )
    if range_match:
        first = _number_value(range_match.group(1))
        second = _number_value(range_match.group(2))
        return min(first, second), max(first, second)

    max_match = re.search(
        rf"(?:under|below|less than|up to|upto|no more than|max(?:imum)?)\s*({PRICE_NUMBER})",
        text,
    )
    min_match = re.search(
        rf"(?:over|above|more than|at least|min(?:imum)?)\s*({PRICE_NUMBER})",
        text,
    )
    if max_match or min_match:
        return (
            _number_value(min_match.group(1)) if min_match else None,
            _number_value(max_match.group(1)) if max_match else None,
        )

    target_match = re.search(
        rf"(?:around|about|near|price\s*(?:of|is|=|:)?|budget\s*(?:of|is|=|:)?)\s*({PRICE_NUMBER})",
        text,
    )
    if not target_match:
        # A currency-marked number by itself usually means "in this price".
        target_match = re.search(rf"([$₹€£]\s*{PRICE_NUMBER})", text)
    if target_match:
        target = _number_value(target_match.group(1))
        return max(0.0, target * 0.8), target * 1.2

    return None, None


def _content_text(value: Any) -> str:
    """Normalize Gemini responses across text and content-block versions."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in value
        ).strip()
    return str(value).strip()


def load_catalog() -> list[dict[str, Any]]:
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Product catalog not found at {CATALOG_PATH}")
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def product_text(product: dict[str, Any], image_caption: str = "") -> str:
    """Create the searchable representation used by the embedding index."""
    fields = [
        product.get("name", ""),
        product.get("brand", ""),
        product.get("model", ""),
        product.get("chip", ""),
        product.get("model_year", ""),
        product.get("category", ""),
        product.get("description", ""),
        product.get("material", ""),
        product.get("display", ""),
        product.get("memory", ""),
        product.get("storage", ""),
        product.get("battery_hours", ""),
        " ".join(product.get("colors", [])),
        " ".join(product.get("tags", [])),
        " ".join(product.get("best_for", [])),
        " ".join(product.get("limitations", [])),
        image_caption,
    ]
    return ". ".join(str(field) for field in fields if field).strip()


def _product_matches_filters(
    product: dict[str, Any], filters: SearchFilters | None
) -> bool:
    if filters is None:
        return True
    if filters.category and product.get("category") != filters.category:
        return False
    price = float(product.get("price", 0))
    if filters.min_price is not None and price < filters.min_price:
        return False
    if filters.max_price is not None and price > filters.max_price:
        return False
    if filters.in_stock_only and not product.get("in_stock", False):
        return False
    return True


class MultimodalSearchEngine:
    """Hybrid catalog search plus grounded Gemini response generation."""

    def __init__(self, k: int = 6):
        index_file = INDEX_PATH / "index.faiss"
        metadata_file = INDEX_PATH / "index.pkl"
        if not index_file.is_file() or not metadata_file.is_file():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run `python build_index.py` from the project folder first."
            )

        self.default_k = k
        self.catalog = load_catalog()
        self.products_by_id = {str(item["id"]): item for item in self.catalog}
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.vectorstore = FAISS.load_local(
            INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    @property
    def categories(self) -> list[str]:
        return sorted({str(item.get("category")) for item in self.catalog})

    @property
    def price_range(self) -> tuple[float, float]:
        prices = [float(item.get("price", 0)) for item in self.catalog]
        return min(prices, default=0), max(prices, default=0)

    def search(
        self,
        query: str,
        image_description: str = "",
        filters: SearchFilters | None = None,
        k: int | None = None,
    ) -> list[ProductMatch]:
        """Retrieve, filter, and rerank products for a text/image request."""
        query_parts = [part.strip() for part in (query, image_description) if part]
        fused_query = " ".join(query_parts).strip()
        if not fused_query:
            return []

        query_min_price, query_max_price = extract_price_constraints(fused_query)
        if filters is None:
            filters = SearchFilters(
                min_price=query_min_price,
                max_price=query_max_price,
            )
        else:
            filters = SearchFilters(
                category=filters.category,
                min_price=max(
                    value
                    for value in (filters.min_price, query_min_price)
                    if value is not None
                )
                if filters.min_price is not None or query_min_price is not None
                else None,
                max_price=min(
                    value
                    for value in (filters.max_price, query_max_price)
                    if value is not None
                )
                if filters.max_price is not None or query_max_price is not None
                else None,
                in_stock_only=filters.in_stock_only,
            )

        result_count = k or self.default_k
        candidate_count = max(result_count * 4, 12)
        candidates = self.vectorstore.similarity_search_with_score(
            fused_query, k=candidate_count
        )
        query_tokens = _tokens(fused_query)
        image_tokens = _tokens(image_description)
        visual_category = _visual_category(image_description)
        requested_chip_terms = set(re.findall(r"\bm\d+\b", fused_query.lower()))
        matches: list[ProductMatch] = []
        seen_ids: set[str] = set()

        for document, distance in candidates:
            product_id = str(document.metadata.get("id", ""))
            product = self.products_by_id.get(product_id)
            if product is None or product_id in seen_ids:
                continue
            if not _product_matches_filters(product, filters):
                continue

            searchable = product_text(product)
            product_tokens = _tokens(searchable)
            lexical_score = len(query_tokens & product_tokens) / max(
                len(query_tokens), 1
            )
            image_lexical_score = len(image_tokens & product_tokens) / max(
                len(image_tokens), 1
            )
            vector_score = 1.0 / (1.0 + max(float(distance), 0.0))
            metadata_score = 0.1 if product.get("in_stock") else 0.0
            if image_description and product.get("image_url"):
                metadata_score += 0.1

            # A stale index can return a semantically broad result with weak
            # catalog-token overlap. Tie vector influence to lexical evidence
            # so an exact product newly added to products.json can still win
            # until the index is rebuilt.
            effective_vector_score = vector_score * (
                0.20 + (0.80 * lexical_score)
            )
            final_score = (
                (0.45 * effective_vector_score)
                + (0.45 * lexical_score)
                + (0.10 * metadata_score)
            )
            if image_description:
                final_score += 0.25 * image_lexical_score
                if visual_category:
                    final_score += 0.25 if product.get("category") == visual_category else -0.20
            product_chip_terms = _tokens(str(product.get("chip", "")))
            if requested_chip_terms:
                final_score += 0.35 if requested_chip_terms & product_chip_terms else -0.10
            matches.append(
                ProductMatch(
                    product=product,
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                    metadata_score=metadata_score,
                    final_score=final_score,
                )
            )
            seen_ids.add(product_id)

        # Keep catalog search useful when products were added after the last
        # FAISS build. These lexical candidates are also a safety net for
        # exact attributes such as "white", "crew neck", or "oversized".
        for product in self.catalog:
            product_id = str(product.get("id", ""))
            if product_id in seen_ids or not _product_matches_filters(product, filters):
                continue
            product_tokens = _tokens(product_text(product))
            lexical_score = len(query_tokens & product_tokens) / max(
                len(query_tokens), 1
            )
            image_lexical_score = len(image_tokens & product_tokens) / max(
                len(image_tokens), 1
            )
            if lexical_score <= 0:
                continue
            metadata_score = 0.1 if product.get("in_stock") else 0.0
            matches.append(
                ProductMatch(
                    product=product,
                    vector_score=0.0,
                    lexical_score=lexical_score,
                    metadata_score=metadata_score,
                    final_score=(0.40 * lexical_score)
                    + (0.10 * metadata_score)
                    + (0.25 * image_lexical_score)
                    + (
                        0.25
                        if visual_category == product.get("category")
                        else -0.20 if visual_category else 0.0
                    )
                    + (
                        0.35
                        if requested_chip_terms
                        and requested_chip_terms
                        & _tokens(str(product.get("chip", "")))
                        else -0.10 if requested_chip_terms else 0.0
                    ),
                )
            )
            seen_ids.add(product_id)

        matches.sort(key=lambda item: item.final_score, reverse=True)
        return matches[:result_count]

    def answer(
        self,
        question: str,
        matches: list[ProductMatch],
        conversation: list[dict[str, str]] | None = None,
        image_description: str = "",
    ) -> str:
        """Generate a concise answer using only retrieved catalog context."""
        context = format_matches(matches)
        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in (conversation or [])[-6:]
        )
        prompt = f"""
You are a helpful e-commerce shopping assistant.
Answer the customer's question using ONLY the catalog context below.
The application has already analyzed any uploaded image and provided its visual
description below. Never claim that you cannot process images. Never invent a
product, price, stock state, rating, specification, or link.
If the catalog does not contain a suitable item, say that clearly and suggest
which constraint the customer could relax. Mention why the top recommendations
fit. For purchase questions, give a clear verdict, explain the best use cases,
and call out meaningful limitations or age-related trade-offs from the catalog.
Do not turn an image description into an unsupported exact model identity.
When an image is present, organize the answer as: visual analysis, best catalog
match, verified specifications, useful features, and limitations. Clearly label
catalog specifications as verified; do not treat visual guesses as specifications.
Keep the answer conversational and concise. Product cards with exact prices
and links are rendered separately by the application.

Recent conversation:
{history or "No previous conversation."}

Catalog context:
{context or "No products matched the active filters."}

Customer question: {question}
Visual description from the uploaded image: {image_description or "None"}
""".strip()
        response = self.llm.invoke(prompt)
        return _content_text(response.content)


def format_matches(matches: list[ProductMatch]) -> str:
    lines: list[str] = []
    for rank, match in enumerate(matches, start=1):
        product = match.product
        lines.append(
            f"{rank}. {product.get('name')} | "
            f"${float(product.get('price', 0)):.2f} | "
            f"category={product.get('category')} | "
            f"stock={product.get('stock_status', 'unknown')} | "
            f"rating={product.get('rating', 'unrated')} | "
            f"chip={product.get('chip', '')} | "
            f"specifications={product.get('display', '')}; "
            f"{product.get('memory', '')}; {product.get('storage', '')}; "
            f"battery={product.get('battery_hours', '')} hours | "
            f"material={product.get('material', '')} | "
            f"best_for={', '.join(product.get('best_for', []))} | "
            f"limitations={', '.join(product.get('limitations', []))} | "
            f"description={product.get('description', '')} | "
            f"colors={', '.join(product.get('colors', []))} | "
            f"link={product.get('product_url', '')}"
        )
    return "\n".join(lines)


def build_search_engine() -> MultimodalSearchEngine:
    return MultimodalSearchEngine()


def analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> VisualAnalysis:
    """Use Gemini Vision for Lens-like visual search and OCR extraction."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    encoded = base64.b64encode(image_bytes).decode()
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Act like a visual shopping search engine. Analyze the image "
                    "and return JSON only with these keys: category, brand, model, "
                    "colors, materials, features, visible_specs, ocr_text, "
                    "keywords, description. Values for the plural keys must be "
                    "arrays of short strings. Extract visible logos, labels, model "
                    "numbers, ports, controls, shape, finish, and distinctive "
                    "design features. Use OCR for readable text. Only report what "
                    "is visible or reasonably inferred from the image. Do not "
                    "invent a chip, model, price, storage, or performance "
                    "specification; use 'unknown' when unavailable."
                ),
            },
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{encoded}",
            },
        ],
    }
    return VisualAnalysis.from_response(_content_text(llm.invoke([message]).content))


def caption_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Backward-compatible text form of :func:`analyze_image`."""
    return analyze_image(image_bytes, mime_type).search_text


# Backwards-compatible helpers for callers using the first version of this app.
def load_retriever(k: int = 4):
    engine = MultimodalSearchEngine(k=k)
    return engine.vectorstore.as_retriever(search_kwargs={"k": k})


def build_chain():
    engine = MultimodalSearchEngine()
    return engine.llm, engine
