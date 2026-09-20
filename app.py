import os
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv

from rag_chain import (
    ProductMatch,
    SearchFilters,
    analyze_image,
    build_search_engine,
    extract_price_constraints,
)

load_dotenv()

AMAZON_MARKETPLACE = os.getenv("AMAZON_MARKETPLACE", "www.amazon.in").strip()
AMAZON_MARKETPLACE = AMAZON_MARKETPLACE.replace("https://", "").rstrip("/")


def amazon_search_url(query: str) -> str:
    return f"https://{AMAZON_MARKETPLACE}/s?k={quote_plus(query.strip())}"


def amazon_query_for_matches(query: str, products: list[dict]) -> str:
    """Use the matched product identity, not the full visual caption, for Amazon."""
    if not products:
        return query.strip()
    product = products[0]
    name = str(product.get("model") or product.get("name") or query).strip()
    brand = str(product.get("brand", "")).strip()
    if brand and brand.lower() not in name.lower():
        return f"{brand} {name}"
    return name

st.set_page_config(
    page_title="ShopLens | Multimodal product assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_search_engine():
    return build_search_engine()


def render_product_cards(products: list[dict]) -> None:
    """Render grounded product metadata as Amazon-style result cards."""
    if not products:
        return
    for start in range(0, len(products), 3):
        row_products = products[start : start + 3]
        columns = st.columns(len(row_products))
        for column, product in zip(columns, row_products):
            with column:
                image_url = product.get("image_url")
                if image_url:
                    st.image(image_url, use_container_width=True)
                st.markdown(f"**{product.get('name', 'Product')}**")
                st.markdown(f"### ${float(product.get('price', 0)):.2f}")
                rating = product.get("rating")
                reviews = product.get("review_count", 0)
                stock_status = product.get("stock_status", "Availability unknown")
                st.caption(f"{rating}★ · {reviews:,} reviews · {stock_status}")
                colors = product.get("colors", [])
                if colors:
                    st.write(f"Colors: {', '.join(colors)}")
                specification_lines = []
                for label, key in (
                    ("Model", "model"),
                    ("Chip", "chip"),
                    ("Display", "display"),
                    ("Memory", "memory"),
                    ("Storage", "storage"),
                    ("Battery", "battery_hours"),
                    ("Material", "material"),
                ):
                    value = product.get(key)
                    if value:
                        suffix = " hours" if key == "battery_hours" else ""
                        specification_lines.append(f"**{label}:** {value}{suffix}")
                features = product.get("best_for", [])
                limitations = product.get("limitations", [])
                if features:
                    specification_lines.append(f"**Best for:** {', '.join(features)}")
                if limitations:
                    specification_lines.append(f"**Limitations:** {', '.join(limitations)}")
                if specification_lines:
                    with st.expander("Specifications and features"):
                        st.markdown("\n\n".join(specification_lines))
                product_url = amazon_search_url(product.get("name", "product"))
                st.markdown(f"[View on Amazon India ↗]({product_url})")


def product_dicts(matches: list[ProductMatch]) -> list[dict]:
    return [match.product for match in matches]


try:
    engine = get_search_engine()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Run `python build_index.py` once from the project folder, then reload.")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error("The shopping engine could not start.")
    st.info("Check that GOOGLE_API_KEY is set in .env and reload the app.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("ShopLens")
st.caption("ChatGPT-style shopping help with Google Lens-style image search and catalog results.")

with st.sidebar:
    st.header("Search controls")
    category = st.selectbox("Category", ["All categories", *engine.categories])
    catalog_min, catalog_max = engine.price_range
    price_range = st.slider(
        "Price range",
        min_value=float(catalog_min),
        max_value=float(catalog_max),
        value=(float(catalog_min), float(catalog_max)),
        step=1.0,
        format="$%.0f",
    )
    in_stock_only = st.checkbox("In-stock products only", value=True)
    sort_by = st.selectbox("Sort results", ["Relevance", "Price: low to high", "Rating"])
    st.divider()
    uploaded_image = st.file_uploader(
        "Search with a product image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload an image, then click Search image or ask a question below.",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
    search_image = st.button(
        "Search image",
        type="primary",
        use_container_width=True,
        disabled=uploaded_image is None,
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_bytes"):
            st.image(
                message["image_bytes"],
                caption="Uploaded product image",
                width=260,
            )
        if message.get("visual_analysis"):
            with st.expander("Visual analysis / OCR"):
                st.json(message["visual_analysis"])
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("products"):
            st.markdown("#### Matching products")
            render_product_cards(message["products"])
        if message.get("amazon_search_url"):
            label = message.get("amazon_search_query", "these products")
            st.link_button(f"Search {label} on Amazon India ↗", message["amazon_search_url"])


query_text = st.chat_input("Ask for products, compare items, or describe what you need…")
should_search = bool(query_text and query_text.strip()) or search_image

if should_search:
    query = query_text.strip() if query_text else "Find products similar to the uploaded image."
    image_description = ""
    image_analysis = None
    image_bytes = uploaded_image.getvalue() if uploaded_image else None

    try:
        if image_bytes:
            with st.spinner("Understanding the image…"):
                image_analysis = analyze_image(
                    image_bytes,
                    mime_type=uploaded_image.type or "image/jpeg",
                )
                image_description = image_analysis.search_text

        filters = SearchFilters(
            category=None if category == "All categories" else category,
            min_price=price_range[0] if price_range[0] > catalog_min else None,
            max_price=price_range[1] if price_range[1] < catalog_max else None,
            in_stock_only=in_stock_only,
        )
        with st.spinner("Searching and reranking the catalog…"):
            matches = engine.search(
                query=query,
                image_description=image_description,
                filters=filters,
            )
        with st.spinner("Writing a grounded answer…"):
            answer = engine.answer(
                query,
                matches,
                st.session_state.messages,
                image_description=image_description,
            )
    except Exception as exc:  # noqa: BLE001
        st.error("I could not complete that search. Check your API key and try again.")
        with st.expander("Technical details"):
            st.exception(exc)
    else:
        if image_description:
            user_content = f"{query}\n\n_Visual analysis: {image_analysis.summary}_"
        else:
            user_content = query
        query_min_price, query_max_price = extract_price_constraints(query)
        if query_min_price is not None or query_max_price is not None:
            if query_min_price is not None and query_max_price is not None:
                price_note = f"${query_min_price:.2f}–${query_max_price:.2f}"
            elif query_max_price is not None:
                price_note = f"up to ${query_max_price:.2f}"
            else:
                price_note = f"from ${query_min_price:.2f}"
            user_content += f"\n\n_Price constraint applied: {price_note}_"
        st.session_state.messages.append({"role": "user", "content": user_content})
        st.session_state.messages[-1]["image_bytes"] = image_bytes
        if image_analysis:
            st.session_state.messages[-1]["visual_analysis"] = image_analysis.as_dict()

        if sort_by == "Price: low to high":
            matches.sort(key=lambda match: float(match.product.get("price", 0)))
        elif sort_by == "Rating":
            matches.sort(key=lambda match: float(match.product.get("rating", 0)), reverse=True)

        matched_products = product_dicts(matches)
        amazon_query = amazon_query_for_matches(query, matched_products)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "products": matched_products,
                "amazon_search_query": amazon_query,
                "amazon_search_url": amazon_search_url(amazon_query),
            }
        )
        st.rerun()
