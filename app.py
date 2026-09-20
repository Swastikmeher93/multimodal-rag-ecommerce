import streamlit as st
from dotenv import load_dotenv

from rag_chain import build_chain, caption_image

load_dotenv()

st.set_page_config(page_title="Product assistant", layout="wide")
st.title("Multimodal product assistant")
st.caption("Ask a question, or upload a photo of something you're looking for.")

if "chain" not in st.session_state:
    with st.spinner("Loading index..."):
        try:
            st.session_state.chain, st.session_state.retriever = build_chain()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("The index is built once from your product catalog and then reused by the app.")
            st.stop()

query_text = st.chat_input("Ask about a product...")
uploaded_image = st.file_uploader("...or upload a product photo", type=["png", "jpg", "jpeg"])

question = None
if uploaded_image is not None:
    st.image(uploaded_image, caption="Your upload", width=250)
    with st.spinner("Looking at the image..."):
        caption = caption_image(uploaded_image.getvalue(), mime_type=uploaded_image.type)
    question = f"Find products similar to this: {caption}"
elif query_text:
    question = query_text

if question:
    with st.spinner("Searching the catalog..."):
        docs = st.session_state.retriever.invoke(question)
        answer = st.session_state.chain.invoke(question)

    st.markdown("### Answer")
    st.write(answer)

    if docs:
        st.markdown("### Matching products")
        cols = st.columns(len(docs))
        for col, doc in zip(cols, docs):
            meta = doc.metadata
            with col:
                if meta.get("image_url"):
                    st.image(meta["image_url"], use_container_width=True)
                st.markdown(f"**{meta.get('name')}**")
                st.write(f"${meta.get('price')}")
