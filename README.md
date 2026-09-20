# ShopLens — multimodal e-commerce product assistant

ShopLens combines a ChatGPT-style conversation, Google Lens-style image
understanding, and Amazon-style product discovery. It is grounded in the local
catalog, so recommendations include the catalog price, availability, rating,
image, and product link instead of invented shopping data.

## System design

```text
User text + image
       │
       ├── Gemini Vision → image attributes
       │
       └── query fusion → FAISS candidate retrieval
                              │
                    metadata filters + hybrid reranking
                              │
                    grounded catalog context
                              │
                 Gemini answer + product result cards
```

The retrieval layer applies category, price, and stock filters before ranking
the candidates with a weighted combination of vector similarity, lexical
overlap, and metadata signals. The answer model receives only the reranked
catalog context and is instructed not to invent product facts.

## 1. Open the folder
Open `ecommerce-assistant/` as a folder/workspace in your IDE (VS Code, PyCharm, etc).

## 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```
Point your IDE's Python interpreter at `.venv` (in VS Code: Cmd/Ctrl+Shift+P -> "Python: Select Interpreter").

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Add your Gemini API key
```bash
cp .env.example .env
```
Then edit `.env` and paste a key from https://aistudio.google.com/apikey.

## 5. Build the vector index (run once)
```bash
python build_index.py
```
This reads `products.json`, captions each product image with Gemini Vision,
embeds the structured product attributes, and saves a FAISS index to
`./faiss_index/`. Re-run it any time the catalog changes. The checked-in demo
index lets the app start immediately after cloning.

## 6. Run the app
```bash
streamlit run app.py
```
It opens at http://localhost:8501. Start a conversation, upload a product photo
in the sidebar, or combine both. Use the sidebar to filter by category, price,
and stock status. Each search also has a plain Amazon India keyword-search button;
it opens Amazon's `/s?k=...` results page and does not use affiliate tags or
fetch Amazon results into the app.

## Swapping in your own catalog
Replace the entries in `products.json` with your real products and re-run
`build_index.py`. Each product should include `id`, `name`, `description`,
`price`, `category`, `image_url`, `product_url`, `rating`, `review_count`,
`in_stock`, `stock_status`, `colors`, `material`, and `tags` for the best
retrieval and result-card quality.

## Notes
- Gemini model names (`gemini-2.5-flash`, `gemini-embedding-001`) may be superseded — check the current list in Google AI Studio if you get a "model not found" error.
- FAISS here is a local, in-memory/on-disk index with no metadata filtering. If you later need filters (price range, category) alongside similarity search, look at Chroma, Qdrant, or pgvector instead.
