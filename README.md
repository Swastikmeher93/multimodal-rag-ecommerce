# Multimodal e-commerce product assistant

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
This reads `products.json`, captions each product image with Gemini Vision, embeds the combined text, and saves a FAISS index to `./faiss_index/`. Re-run it any time the catalog changes.

## 6. Run the app
```bash
streamlit run app.py
```
It opens at http://localhost:8501. Type a question or upload a photo of a product you want to match.

## Swapping in your own catalog
Replace the entries in `products.json` with your real products (id, name, description, price, category, image_url), then re-run `build_index.py`.

## Notes
- Gemini model names (`gemini-2.5-flash`, `gemini-embedding-001`) may be superseded — check the current list in Google AI Studio if you get a "model not found" error.
- FAISS here is a local, in-memory/on-disk index with no metadata filtering. If you later need filters (price range, category) alongside similarity search, look at Chroma, Qdrant, or pgvector instead.
