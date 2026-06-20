"""Streamlit UI for the visual similarity search engine."""

import io
import math
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Visual Similarity Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(234, 88, 12, 0.10), transparent 24%),
                linear-gradient(180deg, #fcfbf8 0%, #f3efe5 100%);
            color: #0f172a;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fffdf8 0%, #f6f0e5 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }

        .hero-card {
            padding: 1.5rem 1.6rem;
            border-radius: 24px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.74);
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.25rem;
        }

        .hero-card h1 {
            margin: 0;
            font-size: clamp(2rem, 5vw, 3.4rem);
            line-height: 1.05;
        }

        .hero-card p {
            margin: 0.65rem 0 0;
            color: #475569;
            max-width: 58rem;
        }

        .result-card {
            padding: 0.85rem;
            border-radius: 18px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }

        .result-meta {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            align-items: baseline;
            margin-bottom: 0.55rem;
            color: #475569;
            font-size: 0.92rem;
        }

        .result-title {
            font-weight: 700;
            color: #0f766e;
        }

        .small-note {
            color: #64748b;
            font-size: 0.88rem;
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, #0f766e, #14b8a6);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fetch_health(api_url: str):
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_gallery_image(api_url: str, index_id: int):
    try:
        response = requests.get(f"{api_url}/image/{index_id}", timeout=10)
        if response.ok:
            return response.content
    except Exception:
        return None
    return None


st.markdown(
    """
    <div class="hero-card">
        <h1>Deep Visual Similarity Search</h1>
        <p>
            Upload an image, let a pretrained CNN turn it into an embedding, and retrieve the closest
            gallery images with FAISS in milliseconds.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=24, value=9, step=1)
    cols_per_row = st.radio("Grid columns", options=[2, 3, 4], index=1, horizontal=True)

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        """
        1. Upload a query image
        2. ResNet50 or EfficientNet-B0 encodes it into a vector
        3. FAISS retrieves the nearest gallery embeddings
        4. The UI renders the top matches with similarity scores
        """
    )

    st.divider()
    health = _fetch_health(API_URL)
    if health:
        st.success("Backend online")
        st.metric("Gallery size", f"{health['gallery_size']:,}")
        st.caption("Device: {0}".format(health["device"]))
    else:
        st.error("Backend offline. Start uvicorn on port 8000.")

    st.caption("API: {0}".format(API_URL))

uploaded_file = st.file_uploader(
    "Drop a query image here",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload any image to find visually similar items in the gallery.",
)

if uploaded_file is not None:
    query_bytes = uploaded_file.getvalue()
    query_image = Image.open(io.BytesIO(query_bytes)).convert("RGB")

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.subheader("Query")
        st.image(query_image, use_container_width=True)
        st.caption(uploaded_file.name)
        st.caption("{0} x {1} px".format(query_image.size[0], query_image.size[1]))

    with right:
        with st.spinner("Searching for similar images..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    files={"file": (uploaded_file.name, query_bytes, uploaded_file.type)},
                    params={"top_k": top_k},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Is uvicorn running on port 8000?")
                st.stop()
            except Exception as exc:
                st.error("Search failed: {0}".format(exc))
                st.stop()

        results = data.get("results", [])
        st.subheader("Top {0} Similar Images".format(len(results)))

        if not results:
            st.info("No results returned from the backend.")
        else:
            rows = math.ceil(len(results) / cols_per_row)
            for row_index in range(rows):
                columns = st.columns(cols_per_row, gap="small")
                for col_index in range(cols_per_row):
                    result_index = row_index * cols_per_row + col_index
                    if result_index >= len(results):
                        break

                    result = results[result_index]
                    with columns[col_index]:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        image_bytes = _fetch_gallery_image(API_URL, result["index_id"])
                        if image_bytes:
                            gallery_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            st.image(gallery_image, use_container_width=True)
                        else:
                            st.warning("Image unavailable")

                        similarity_pct = max(0.0, min(float(result["similarity"]), 1.0)) * 100.0
                        st.markdown(
                            "<div class='result-meta'><span class='result-title'>Rank #{0}</span><span>{1:.1f}% similar</span></div>".format(
                                result["rank"], similarity_pct
                            ),
                            unsafe_allow_html=True,
                        )
                        st.progress(int(round(similarity_pct)))
                        st.caption(Path(result["image_path"]).name)
                        st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Raw results"):
                st.dataframe(
                    [
                        {
                            "rank": result["rank"],
                            "similarity": "{0:.6f}".format(result["similarity"]),
                            "path": result["image_path"],
                        }
                        for result in results
                    ],
                    use_container_width=True,
                )
else:
    st.info("Upload a query image to start searching the gallery.")
