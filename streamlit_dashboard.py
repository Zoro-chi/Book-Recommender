import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import streamlit as st

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISSING_COVER_URL = "https://via.placeholder.com/150"

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in .env file or HF Space secrets."
    )

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Page config
st.set_page_config(
    page_title="AI Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .book-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        height: 100%;
        transition: transform 0.2s;
    }
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .book-image {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .book-title {
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .book-description {
        font-size: 0.9em;
        color: #666;
        line-height: 1.4;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load data
@st.cache_data
def load_books():
    books = pd.read_csv("books_final.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        MISSING_COVER_URL,
        books["large_thumbnail"],
    )
    return books


books = load_books()


# Initialize vector database
@st.cache_resource
def initialize_db():
    persist_directory = "./chroma_db"

    if os.path.exists(persist_directory):
        print("Loading existing vector database...")
        db_books = Chroma(
            persist_directory=persist_directory, embedding_function=OpenAIEmbeddings()
        )
    else:
        print("Creating vector database (this will take a moment)...")
        raw_documents = TextLoader("tagged_descriptions.text").load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1, chunk_overlap=0, separator="\n"
        )
        documents = text_splitter.split_documents(raw_documents)
        db_books = Chroma.from_documents(
            documents, OpenAIEmbeddings(), persist_directory=persist_directory
        )
        print("Vector database created and saved!")
    return db_books


db_books = initialize_db()


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recommendations = db_books.similarity_search(query, k=initial_top_k)
    books_list = [
        int(recommendation.page_content.strip('"').split()[0])
        for recommendation in recommendations
    ]
    book_recommendations = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recommendations = book_recommendations[
            book_recommendations["simple_category"] == category
        ][:final_top_k]
    else:
        book_recommendations = book_recommendations.head(final_top_k)

    if tone == "Happy":
        book_recommendations = book_recommendations.sort_values("joy", ascending=False)
    elif tone == "Surprising":
        book_recommendations = book_recommendations.sort_values(
            "surprise", ascending=False
        )
    elif tone == "Sad":
        book_recommendations = book_recommendations.sort_values(
            "sadness", ascending=False
        )
    elif tone == "Angry":
        book_recommendations = book_recommendations.sort_values(
            "anger", ascending=False
        )
    elif tone == "Suspensful":
        book_recommendations = book_recommendations.sort_values("fear", ascending=False)

    return book_recommendations


# Header
st.markdown(
    """
    <div class="main-header">
        <h1>📚 AI Book Recommender</h1>
        <p style="font-size: 1.1em; margin-top: 1rem; opacity: 0.95;">
            Discover your next favorite book with AI-powered recommendations based on content and emotional tone
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# Input Section
st.markdown("### 🔍 Tell us what you're looking for")

query = st.text_area(
    "📖 Describe your ideal book",
    placeholder="e.g., A thrilling mystery set in Victorian London with strong female characters...",
    height=100,
    help="Be as specific as you'd like! The more details, the better the recommendations.",
)

col1, col2, col3 = st.columns([2, 2, 2])

categories = ["All"] + sorted(books["simple_category"].unique().tolist())
tones = ["All", "Happy", "Sad", "Angry", "Suspensful", "Surprising"]

with col1:
    category = st.selectbox("📑 Category", categories, help="Filter by book category")

with col2:
    tone = st.selectbox(
        "🎭 Emotional Tone", tones, help="Select the mood you're looking for"
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("✨ Get Recommendations", use_container_width=True)

# Results Section
if search_button and query:
    with st.spinner("🔮 Finding your perfect books..."):
        recommendations = retrieve_semantic_recommendations(query, category, tone)

        if len(recommendations) > 0:
            st.markdown("### 🎯 Your Personalized Recommendations")

            # Display in 4 columns
            cols = st.columns(4)
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                col = cols[idx % 4]

                with col:
                    # Prepare description
                    description = row["description"]
                    truncated_description_split = description.split()
                    truncated_description = (
                        " ".join(truncated_description_split[:30]) + "..."
                    )

                    # Prepare authors
                    authors_split = row["authors"].split(";")
                    if len(authors_split) == 2:
                        authors_string = f"{authors_split[0]} and {authors_split[1]}"
                    elif len(authors_split) > 2:
                        authors_string = (
                            f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                        )
                    else:
                        authors_string = row["authors"]

                    # Display book card
                    st.image(row["large_thumbnail"], use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"by {authors_string}")
                    with st.expander("Read more"):
                        st.write(truncated_description)
        else:
            st.warning(
                "No books found matching your criteria. Try adjusting your filters!"
            )

elif search_button:
    st.warning("Please describe what kind of book you're looking for!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>Built with ❤️ by Zoro-chi</p>
    </div>
""",
    unsafe_allow_html=True,
)
