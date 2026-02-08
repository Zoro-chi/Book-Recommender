import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISSING_COVER_URL = "https://i.postimg.cc/Kcfp3D6T/cover-not-found.jpg"

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in .env file or HF Space secrets."
    )

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


books = pd.read_csv("books_final.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    MISSING_COVER_URL,
    books["large_thumbnail"],
)

# Check if the vector database already exists
persist_directory = "./chroma_db"

if os.path.exists(persist_directory):
    # Load existing database
    print("Loading existing vector database...")
    db_books = Chroma(
        persist_directory=persist_directory, embedding_function=OpenAIEmbeddings()
    )
else:
    # Create and persist the database
    print("Creating vector database (this will take a moment)...")
    raw_documents = TextLoader("tagged_descriptions.text").load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(
        documents, OpenAIEmbeddings(), persist_directory=persist_directory
    )
    print("Vector database created and saved!")


def retrieve_semantic_reccomendations(
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
    book_reccomendations = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_reccomendations = book_reccomendations[
            book_reccomendations["simple_category"] == category
        ][:final_top_k]
    else:
        book_reccomendations = book_reccomendations.head(final_top_k)

    if tone == "Happy":
        book_reccomendations.sort_values("joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_reccomendations.sort_values("surprise", ascending=False, inplace=True)
    elif tone == "Sad":
        book_reccomendations.sort_values("sadness", ascending=False, inplace=True)
    elif tone == "Angry":
        book_reccomendations.sort_values("anger", ascending=False, inplace=True)
    elif tone == "Suspensful":
        book_reccomendations.sort_values("fear", ascending=False, inplace=True)
    return book_reccomendations


def recommended_books(query, category, tone):
    reccomendations = retrieve_semantic_reccomendations(query, category, tone)
    result = []
    for _, row in reccomendations.iterrows():
        description = row["description"]
        truncated_description_split = description.split()
        truncated_description = " ".join(truncated_description_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_string = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_string = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_string = row["authors"]

        caption = f"{row['title']} by {authors_string}\n\n{truncated_description}"
        result.append((row["large_thumbnail"], caption))
    return result


categories = ["All"] + sorted(books["simple_category"].unique())
tones = ["All"] + ["Happy", "Sad", "Angry", "Suspensful", "Surprising"]

# Custom CSS for enhanced styling
custom_css = """
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .input-section h3, .input-section label {
        color: #333; /* Updated text color for better readability */
    }
    .gallery-section {
        margin-top: 2rem;
        padding: 1rem;
    }
    .search-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        transition: transform 0.2s !important;
        margin-top: 1.5rem !important;
    }
    .search-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:
    # Hero Section
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                <div class="main-header">
                    <h1>📚 AI Book Recommender</h1>
                    <p style="font-size: 1.1em; margin-top: 1rem; opacity: 0.95;">
                        Discover your next favorite book with AI-powered recommendations based on content and emotional tone
                    </p>
                </div>
                """,
                elem_classes=["main-header"],
            )

    # Input Section - Full Width Row
    with gr.Row(elem_classes=["input-section"]):
        with gr.Column():
            gr.Markdown("### 🔍 Tell us what you're looking for")

            query_input = gr.Textbox(
                label="📖 Describe your ideal book",
                placeholder="e.g., A thrilling mystery set in Victorian London with strong female characters...",
                lines=3,
                info="Be as specific as you'd like! The more details, the better the recommendations.",
            )

            with gr.Row():
                category_input = gr.Dropdown(
                    label="📑 Category",
                    choices=categories,
                    value="All",
                    info="Filter by book category",
                )
                tone_input = gr.Dropdown(
                    label="🎭 Emotional Tone",
                    choices=tones,
                    value="All",
                    info="Select the mood you're looking for",
                )
                submit_button = gr.Button(
                    "✨ Get Recommendations",
                    variant="primary",
                    size="lg",
                    elem_classes=["search-button"],
                )

    # Results Section - Full Width Row
    with gr.Row(elem_classes=["gallery-section"]):
        with gr.Column():
            gr.Markdown("### 🎯 Your Personalized Recommendations")
            output_gallery = gr.Gallery(
                label="",
                columns=4,
                rows=4,
                height="auto",
                object_fit="contain",
                show_label=False,
            )

    # Footer
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666;">
                <p>Built with ❤️ by Zoro-chi</p>
            </div>
            """
        )

    # Connect the button
    submit_button.click(
        fn=recommended_books,
        inputs=[query_input, category_input, tone_input],
        outputs=output_gallery,
    )

if __name__ == "__main__":
    dashboard.launch()
