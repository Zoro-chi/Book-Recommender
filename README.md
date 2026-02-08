---
title: AI Book Recommender
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 📚 AI Book Recommender

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-purple.svg)

_Discover your next favorite book with AI-powered recommendations based on semantic search and emotional tone analysis_

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Technologies](#-technologies) • [Project Structure](#-project-structure)

</div>

---

## 🌟 Features

- **🤖 AI-Powered Semantic Search** - Uses OpenAI embeddings to understand the meaning behind your book preferences
- **🎭 Emotion-Based Filtering** - Filter books by emotional tone (Happy, Sad, Angry, Suspenseful, Surprising)
- **📑 Category Filtering** - Browse books by category for more targeted recommendations
- **⚡ Fast Vector Database** - Persistent Chroma vector database for quick similarity searches
- **🎨 Beautiful UI** - Modern, responsive Gradio interface with gradient themes
- **📊 Rich Book Metadata** - Displays book covers, titles, authors, and descriptions

## 🎯 How It Works

1. **Describe Your Ideal Book** - Enter a natural language description of what you're looking for
2. **Filter by Preferences** - Optionally select a category and emotional tone
3. **Get Personalized Recommendations** - Receive 16 curated book suggestions with covers and descriptions

The system uses:

- **Semantic Search**: Converts book descriptions to vector embeddings for intelligent matching
- **Emotion AI**: Analyzes emotional content in book descriptions (joy, sadness, anger, fear, surprise)
- **Hybrid Filtering**: Combines semantic similarity with category and emotion filters

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Book-recommender.git
   cd Book-recommender
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Prepare the data**

   The first time you run the application, it will automatically create a vector database from the book descriptions. This may take a few minutes.

## 💻 Usage

### Running the Application

```bash
python gradio_dashboard.py
```

The application will launch in your default web browser at `http://127.0.0.1:7860`

### Example Queries

Try these example searches to get started:

- _"A coming-of-age story about friendship and self-discovery"_ → Category: Fiction, Tone: Happy
- _"A historical novel set during World War II"_ → Category: Nonfiction
- _"An epic fantasy adventure with dragons and magic"_ → Category: Fiction, Tone: Surprising
- _"A psychological thriller with unexpected twists"_ → Category: Fiction, Tone: Suspenseful

### Data Exploration

Explore the data cleaning and analysis process in the Jupyter notebook:

```bash
jupyter notebook data-exploration.ipynb
```

## 🛠️ Technologies

| Technology                      | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| **Python 3.11+**                | Core programming language                       |
| **Gradio**                      | Interactive web UI framework                    |
| **LangChain**                   | Framework for building LLM applications         |
| **OpenAI Embeddings**           | Text vectorization for semantic search          |
| **Chroma**                      | Vector database for efficient similarity search |
| **Transformers (Hugging Face)** | Emotion analysis on book descriptions           |
| **Pandas & NumPy**              | Data manipulation and analysis                  |
| **Matplotlib & Seaborn**        | Data visualization                              |

## 📁 Project Structure

```
Book-recommender/
│
├── gradio_dashboard.py          # Main Gradio application
├── data-exploration.ipynb       # Data cleaning & EDA notebook
├── requirements.txt             # Python dependencies
│
├── books_final.csv              # Cleaned book dataset
├── tagged_descriptions.text     # Book descriptions for embeddings
│
├── chroma_db/                   # Persistent vector database (auto-generated)
├── cover-not-found.jpg          # Placeholder for missing covers
│
└── .env                         # Environment variables (not tracked)
```

## 📊 Dataset

The project uses a curated dataset of books containing:

- **ISBN-13** - Unique book identifier
- **Title** - Book title
- **Authors** - Author(s) name(s)
- **Description** - Book synopsis/summary
- **Categories** - Book genre/category
- **Thumbnails** - Cover image URLs
- **Emotion Scores** - Joy, sadness, anger, fear, surprise metrics

## 🎨 Features in Detail

### Semantic Search

The system converts your natural language query into a vector embedding and finds books with similar meaning in their descriptions, even if they don't share the exact same words.

### Emotion Analysis

Each book description is analyzed for emotional content using transformer models, allowing you to find books that match your desired mood.

### Persistent Vector Database

The Chroma database saves embeddings locally, so subsequent runs are much faster—no need to re-embed thousands of books each time!

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Book data sourced from Kaggle
- OpenAI for embeddings API
- Gradio team for the amazing UI framework
- Hugging Face for transformer models

---

<div align="center">

**Built with ❤️ by Zoro-chi**

⭐ Star this repo if you found it helpful!

</div>
