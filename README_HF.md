---
title: AI Book Recommender
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_dashboard.py
pinned: false
license: mit
---

# 📚 AI Book Recommender

Discover your next favorite book with AI-powered recommendations based on semantic search and emotional tone analysis.

## Features

- 🤖 AI-Powered Semantic Search using OpenAI embeddings
- 🎭 Emotion-Based Filtering (Happy, Sad, Angry, Suspenseful, Surprising)
- 📑 Category Filtering for targeted recommendations
- ⚡ Fast Vector Database with persistent Chroma storage
- 🎨 Beautiful, modern Gradio interface

## How to Use

1. **Describe Your Ideal Book** - Enter what you're looking for in natural language
2. **Filter by Preferences** - Select a category and emotional tone (optional)
3. **Get Recommendations** - View 16 personalized book suggestions

## Configuration

This Space requires an OpenAI API key to generate embeddings for semantic search.

Add your API key in the Space Settings → Repository Secrets:

- Name: `OPENAI_API_KEY`
- Value: Your OpenAI API key

[Get an OpenAI API key](https://platform.openai.com/api-keys)
