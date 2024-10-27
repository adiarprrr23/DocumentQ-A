# 📄 Gemma Model Document Report Generator

## 🚀 Overview

The **Gemma Model Document Report Generator** is a powerful tool that leverages the capabilities of the Groq API and Google Generative AI to create detailed reports based on PDF documents. This application allows users to extract insights and generate comprehensive reports from multiple documents seamlessly.

## 🌟 Features

- **Automatic Report Generation**: Create in-depth reports that span at least 5 pages based on the provided context from PDF documents.
- **Structured Output**: Reports are structured with an introduction, analysis, insights, and a conclusion.
- **PDF Document Loading**: Load multiple PDF documents from a specified directory for report generation.
- **Embedding and Vector Storage**: Utilize embeddings for efficient retrieval of relevant document sections.

## 🛠️ Technologies Used

- **Python**: The primary language for the application.
- **Streamlit**: A framework for building interactive web applications.
- **LangChain**: For managing document loading, text splitting, and generative AI models.
- **Groq API**: For leveraging the Gemma model to generate reports.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
