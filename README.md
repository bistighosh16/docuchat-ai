# 🤖 DocuChat AI - Chat with Your Documents

<div align="center">

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32-red.svg)
![LangChain](https://img.shields.io/badge/langchain-0.2.16-green.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

### ✨ A beautiful RAG (Retrieval Augmented Generation) chatbot that lets you have intelligent conversations with your PDF documents using AI.

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 🌟 Features

- 🎨 **Stunning UI** - Modern glassmorphism design with smooth animations
- ⚡ **Lightning Fast** - Powered by Groq's ultra-fast LLM inference
- 🔒 **Privacy First** - Option to run completely locally with Ollama
- 🧠 **Smart Search** - Semantic search using vector embeddings
- 📚 **Source Citations** - See exactly which pages answers come from
- 💬 **Streaming Responses** - Real-time word-by-word responses like ChatGPT
- 🔄 **Auto-Switching** - Automatically uses Groq (cloud) or Ollama (local)
- 📊 **Stats Dashboard** - Track documents and chunks processed
- 🎯 **Multi-PDF Support** - Upload and query multiple PDFs at once

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Either [Ollama](https://ollama.com) installed locally **OR** a [Groq API key](https://console.groq.com)

### Installation

1. **Clone the repository**
   git clone https://github.com/YOUR_USERNAME/docuchat-ai.git
   cd docuchat-ai

2. **Create virtual environment**
    python -m venv venv
    venv\Scripts\activate  # Windows
    source venv/bin/activate  # Mac/Linux

3. **Install dependencies**
    pip install -r requirements.txt

4. **Set up environment variables**
    Create a .env file in the root directory:
    GROQ_API_KEY=your_groq_api_key_here
Or skip this step to use local Ollama (requires Ollama installed)

5. **Run the app**
    streamlit run app.py

6. **Open in browser**

💻 Usage
📁 Upload PDFs - Drop your PDF files in the sidebar
✨ Process Documents - Click the magic button to analyze your docs
💬 Ask Questions - Type any question about your documents
📚 View Sources - Expand source citations to see references

Example Questions:
"Summarize this document"
"What are the key findings?"
"Find information about [topic]"
"List all the skills mentioned"

🛠️ Tech Stack
Technology	Purpose
Python	Core language
Streamlit	Web UI framework
LangChain	LLM orchestration
Groq	Cloud LLM (Llama 3.1)
Ollama	Local LLM (Llama 3.2)
ChromaDB	Vector database
HuggingFace	Embeddings (MiniLM-L6-v2)
PyPDF	PDF processing


🧠 How It Works

1. 📄 LOAD     → Read PDF documents
2. ✂️ SPLIT    → Break into manageable chunks
3. 🔢 EMBED    → Convert text to vector embeddings
4. 💾 STORE    → Save in ChromaDB vector database
5. 🔍 RETRIEVE → Find relevant chunks for query
6. 🤖 GENERATE → AI generates answer using context

📁 Project Structure

docuchat-ai/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
├── .gitignore          # Git ignore rules
├── README.md           # You are here!
└── documents/          # Sample documents folder

🌐 Deployment
This app can be deployed on:

✅ Streamlit Cloud (Recommended)
✅ Render
✅ Hugging Face Spaces
✅ Railway


🔧 Configuration
Switch between Groq and Ollama
The app automatically detects which to use:

With GROQ_API_KEY in .env → Uses Groq (cloud, fast)
Without API key → Uses Ollama (local, private)

**Customize the LLM**
In app.py, modify the get_llm() function:
# For Groq - change model
return ChatGroq(model="llama-3.1-70b-versatile")

# For Ollama - change model
return OllamaLLM(model="mistral")

🤝 Contributing
Contributions are welcome! Feel free to:

🐛 Report bugs
💡 Suggest features
🔧 Submit pull requests

<div align="center">
⭐ If you find this project useful, please give it a star!
Made with 💜 and lots of ☕

</div>