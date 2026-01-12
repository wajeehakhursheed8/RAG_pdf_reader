 RAG PDF Chatbot – Question Answering System
A Retrieval-Augmented Generation (RAG) based chatbot that can **read PDF files and answer questions** from them using **LangChain, HuggingFace models, ChromaDB, and Gradio**.

This project is beginner-friendly and well-documented in **simple Roman Urdu + English comments**, making it ideal for students learning AI, NLP, and RAG systems.

Features
* Load and parse PDF documents
* Split large text into meaningful chunks
* Generate embeddings using Sentence Transformers
* Store embeddings in Chroma Vector Database
* Use FLAN-T5 for answer generation
* Retrieve relevant chunks using semantic search
* Interactive chat UI using Gradio
* Fully open-source & free models

How It Works (RAG Flow)
```
PDF → Text Extraction → Chunking → Embeddings → Vector DB
                                   ↓
User Question → Similar Chunks Retrieval → LLM → Answer
```

Project Structure
```
RAG-PDF-Chatbot/
│
├── documents/
│   └── ai_basics.pdf        # Your PDF file
│
├── chroma_db/               # Vector database (auto-created)
│
├── app.py                   # Main chatbot script
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
```

 Tech Stack
* **Python 3.9+**
* **LangChain**
* **HuggingFace Transformers**
* **Sentence Transformers**
* **ChromaDB**
* **Gradio**

Installation
 Clone the Repository

```bash
git clone https://github.com/your-username/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```

 Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
Install Dependencies

```bash
pip install -r requirements.txt
```
 Add Your PDF
* Place your PDF file inside the `documents/` folder
* Update the PDF path in `app.py` if needed:

```python
PDF_PATH = "documents/ai_basics.pdf"
```
 Run the Application
```bash
python app.py
```

After successful setup, Gradio will launch a **web-based chat interface** in your browser 

Example Questions
* What is Artificial Intelligence?
* Explain Machine Learning
* What are the applications of AI?
* Define Deep Learning

 Common Issues
 PDF not found

Make sure:
* `documents` folder exists
* PDF name matches exactly

 First run is slow
* Embeddings & models are downloaded only once
* Subsequent runs will be faster

 Future Improvements

* ✅ Support multiple PDFs
* ✅ Chat history memory
* ✅ Source citation display
* ✅ Streamed responses
* ✅ Docker support

Author
Wajeeha Khursheed


 Support

If you found this project helpful:
* ⭐ Star this repository
* 🧠 Use it for learning RAG systems
Happy Learning 🚀🤖

