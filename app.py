"""
RAG Chatbot - PDF Question Answering System
Yeh chatbot tumhari PDF ko read karke questions ka answer deta hai
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr

# ==================== STEP 1: PDF LOAD KARNA ====================
def load_pdf(pdf_path):
    """
    PDF ko load karke text extract karta hai
    """
    print("📄 PDF load ho rahi hai...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ PDF load ho gayi! Total pages: {len(documents)}")
    return documents

# ==================== STEP 2: TEXT KO CHUNKS MEIN TODNA ====================
def split_documents(documents):
    """
    Bade text ko chhote chunks mein tod deta hai
    Har chunk 500 characters ka hoga
    """
    print("✂️ Text ko chunks mein tod rahe hain...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Har chunk ki size
        chunk_overlap=50     # Overlap taki context na toote
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Total chunks bane: {len(chunks)}")
    return chunks

# ==================== STEP 3: EMBEDDINGS BANANA ====================
def create_embeddings():
    """
    Embedding model load karta hai
    Yeh text ko numbers mein convert karega
    """
    print("🔢 Embedding model load ho raha hai...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model ready!")
    return embeddings

# ==================== STEP 4: VECTOR DATABASE BANANA ====================
def create_vector_store(chunks, embeddings):
    """
    ChromaDB mein chunks ke embeddings store karta hai
    """
    print("💾 Vector database bana rahe hain...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Yahan store hoga
    )
    print("✅ Vector store ready!")
    return vectorstore

# ==================== STEP 5: LLM MODEL LOAD KARNA ====================
def load_llm():
    """
    Answer generate karne wala AI model load karta hai
    Hum Flan-T5 use kar rahe hain (free and good)
    """
    print("🤖 AI model load ho raha hai (thoda time lagega)...")
    
    model_name = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ AI model ready!")
    return llm

# ==================== STEP 6: RAG CHAIN BANANA ====================
def create_rag_chain(vectorstore, llm):
    """
    RAG pipeline setup karta hai
    """
    print("🔗 RAG chain bana rahe hain...")
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Top 3 similar chunks lao
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    print("✅ RAG chain ready!")
    return qa_chain

# ==================== STEP 7: QUESTION ANSWER FUNCTION ====================
def answer_question(question, qa_chain):
    """
    User ka question leke answer return karta hai
    """
    if not question.strip():
        return "Please enter a question!"
    
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        
        # Source documents bhi dikha sakte ho (optional)
        sources = result.get("source_documents", [])
        
        return answer
    
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== MAIN SETUP ====================
def setup_chatbot(pdf_path):
    """
    Pura setup ek baar mein karta hai
    """
    print("\n🚀 Chatbot setup shuru ho raha hai...\n")
    
    # Step by step setup
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = create_vector_store(chunks, embeddings)
    llm = load_llm()
    qa_chain = create_rag_chain(vectorstore, llm)
    
    print("\n✨ Chatbot fully ready! Ab tum questions puch sakte ho!\n")
    return qa_chain

# ==================== GRADIO UI ====================
def create_ui(qa_chain):
    """
    Web interface banata hai
    """
    def chat(message, history):
        response = answer_question(message, qa_chain)
        return response
    
    # Simple chat interface
    demo = gr.ChatInterface(
        fn=chat,
        title="🤖 RAG PDF Chatbot",
        description="Apni PDF ke baare mein kuch bhi pucho!",
        examples=[
            "What is Artificial Intelligence?",
            "Tell me about Machine Learning",
            "What are the applications of AI?",
            "Explain Deep Learning"
        ]
    )
    
    return demo

# ==================== MAIN PROGRAM ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING RAG CHATBOT...")
    print("=" * 60)
    
    # PDF ka path (apne according change karo)
    PDF_PATH = "documents/ai_basics.pdf"
    
    print(f"\n📂 Checking for PDF at: {PDF_PATH}")
    
    # Check karo PDF exist karti hai ya nahi
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF nahi mili at {PDF_PATH}")
        print("Please make sure PDF 'documents' folder mein hai!")
        exit()
    
    print("✅ PDF found!")
    print(f"📊 File size: {os.path.getsize(PDF_PATH)} bytes")
    
    try:
        # Chatbot setup karo
        qa_chain = setup_chatbot(PDF_PATH)
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Launching web interface...")
        print("=" * 60)
        
        # UI launch karo
        demo = create_ui(qa_chain)
        demo.launch(share=False)  # share=True karoge toh public link milega
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR OCCURRED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Agar error samajh nahi aaya toh screenshot bhejo!")