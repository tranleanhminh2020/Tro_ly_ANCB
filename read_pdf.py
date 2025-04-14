import os
from pathlib import Path
from typing import List
from pypdf import PdfReader

# Cho OCR nếu máy hỗ trợ (chạy local)
ENABLE_OCR = True

if ENABLE_OCR:
    try:
        from pdf2image import convert_from_path
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        OCR_AVAILABLE = True
    except ImportError:
        OCR_AVAILABLE = False
else:
    OCR_AVAILABLE = False


def extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if text.strip():
            return text
        elif OCR_AVAILABLE:
            print(f"📸 OCR đang xử lý file: {file_path}")
            return extract_text_with_ocr(file_path)
        else:
            return ""
    except Exception as e:
        print(f"[Lỗi đọc PDF]: {file_path} - {e}")
        return ""


def extract_text_with_ocr(file_path: str) -> str:
    text = ""
    try:
        images = convert_from_path(file_path)
        for img in images:
            text += pytesseract.image_to_string(img, lang='vie+eng')
        return text
    except Exception as e:
        print(f"[Lỗi OCR]: {file_path} - {e}")
        return ""


def load_all_pdf_texts(data_folder: str = "data") -> List[str]:
    texts = []
    for pdf_file in Path(data_folder).rglob("*.pdf"):
        print(f"📄 Đang xử lý: {pdf_file}")
        content = extract_text_from_pdf(str(pdf_file))
        if content.strip():
            texts.append(content)
            print(f"✅ Trích xuất thành công: {pdf_file}")
        else:
            print(f"⚠️ Không đọc được nội dung: {pdf_file}")
    return texts

# ============================
# 🔍 Phần tạo vectorstore ở đây
# ============================

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def create_vector_store():
    raw_texts = load_all_pdf_texts("data")

    if not raw_texts:
        print("🚫 Không tìm thấy nội dung nào từ PDF.")
        return

    # Tách văn bản thành đoạn nhỏ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for text in raw_texts:
        docs = text_splitter.split_text(text)
        documents.extend([Document(page_content=doc) for doc in docs])

    # Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Tạo vectorstore
    print("🔧 Đang tạo vectorstore...")
    db = FAISS.from_documents(documents, embedding_model)

    # Lưu vectorstore
    os.makedirs("vectorstore", exist_ok=True)
    db.save_local("vectorstore")
    print("✅ Đã lưu vectorstore vào thư mục vectorstore/")
