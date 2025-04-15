import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pickle

# Đường dẫn thư mục chứa file PDF
DATA_DIR = "data"
VECTOR_DIR = "vectorstores"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

st.set_page_config(page_title="Trợ Lý AI - An ninh Cảng biển")
st.title("Trợ Lý AI - An ninh Cảng biển")
st.markdown("---")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Hàm xử lý lại toàn bộ file PDF thành vector
def reload_all_data():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIR, pdf_file)
        vector_path = os.path.join(VECTOR_DIR, pdf_file + ".pkl")

        if os.path.exists(vector_path):
            continue  # Đã có vector, bỏ qua

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        db = FAISS.from_documents(texts, embeddings)
        with open(vector_path, "wb") as f:
            pickle.dump(db, f)

# UI - cập nhật dữ liệu
if st.button("🔄 Cập nhật dữ liệu"):
    with st.spinner("Đang xử lý dữ liệu..."):
        reload_all_data()
    st.success("✅ Dữ liệu đã được cập nhật!")

# Giao diện chọn file
st.markdown("## 📁 Chọn nguồn dữ liệu để hỏi")
pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
selected_file = st.selectbox("Chọn file dữ liệu", pdf_files)

vector_path = os.path.join(VECTOR_DIR, selected_file + ".pkl")
if not os.path.exists(vector_path):
    st.warning("❗ Bạn cần bấm '🔄 Cập nhật dữ liệu' để tạo vector cho file này.")
else:
    st.success("✅ Dữ liệu đã sẵn sàng để hỏi.")
