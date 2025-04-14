import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from read_pdf import extract_text_from_pdf

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

# === Tạo hoặc cập nhật vectorstore từ các file PDF ===
def reload_all_data():
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            index_name = filename.replace(".pdf", "")
            vector_path = os.path.join(VECTORSTORE_DIR, index_name)

            raw_text = extract_text_from_pdf(file_path)
            if not raw_text.strip():
                st.warning(f"⚠️ Không tìm thấy nội dung từ file: {filename}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_text(raw_text)

            db = Chroma.from_texts(texts, embedding=embeddings, persist_directory=vector_path)
            db.persist()

# === Giao diện Streamlit ===
st.set_page_config(page_title="Trợ Lý AI - ANCB", layout="wide")
st.title("🤖 Trợ Lý AI - An ninh Cảng biển")

if st.button("🔄 Cập nhật dữ liệu"):
    with st.spinner("🔃 Đang xử lý lại tất cả các file PDF..."):
        reload_all_data()
    st.success("✅ Dữ liệu đã được cập nhật!")

# === Chọn file để hỏi ===
st.subheader("📁 Chọn nguồn dữ liệu để hỏi")
pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

if not pdf_files:
    st.warning("❗ Thư mục data/ chưa có file PDF nào.")
else:
    selected_file = st.selectbox("Chọn file dữ liệu", pdf_files)

    if selected_file:
        index_name = selected_file.replace(".pdf", "")
        vector_path = os.path.join(VECTORSTORE_DIR, index_name)

        if os.path.exists(vector_path):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma(persist_directory=vector_path, embedding_function=embeddings)

            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False
            )

            user_question = st.text_input("💬 Nhập câu hỏi của bạn:")

            if user_question:
                with st.spinner("🔎 Đang tìm câu trả lời..."):
                    answer = qa_chain.run(user_question)
                st.success("🗨️ Trợ lý: " + answer)
        else:
            st.warning("❗ Bạn cần bấm '🔄 Cập nhật dữ liệu' để tạo vector cho file này.")
