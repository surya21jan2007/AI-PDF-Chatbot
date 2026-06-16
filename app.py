import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from gpt4all import GPT4All
import tempfile

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        st.success("✅ PDF Processed Successfully")

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        retrieved_docs = retriever.invoke(question)

        context = "\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful AI assistant.

Answer only using the provided context.

If the answer is not available in the context,
say:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        final_prompt = prompt_template.format(
            context=context,
            question=question
        )

        with st.spinner("Generating Answer..."):

            llm = GPT4All(
                "mistral-7b-instruct-v0.1.Q4_0.gguf"
            )

            answer = llm.generate(final_prompt)

        st.subheader("💡 Answer")
        st.write(answer)

        st.subheader("📄 Source Pages")

        pages = sorted(
            list(
                set(
                    [
                        doc.metadata.get("page", "Unknown")
                        for doc in retrieved_docs
                    ]
                )
            )
        )

        for page in pages:
            st.write(f"Page {page + 1}")