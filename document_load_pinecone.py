
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
import time
import tiktoken
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def delete_index(index_value):
    """
    Deletes the index and namespace from Pinecone.

    Args:
        index_value (str): The name of the index and namespace to be deleted.

    Returns:
        bool: True if the index and namespace were successfully deleted.
    """
    try:
        index.delete(delete_all=True, namespace=index_value)
        st.session_state["file_uploader_key"] += 1
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Error deleting index: {e}")
        return False


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


load_dotenv()


PINECONE_API_KEYs = st.session_state.get("PINECONE_API_KEYs")
openai_api_key = st.session_state.get("OPENAI_API_KEY")


def main():
    """
    Main function that runs the Streamlit app.

    The app allows users to upload a PDF file, which is then processed and stored in a Pinecone vector store.
    Users can then ask questions about the PDF, and the app will retrieve relevant information from the vector store
    and provide an answer using an OpenAI language model.
    """

    with st.sidebar:
        st.title("🤗💬 LLM Chat App")
        st.markdown(
            """
            ## About
            This app is an LLM-powered chatbot built using:
            - [Streamlit](https://streamlit.io/)
            - [LangChain](https://python.langchain.com/)
            - [OpenAI](https://platform.openai.com/docs/models) LLM model

            """
        )

        PINECONE_API_KEYs = st.text_input(
            "Pinecone API",
            type="password",
            placeholder="Paste your Pinecone API key here (sk-...)",
            help="You can get your API key from https://docs.pinecone.io/guides/get-started/quickstart#2-get-your-api-key",  # noqa: E501
            value=os.environ.get("PINECONE_API_KEYs", None)
            or st.session_state.get("PINECONE_API_KEYs", ""),
        )
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )
        
        st.session_state["PINECONE_API_KEYs"] = PINECONE_API_KEYs
        st.session_state["OPENAI_API_KEY"] = openai_api_key
        add_vertical_space(5)
    
    if not openai_api_key:
        st.warning(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    if not PINECONE_API_KEYs:
        st.warning(
            "Enter your Pinecone API key in the sidebar. You can get a key at "
            " https://docs.pinecone.io/guides/get-started/quickstart#2-get-your-api-key"
        )
    if not PINECONE_API_KEYs or not openai_api_key:
        return
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEYs
    pinecone = Pinecone(api_key=PINECONE_API_KEYs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    st.header("Chat with PDF 💬 using Pinecone Vector store")

    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    pdf = st.file_uploader(
        "Upload your PDF", type="pdf", key=st.session_state["file_uploader_key"]
    )

    # st.write(pdf)
    if pdf is not None:
        st.session_state["uploaded_files"] = pdf
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(text=text)
        print(text)

        index_name = pdf.name[:-4]
        st.write(f"{index_name}")

        Index_name = "serverless-index"

        if Index_name not in pinecone.list_indexes().names():
            print("creating index.....")
            st.write("creating index.....")
            pinecone.create_index(
                name=Index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pinecone.describe_index(index_name).status["ready"]:
                time.sleep(1)

        global index
        index = pinecone.Index(Index_name)

        nameSpace = index.describe_index_stats()

        if index_name not in nameSpace["namespaces"].keys():
            st.write("Creating namespace")
            print("Creating namespace")
            docsearch = PineconeVectorStore.from_texts(
                chunks,
                embedding=embeddings,
                index_name=Index_name,
                namespace=index_name,
            )

        

        docsearch = PineconeVectorStore(index=index,embedding= embeddings,pinecone_api_key=PINECONE_API_KEYs, namespace=index_name)

        query = st.text_input("Ask questions about your PDF file:")

        st.button(
            label="Delete PDF/Vectors",
            help="Delete the uploaded pdf file",
            on_click=delete_index,
            args=[index_name],
        )

        # st.write(query)
        llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0,
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        user_token = num_tokens_from_string(query, "gpt-3.5-turbo")

        if query:
            
            docs = docsearch.similarity_search(query, k=3)
            

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()

