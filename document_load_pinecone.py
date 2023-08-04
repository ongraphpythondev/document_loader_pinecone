import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone 
from langchain.callbacks import get_openai_callback
import os
from langchain.vectorstores import Pinecone
import time
def delete_index():
    pinecone.delete_index('pdffile')
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()
    
    return True
    
# Sidebar contents

load_dotenv()
try: 
    openai_api_key=st.secrets["OPENAI_API_KEY"]
    Pinecone_ENV=st.secrets["PINECONE_ENV"]
    Pinecone_API=st.secrets["PINECONE_API_KEY"]
    demo=st.secrets['DEMO']    
    

except:
    openai_api_key=os.getenv('OPENAI_API_KEY')
    # print(openai_api_key)
    Pinecone_ENV=os.getenv("PINECONE_ENV")
    Pinecone_API=os.getenv("PINECONE_API_KEY")
    demo=os.getenv('DEMO')
def main():
    if demo:
        with st.sidebar:
            st.title('🤗💬 LLM Chat App')
            st.markdown('''
            ## About
            This app is an LLM-powered chatbot built using:
            - [Streamlit](https://streamlit.io/)
            - [LangChain](https://python.langchain.com/)
            - [OpenAI](https://platform.openai.com/docs/models) LLM model

            ''')
            add_vertical_space(5)
            
                

        pinecone.init(api_key=Pinecone_API,environment=Pinecone_ENV  )
            
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        st.header("Chat with PDF 💬 using Pinecone Vector store")


        # # upload a PDF file
        # with st.form("my-form", clear_on_submit=True):
        #     file = st.file_uploader("FILE UPLOADER")
        #     submitted = st.form_submit_button("UPLOAD!")
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 0   

        if "uploaded_files" not in st.session_state:
            st.session_state["uploaded_files"] = []
        pdf = st.file_uploader("Upload your PDF", type='pdf',key= st.session_state["file_uploader_key"])
        


        # st.write(pdf)
        if pdf is not None:
            st.session_state["uploaded_files"] =pdf
            
                

            # bytes_data = pdf.read()
            # with open(os.path.join("/tmp", pdf.name), "wb") as f:
            #     f.write(bytes_data)
            # loader = UnstructuredFileLoader(f"/tmp/{pdf.name}")
            # print(pdf)
            # print(loader)
            # docs = loader.load()
            # print(docs)
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            
            # # break
            # chunks = text_splitter.split_text(docs[0].page_content)
            # # print(chunks)
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
            chunks = text_splitter.split_text(text=text)


            # pdf_reader = PdfReader(pdf)
            
            # text = ""
            # for page in pdf_reader.pages:
            #     text += page.extract_text()

            

            # # embeddings
            index_name = pdf.name[:-4]
            st.write(f'{index_name}')
            # st.write(chunks)
            Index_name="pdffile"
            
            
            
            active_indexes = pinecone.list_indexes()


            if Index_name not in active_indexes:
                print('creating index.....')
                st.write('creating index.....')
                pinecone.create_index(name=Index_name,metric='cosine',dimension=1536,)
                index = pinecone.Index(Index_name) 
                time.sleep(5)
                print(pinecone.list_indexes())
                print(index.describe_index_stats())

                docsearch = Pinecone.from_texts(chunks, embedding=embeddings, index_name=Index_name)

            # print(active_indexes)

            else:
                index = pinecone.Index(Index_name) 
            # wait a moment for the index to be fully initialized
                # time.sleep(5)
                st.write('use existing index.....')
                print(index.describe_index_stats())
                docsearch = Pinecone.from_existing_index(Index_name, embeddings)
        

            # # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            # docsearch = Pinecone.from_texts(chunks,embedding=embeddings,index_name=index_name)
            # # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
            st.button(label='Delete PDF and indexes',help='Delete the uploaded pdf file',on_click=delete_index )
            # st.write(query)
            llm = OpenAI(openai_api_key=openai_api_key,temperature=0,)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            if query:
                print(query)
                docs = docsearch.similarity_search(query,k=3)
                # st.write(docs[0].page_content)
                # docs = VectorStore.similarity_search(query=query, k=3)

                
                

                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.write(response)
    else:
        st.header("This App is Private!!!")            

if __name__ == '__main__':
    main()