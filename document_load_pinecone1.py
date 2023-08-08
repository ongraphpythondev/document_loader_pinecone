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
import tiktoken
def delete_index(index_value):

    index.delete(delete_all=True, namespace=index_value)
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()
    
    return True
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

    
# Sidebar contents

load_dotenv()
try: 
    openai_api_key=st.secrets["OPENAI_API_KEY"]
    Pinecone_ENV=st.secrets["PINECONE_ENV"]
    Pinecone_API=st.secrets["PINECONE_API_KEY"]
    max_token_user=st.secrets["MAX_TOKEN_USER"]

    demo=st.secrets['DEMO']    
    

except:
    openai_api_key=os.getenv('OPENAI_API_KEY')
    # print(openai_api_key)
    max_token_user=os.getenv("MAX_TOKEN_USER")
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
            st.write(f'{index_name}')
            # st.write(chunks)
            Index_name="pdffile"
            
            
            
            active_indexes = pinecone.list_indexes()


            if Index_name not in active_indexes:
                print('creating index.....')
                st.write('creating index.....')
                pinecone.create_index(name=Index_name,metric='cosine',dimension=1536,)
                time.sleep(10)
            global index
            index = pinecone.Index(Index_name) 
            
            print(pinecone.list_indexes())
            nameSpace=index.describe_index_stats()
            print(nameSpace)
            if index_name not in nameSpace["namespaces"].keys():
                st.write("Creating namespace")
                print("Creating namespace")
                docsearch = Pinecone.from_texts(chunks, embedding=embeddings, index_name=Index_name,namespace=index_name)


            # print(active_indexes)

            
            time.sleep(5)
            index = pinecone.Index(Index_name) 
            st.write('use existing index.....')
            print(index.describe_index_stats())
            docsearch = Pinecone.from_existing_index(Index_name, embeddings,namespace=index_name)
    

            query = st.text_input("Ask questions about your PDF file:")

            st.button(label='Delete PDF/Vectors',help='Delete the uploaded pdf file',on_click=delete_index ,args=[index_name])
            
            # st.write(query)
            llm = OpenAI(openai_api_key=openai_api_key,temperature=0,)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            user_token=num_tokens_from_string(query, "gpt-3.5-turbo")
            print("USER TOKEN COUNT: ", user_token)
            if query :
                if user_token < int(max_token_user):
                    print(query)
                    docs = docsearch.similarity_search(query,k=3)
                    print(docs)
                # st.write(docs[0].page_content)
                # docs = VectorStore.similarity_search(query=query, k=3)

                
                

                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)
                else:
                    st.write(f"EXCEED ALLOCATED PROMPT,\n MAX TOKEN: {max_token_user} \n YOUR TOKEN: {user_token}")

    else:
        st.header("This App is Private!!!")            

if __name__ == '__main__':
    main()