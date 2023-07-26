from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    ##print(os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="Ask your pdf")
    st.header("Ask your pdf")

    pdf=st.file_uploader("Upload your pdf", type="pdf")

    #extract text
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        ##st.write(text)

        #split into chunks
        text_splitter=CharacterTextSplitter(separator='\n',chunk_size=60,chunk_overlap=20,length_function=len)
        chunks=text_splitter.split_text(text)
        ##st.write(chunks)

        #convert the chunks into embeddings,embeddings are vectors
        ##FAAIS is a library for efficient similarity search and clustering of dense vectors
        ##It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning
        ##developed by Facebook AI Research
        embeddings=OpenAIEmbeddings()
        
        #create the document/knowledge base on which u can perform semantic search
        knowledge_base=FAISS.from_texts(chunks,embeddings)
       

        user_question=st.text_input("Ask a question about your pdf")

        if user_question:
            docs=knowledge_base.similarity_search(user_question)
            
            llm=OpenAI()
            chain=load_qa_chain(llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=user_question)
                print(cb)

            st.write(response)
         
if __name__== '__main__':
    main()
 

