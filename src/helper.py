
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
# from langchain_text_splitters import HuggingFaceTokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.chains.retrieval_qa.base import RetrievalQA

import os


from dotenv import load_dotenv


from src.prompt import *


from langchain_community.vectorstores import FAISS


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing. Add it to .env.")



def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''.join([page.page_content for page in data])
        
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen



def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

   
    try:
        llm_ques_gen_pipeline = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.1,
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        )
        print("Groq model loaded successfully.")
    except Exception as e:
        print(f"❌ Groq failed: {e}")


   

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)


    
    try:
        llm_answer_gen = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.1,
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        )
        print("Groq model loaded successfully.")
    except Exception as e:
        print(f"❌ Groq failed: {e}")

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                chain_type="stuff", 
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list



