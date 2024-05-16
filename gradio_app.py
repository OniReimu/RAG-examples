import sys
import subprocess
from dotenv import dotenv_values

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--quiet'])

import os
import openai
import fasttext
import gradio as gr

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


'''
    |------------------------------------------> Human: Question ---------|
    |                                                                      } LLM ---> Answer
Question ---> Vector Database ---> Relevant splits ---> System: Prompt  --|


1. Template Definition:
The template string you've defined is a PromptTemplate for the RetrievalQA chain. In this template, {context} and {question} are placeholders intended to be dynamically replaced with actual context and question text during the execution of the RetrievalQA chain.

2. Filling Placeholders:
In Langchain's RetrievalQA chain:

    - The {context} placeholder is typically filled with context information retrieved from a data source or knowledge base. This is done through the retriever component of the RetrievalQA chain, which in your case is the vectorStore.as_retriever().
    - The {question} placeholder is replaced with the actual question asked by the user. In your predict function, this corresponds to the message argument.

3. Execution Flow:
When you call qa_chain({"query": message}), the RetrievalQA chain:

    - First, uses the vectorStore retriever to fetch relevant context based on the input message.
    - Then, it formats this context and the message into the PromptTemplate you defined. This is where {context} gets filled with the retrieved information, and {question} gets replaced with the user's query (message).

'''

# # Create the vector store
# loader = TextLoader("dataset.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# FAISS.from_documents(docs, OpenAIEmbeddings()).save_local("faiss_doc_idx")

# Load the fasttext model
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = dotenv_values(".env")["OPENAI_API_KEY"]

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Load the vector store
vectorStore = FAISS.load_local("faiss_doc_idx", embeddings) # Using LangChain LLamaIndex RAG

def predict(message, history):
    history_langchain_format = []
        
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    
    language = model.predict(message)[0][0].split('__')[-1]
    with open('prompts.txt', 'r') as file:
        template = file.read()
    template = template.format(language=str(language), context='{context}', question='{question}')
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorStore.as_retriever(), 
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": QA_CHAIN_PROMPT
        }
    )

    result = qa_chain({"query": message})
    
    history_langchain_format.append(HumanMessage(content=message))
    history_langchain_format.append(AIMessage(content=result['result']))
    
    return result['result']

gr.ChatInterface(predict,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question related to PAN Services", container=False, scale=7),
    title="DocumentQABot",
    theme="soft",
    examples=["What is the cost/fees of a PAN card?", "How long does it usually take to receive the PAN card after applying?"],
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",).launch(share=True) 