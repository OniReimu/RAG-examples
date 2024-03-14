from flask import Flask, request, jsonify
import fasttext
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import dotenv_values
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

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
vectorStore = FAISS.load_local("faiss_doc_idx", embeddings)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data['message']
    history = data.get('history', [])

    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    
    language = model.predict(message)[0][0].split('__')[-1]
    template = """I want you to act as a question answering bot which uses the context mentioned and answer in a concise manner and doesn't make stuff up.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            You will answer question based on the context - {context}.
            Any topics that are NOT related to the context, you will answer : "As an AI-assistant, I will only answer the domain-specific questions, please try again."
            You will create content in""" + str(language) + """language.
            Question: {question}
            Answer:
            """
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

    result = qa_chain({"query": message, "history": history_langchain_format})
    
    history_langchain_format.append(HumanMessage(content=message))
    history_langchain_format.append(AIMessage(content=result['result']))

    response = {
        'result': result['result'],
        'history': history + [(message, result['result'])]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
