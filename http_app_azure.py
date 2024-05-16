from flask import Flask, request, jsonify
import fasttext
import openai
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
# from langchain.embeddings.oppenai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


# Load the fasttext model
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

# Set OpenAI API key
load_dotenv()
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_VERSION'] = "2024-02-01" # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
os.environ['OPENAI_API_BASE'] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ['OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_KEY")


# Initialize embeddings
embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        openai_api_type="azure",
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        chunk_size=1
        )

# # Create the vector store
# loader = TextLoader("dataset.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# FAISS.from_documents(docs, embeddings).save_local("faiss_doc_idx_data_sharing")

# Initialize LLM
llm = AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), model_name="gpt-4",temperature=0)

# Load the vector store
vectorStore = FAISS.load_local("faiss_doc_idx_data_sharing", embeddings, allow_dangerous_deserialization=True)

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