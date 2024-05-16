from dotenv import dotenv_values
import os
from flask import Flask, request, jsonify
# import fasttext

from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from huggingface_hub import hf_hub_download
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain.schema import AIMessage, HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain.tools.retriever import create_retriever_tool
from langchain import LLMMathChain
from langchain.agents.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.load import dumps

from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner


# # Load the fasttext model
# model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
# model = fasttext.load_model(model_path)

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = dotenv_values(".env")["OPENAI_API_KEY"]


# Short-term memory
# message_history = ChatMessageHistory()
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Sensory memory
with open('prompts_agent.txt', 'r') as file:
    template = file.read()
template = template.format(
    input='{input}',
    agent_scratchpad= '{agent_scratchpad}',
)
prompt = ChatPromptTemplate.from_template(template)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)

# 1) Retriever
# Initialize embeddings
embeddings = OpenAIEmbeddings()

## Create the vector store
loader = TextLoader("dataset_data_sharing.txt")
documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
FAISS.from_documents(docs, OpenAIEmbeddings()).save_local("faiss_doc_idx_data_sharing")


# Long-term memory
vectorStore = FAISS.load_local("faiss_doc_idx_data_sharing", embeddings, allow_dangerous_deserialization=True)
retriever = vectorStore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "data_sharing_retriever",
    "Search for information about Data Sharing if necessary. For any questions about Data Sharing, you must use this tool!",
)

# 2) Calculator
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
calcuator_tool = Tool(
    name="Calculator",
    func=llm_math_chain.run,
    description="Useful for when you need to do math"
)

tools = [retriever_tool, calcuator_tool]
for t in tools:
    print(t.name, t.description)

# Initialize agent

## Normal agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

## Plan-and-execite agent
# planner = load_chat_planner(llm)
# planner.llm_chain.prompt = prompt
# executor = load_agent_executor(llm, tools, verbose=True)
# agent_executor = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # lambda session_id: message_history,
    get_session_history,
    input_messages_key="input",
    history_message_key ="chat_history",
)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data['message']
    session_id = data.get('session_id')

    result = agent_with_chat_history.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}  # Make sure session_id is provided here
    )

    response = {
        'result': result['output'],
        'input': result['input']
    }

    return dumps(response)


if __name__ == '__main__':
    app.run(debug=True)
