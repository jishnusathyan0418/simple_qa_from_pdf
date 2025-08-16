import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_community.document_loaders import PyPDFLoader

file_path = "./data/Be_Good.pdf"
loaded_file = PyPDFLoader(file_path)
docs = loaded_file.load()

from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

splitting_texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitting = splitting_texts.split_documents(docs)
vectorstores = Chroma.from_documents(documents=splitting, embedding=OpenAIEmbeddings())
retrievers = vectorstores.as_retriever()

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

questio_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retrievers, questio_answer_chain)
response = rag_chain.invoke({"input": "what is this article about?"})
print(response["answer"])
