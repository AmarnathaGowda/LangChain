import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS


from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


loaded_document = TextLoader('./data/state_of_the_union.txt').load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)


vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())


question = "What did the president say about the John Lewis Voting Rights Act?"

response = vector_db.similarity_search(question)


print("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?")
print(response[0].page_content)

vector_db = FAISS.from_documents(chunks_of_text, OpenAIEmbeddings())


retriever = vector_db.as_retriever(search_kwargs={"k": 1})

response = retriever.invoke("what did he say about ketanji brown jackson?")
