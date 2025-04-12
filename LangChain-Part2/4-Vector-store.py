import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


loaded_document = TextLoader('./data/be-good.txt').load()

text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
chunks_of_text = text_splitter.split_documents(loaded_document)


vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())

question = "What is the Y Combinator motto?"
response = vector_db.similarity_search(question)

print(response[0].page_content)

