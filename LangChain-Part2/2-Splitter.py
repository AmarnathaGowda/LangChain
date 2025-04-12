# Step 1: Load Environment Variables
import os
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI


_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


# Step 2: Load the Chat Model
chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Step 3: Load a Text File
loader = TextLoader("./data/be-good.txt")
loaded_data = loader.load()

# Step 4: Print Info
# print(loaded_data[0].page_content)

# Step 5: Split the Text into Chunks
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Step 6: Create the Chunks
texts = text_splitter.create_documents([loaded_data[0].page_content])

# Step 7: Print Chunk Info
# print("How many chunks?")
# print(len(texts))

# print("First chunk:")
# print(texts[0])

# Step 8: Add Metadata to Chunks
metadatas = [{"chunk": 0}, {"chunk": 1}]
documents = text_splitter.create_documents(
    [loaded_data[0].page_content, loaded_data[0].page_content], 
    metadatas=metadatas
)

# Step 9: Print Chunk with Metadata
print("Chunk with metadata:")
print(documents[0])