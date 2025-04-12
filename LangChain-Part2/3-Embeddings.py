# Step 0: Load Environment Variables
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Step 1: Import OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

# Step 2: Sample Texts to Embed
chunks_of_text = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
]

# Step 3: Create Embeddings
embeddings = embeddings_model.embed_documents(chunks_of_text)

# Step 4: Check the Embeddings Count
print("How many embeddings?")
print(len(embeddings))

# Step 5: Check One Embedding Size
print("How long is the first embedding?")
print(len(embeddings[0]))

# Step 6: Print a Sample from the Embedding]
print("Print the first 5 values of the first embedding")
print(embeddings[0][:5])

# Step 7: Embed a Query (For Search or Retrieval)
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
