# Step 1: Load Environment Variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Step 2: Import and Setup Chat Models
from langchain_groq import ChatGroq

llamaChatModel = ChatGroq(
    model="llama3-70b-8192"
)

mistralChatModel = ChatGroq(
    model="mistral-saba-24b"
)

# Step 3: Define the Chat Messages
messages = [
    ("system", "You are an historian expert in the Kennedy family."),
    ("human", "How many members of the family died tragically?"),
]

# Step 4: Get Response from LLaMA3
llamaResponse = llamaChatModel.invoke(messages)
print(llamaResponse.content)

# Step 5: Get Response from Mistral
mistralResponse = mistralChatModel.invoke(messages)
print(mistralResponse.content)

