# Step 1: Import required libraries
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

# Step 2: Load the OpenAI API key
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

#  Step 3: Create the LLM model
llmModel = OpenAI()

# Step 4: Send a simple prompt
response = llmModel.invoke("Tell me one fun fact about the Donald Trump.")
print(response)

# Step 5: Streaming the output
for chunk in llmModel.stream("Tell me one fun fact about the Donald Trump family."):
    print(chunk, end="", flush=True)

# Step 6: Use temperature to change creativity
creativeLlmModel = OpenAI(temperature=0.9)
response = llmModel.invoke("Write a short 5 line poem about Donald Trump")
print(response)


# Step 7: Using Chat Model for conversation
chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Step 8: Send system and human messages
messages = [
    ("system", "You are an historian expert in the Donald Trump family."),
    ("human", "Tell me one curious thing about Donald Trump."),
]
response = chatModel.invoke(messages)
print(response.content)

# Step 9: Streaming Chat response
for chunk in chatModel.stream(messages):
    print(chunk.content, end="", flush=True)


