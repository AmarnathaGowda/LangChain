# Load environment variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Import Prompt Templates and models
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate

# Import OpenAI models
llmModel = OpenAI()
chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Create Examples
examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

# Define how each example looks in the prompt
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Create Few-shot template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Final chat prompt with system message
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# Combine Prompt and Model into a Chain
chain = final_prompt | chatModel

# Invoke the chain with a new sentence
response = chain.invoke({"input":"Who was JFK?"})

# Print the result
print("\n----------\n")
print("Translate: Who was JFK?")
print(response.content)
print("\n----------\n")


