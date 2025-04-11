# Step 1: Load API Key
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Step 2: Import and Initialize the Model

from langchain_openai import OpenAI 
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

llmModel = OpenAI()

# Part 1: Simple JSON Output using SimpleJsonOutputParser

# Step 3: Create a Prompt Template
json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

# Step 4: Use the Simple JSON Output Parser
json_parser = SimpleJsonOutputParser()

# Step 5: Create the Chain
json_chain = json_prompt | llmModel | json_parser

# Step 6: Call the Chain
response = json_chain.invoke({"question": "Which is the biggest country?"})

# Step 7: Print the Output
print(response)

print("-----------------------------------------------------------------------------------")

# Part 2: Structured Output using Pydantic + JsonOutputParser

# Step 1: Define the JSON Format using Pydantic
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

# {
#   "setup": "...",
#   "punchline": "..."
# }

# Step 2: Setup the Parser
parser = JsonOutputParser(pydantic_object=Joke)

# Step 3: Create the Prompt with Format Instructions

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Step 4: Create the Chain
chain = prompt | llmModel | parser

# Step 5: Ask for a Joke
response = chain.invoke({"query": "Tell me a joke."})

# Step 6: Print the Output
print(response)

