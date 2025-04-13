import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


prompt = PromptTemplate.from_template("Translate '{text}' into Hindi.")
model = ChatOpenAI(model="gpt-3.5-turbo-0125")
parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"text": "How are you?"})
print(response)
