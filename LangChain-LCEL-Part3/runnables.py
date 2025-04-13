# Part 1: Basic LCEL Chain
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("Write one brief sentence about {politician}")
output_parser = StrOutputParser()


chain = prompt | model | output_parser
response = chain.invoke({"politician": "JFK"})


# Part 2: LCEL with Retrivever

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings


vectorstore = DocArrayInMemorySearch.from_texts([...], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


template = """Answer the question based only on the following context:
{context}

Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)


from langchain_core.runnables import RunnableParallel, RunnablePassthrough
get_question_and_retrieve_relevant_docs = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})


chain = get_question_and_retrieve_relevant_docs | prompt | model | output_parser
response = chain.invoke("In how many countries has AI Accelera provided services?")


# Part 3: Using FAISS and itemgetter
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts([...], embedding=OpenAIEmbeddings())


from operator import itemgetter
chain = {
    "context": itemgetter("question") | retriever,
    "question": itemgetter("question"),
    "language": itemgetter("language"),
} | prompt | model | StrOutputParser()

# Part 4: Lambda Functions in Chains
runnable = RunnableParallel(
    user_input = RunnablePassthrough(),
    transformed_output = lambda x: x["num"] + 1,
)
response = runnable.invoke({"num": 1})

#  Part 5: Chaining Chains
prompt = ChatPromptTemplate.from_template("tell me a sentence about {politician}")
chain = prompt | model | StrOutputParser()

historian_prompt = ChatPromptTemplate.from_template("Was {politician} positive for Humanity?")
composed_chain = {"politician": chain} | historian_prompt | model | StrOutputParser()

# Part 6: Fallbacks in Chains
chat_model = ChatOpenAI(model="gpt-fake")  # This will fail
bad_chain = chat_prompt | chat_model | StrOutputParser()

good_chain = prompt | OpenAI()
chain = bad_chain.with_fallbacks([good_chain])

# Part 7: Nested Chains with Custom Functions
def russian_lastname_from_dictionary(person):
    return person["name"] + "ovich"

# Part 8: Multi-Step Workflow
prompt1 = ChatPromptTemplate.from_template("what is the country {politician} is from?")
prompt2 = ChatPromptTemplate.from_template("what continent is the country {country} in? respond in {language}")
