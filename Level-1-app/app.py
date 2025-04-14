# Step 1: Loading the API key
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma


# Step 2: Setting up the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")


# Step 3: Loading and preparing the document
loader = TextLoader("./data/be-good.txt")
docs = loader.load()


# Step 4: Splitting the text into small parts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# Step 5: Creating embeddings and storing them
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Step 6: Creating the retriever
retriever = vectorstore.as_retriever()



# Step 7: Creating the prompt template
prompt = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
            )
        )
    ]
)

# Step 8: Creating a function to format the docs
def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)



# Step 9: Building the RAG pipeline
rag_chain = (
   {"context": retriever | format_docs, "question": RunnablePassthrough()}
   | prompt
   | llm
   | StrOutputParser()
)

# Step 10: Asking a question
responce = rag_chain.invoke("What is this article about?")

# Step 11: Responce from LLM
print(responce)


