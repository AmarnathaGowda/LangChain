
# Step 1: Setup Environment Variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Step 2: Initialize Chat Model
from langchain_openai import ChatOpenAI
chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

# step 3: Load TXT File
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredHTMLLoader, PyPDFLoader, WikipediaLoader

from langchain_core.prompts import ChatPromptTemplate


loader = TextLoader("./data/be-good.txt")
loaded_data = loader.load()


# step 4: Load CSV File
loader = CSVLoader("./data/Street_Tree_List.csv")
loaded_data = loader.load()

print(loaded_data)

#Step 5: Load HTML File
loader = UnstructuredHTMLLoader('./data/100-startups.html')
loaded_data = loader.load()

# Step 6: Load PDF File
loader = PyPDFLoader('./data/5pages.pdf')
loaded_data = loader.load_and_split()

# Step 7: Load Wikipedia Data
loader = WikipediaLoader('query="JFK", load_max_docs=1')
loaded_data = loader.load()[0].page_content

# print(loaded_data)


# Step 8: Create Prompt Template]
chat_template = ChatPromptTemplate.from_messages([
    ("human", "Answer this {question}, here is some extra {context}"),
])

# Step 9 : 9: Format the Prompt and Ask the Model]
messages = chat_template.format_messages(
    name="JFK",
    question="Where was JFK born?",
    context=loaded_data
)

response = chatModel.invoke(messages)

# Final Step: Print the Response
print("Respond from Wikipedia: Where was JFK born?")
print(response.content)

