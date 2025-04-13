import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI,OpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


print("------------------------------------------------------------------------------------/n")
# --------------------- Conversation Buffer Memory --------------------------
print("Conversation Buffer Memory\n")

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("What is Generative AI?")
memory.chat_memory.add_ai_message("Generative AI creates new content like text or images.")

print(memory.chat_memory.messages)

print("------------------------------------------------------------------------------------/n")

# ------------------------------ Conversation Buffer Window Memory ------------------
print("Conversation Buffer Window Memory\n")

from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)
memory.chat_memory.add_user_message("What is Generative AI?")
memory.chat_memory.add_ai_message("Generative AI creates new content.")
memory.chat_memory.add_user_message("How does it work?")
memory.chat_memory.add_ai_message("It uses models trained on large datasets.")

print(memory.load_memory_variables({}))

print("------------------------------------------------------------------------------------/n")

# ---------------------------- Conversation Token Buffer Memory -------------------------
print("Conversation Token Buffer Memory\n")
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(max_token_limit=100,llm=llm)
memory.chat_memory.add_user_message("Explain token buffer memory.")
memory.chat_memory.add_ai_message("It limits tokens stored in memory.")

print(memory.load_memory_variables({}))

print("------------------------------------------------------------------------------------/n")

#------------------------- Conversation Summary Memory -----------------------------------
print("Conversation Summary Memory\n")
from langchain.memory import ConversationSummaryMemory

memory_ConvSummary = ConversationSummaryMemory(llm=OpenAI())
memory_ConvSummary.chat_memory.add_user_message("Tell me about AI.")
memory_ConvSummary.chat_memory.add_ai_message("AI stands for Artificial Intelligence.")

print(memory_ConvSummary.load_memory_variables({}))
print(memory.chat_memory.messages)

print("------------------------------------------------------------------------------------/n")

#-------------------------- CHAT MESSAGE HISTORY WORKFLOW --------------------------------
print("CHAT MESSAGE HISTORY WORKFLOW\n")
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

history = ChatMessageHistory()
history.add_message(HumanMessage(content="What is LangChain?"))
history.add_message(AIMessage(content="LangChain helps build applications powered by LLMs."))

print(history.messages)

print("------------------------------------------------------------------------------------/n")

#-------------------combine memory with LCEL chains for a real-world example------------------------
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory

# Define components
prompt = PromptTemplate.from_template("{history}\nUser: {input}\nAI:")
llm = OpenAI()
memory = ConversationBufferMemory()

# Add messages to memory
memory.chat_memory.add_user_message("What is Generative AI?")
memory.chat_memory.add_ai_message("Generative AI creates new content.")

# Create chain with memory
chain = prompt | llm
response = chain.invoke({"history": memory.chat_memory.messages, "input": "How does it work?"})

print(response)


