#1. Load Environment and API Key

import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# 2. Create OpenAI Models
from langchain_openai import OpenAI
llmModel = OpenAI()


# 3. Use PromptTemplate for Single Message
chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} story about {topic}."
)

llmModelPrompt = prompt_template.format(
    adjective="curious", 
    topic="the Kennedy family"
)

# 4. Run Prompt through OpenAI Model
response = llmModel.invoke(llmModelPrompt)

print("Tell me one curious thing about the Kennedy family:")
print(response)

print("--------------------------------------------------------------------------------------------")

# 5. Use ChatPromptTemplate for Multi-Turn Conversation
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an {profession} expert on {topic}."),
        ("human", "Hello, Mr. {profession}, can you please answer a question?"),
        ("ai", "Sure!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(
    profession="Historian",
    topic="The Kennedy family",
    user_input="How many grandchildren had Joseph P. Kennedy?"
)

# 6. Get Response from ChatOpenAI
response = chatModel.invoke(messages)

print("How many grandchildren had Joseph P. Kennedy?:")
print(response.content)

print("----------------------------------------------------------------------------------")

# 7. Few-Shot Prompt for Pattern Learning
examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 8. Final Few-Shot Chat Prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# 9.Get Response from ChatOpenAI
messages = final_prompt.format_messages(
    input="good morning"
)

response = chatModel.invoke(messages)

print("good morning")
print(response.content)
