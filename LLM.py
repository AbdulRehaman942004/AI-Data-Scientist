from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

query_1 = "hi I'm Nouman Ejaz, How are you?"
response_1 = "Hey how are you doing, please to meet you nouman"

chat_history.append(HumanMessage(content=query_1))
chat_history.append(AIMessage(content=response_1))

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=384
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI Data Scientist. Your job is to perform Exploratory Data Analysis (EDA) on datasets provided by the user.

Guidelines:
- The dataset provided by the user will be at most 10 MB.
- Assume the dataset is typically in CSV format.
- Your goal is to analyze the dataset and explain findings clearly so the user understands the data.
"""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "chat_history": chat_history,
    "input": "How would you perform EDA on my dataset?"
})

print(f"prompt: {prompt}")
print("-"*20)

chat_history.append(AIMessage(content=response))

print(f"chat_history: {chat_history}")
print("-"*20)
print(f"response: {response}")