from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.3
)

system_prompt = "You are a helpful assistant. You will help the user add tasks."
user_input = "What day is it today?"

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", user_input)
])

