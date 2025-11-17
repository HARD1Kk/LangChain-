from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(expression: str) -> str:
    "Evaluates Python math expressions safely"
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"


tools = [calculator]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant that can perform accurate calculations."
)

# Ask user for a calculation
user_input = input("Enter a calculation question: ")
result = agent.invoke({"messages": [("user", user_input)]})
# Extract the final answer from the last message
final_message = result["messages"][-1]
print(final_message.content)