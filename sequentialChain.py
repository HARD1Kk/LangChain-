from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
prompt1 = PromptTemplate(
    template="Generate a report on topic : {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="summaries this report briefly in 5 pointer : {report}",
    input_variables=["report"],
)
output_parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

chain = prompt1 | model | output_parser | prompt2 | model | output_parser 

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)