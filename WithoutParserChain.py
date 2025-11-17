from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser # parses output strings into more structured formats

load_dotenv()

prompt = ChatPromptTemplate.from_template("Generate a report on {topic}")
model= ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_output_tokens=500
)

chain = prompt | model

result = chain.invoke({"topic": "cats"})
print(result.content)

chain.get_graph().print_ascii()