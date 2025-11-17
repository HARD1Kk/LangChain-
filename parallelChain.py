from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv


load_dotenv()


prompt1 = PromptTemplate( 
    template="Generate notes on topic : {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template = "Generate 5 quiz questions and answers from these notes : {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template = "merge the provided notes and quiz into a single document : \n Notes: {notes} \n Quiz: {quiz}",
    input_variables=["notes", "quiz"],
)
output_parser = StrOutputParser()

model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | output_parser,
    'quiz': prompt2 | model2 | output_parser,}
)

merge_chain = prompt3 | model1 | output_parser 
chain = parallel_chain | merge_chain

text = """ 
    Artificial Intelligence (AI) is transforming education by introducing tools that personalize learning, 
    automate administrative tasks, and enhance accessibility for students. AI-powered platforms analyze learner data to create
    customized learning paths, while intelligent tutoring systems provide real-time guidance similar to human tutors.
    Educators benefit from automation of tasks such as grading and attendance, giving them more time to focus on teaching. 
    AI also helps students with disabilities through features like text-to-speech and translation tools. Despite these advantages,
    challenges such as data privacy concerns, unequal access to technology, overdependence on AI, and high implementation costs
    remain. Overall, AI has significant potential to improve learning outcomes and expand access to quality education, shaping 
    a more advanced and inclusive future for learners worldwide.
"""

result = chain.invoke({"text": text})
print(result)
