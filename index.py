from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_groq import ChatGroq



# Load environment variables
load_dotenv()

# Streamlit app config
st.set_page_config(page_title="ClarifyAI", page_icon="🤖")
st.title("ClarifyAI")

# Create sidebar
st.sidebar.title("Prompt Enhancer")

# Write some text about the prompt enhancer
st.sidebar.write(
    "Welcome to Prompt Enhancer! This tool helps you craft effective prompts that are specific, clear, and actionable.\n\n"
    "### Key Features:\n"
    "- **Analysis**: Identifies ambiguity in your prompts.\n"
    "- **Suggestions**: Provides tailored enhancements for clarity and context.\n"
    "- **User-Friendly**: Easy input with immediate improvements.\n"
    "- **Learning Resources**: Tips and best practices to boost your skills.\n"
    "\nStart creating better prompts today!"
)




# Retrieve API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROk_API_KEY=os.getenv('GROk_API_KEY')
# Instructions template for enhancing the prompt
instructions = """
Simulate Persona: Imagine you are a mentor guiding a group of individuals new to prompt engineering.
Task: Identify and address common issues in prompt creation to improve overall effectiveness.
Steps to complete Task:
1. Analyze the prompt for Persona, Task, Steps, Context/Constraints, Goals, and Output Format.
2. Identify and rectify instances of ambiguity or vagueness in prompt language.
3. Ensure that prompts provide sufficient information for the desired task without unnecessary complexity.
Context/Constraints: Consider the diverse backgrounds and skill levels of those engaging with the prompts.
Goal: Enhance the quality of prompts by eliminating potential sources of confusion or misinterpretation.
Format Output: Provide a revised version of a poorly constructed prompt, highlighting the changes made and explaining how each modification contributes to better prompt engineering.

Example:
Original Prompt: "Explain AI."
Revised Prompt: "Explain how artificial intelligence is applied in healthcare, particularly in diagnostic tools and patient data analysis."
Changes Made: 
- Added specific domain ("healthcare") to focus the explanation.
- Provided a clear task ("applied in diagnostic tools and patient data analysis") for more targeted responses.
- Avoided unnecessary complexity while ensuring clarity and actionable direction.
"""
# Function to get the response based on user prompt
def get_response(user_prompt):
    # Create an instance of GoogleGenerativeAI
    # llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)
    llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.3,
    api_key=GROk_API_KEY,
    streaming=True
)

    # Create a prompt template with input variables
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', instructions),
            ('user', "question: {question}")
        ]
    )

    # Create a sequence to run the prompt and LLM
    chain = RunnableSequence(
        prompt_template,
        llm,
    )

    # Execute the sequence with the user prompt
    return chain.stream({"question": user_prompt})

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.3,
    api_key=GROk_API_KEY,
    streaming=True
)

#set session state
if 'Chat_history' not in st.session_state:
    st.session_state.Chat_history = [
AIMessage(content="Hi there! I'm your prompt enhancement assistant. How can I assist you in crafting better prompts today?")
    ]


# conversation history

# conversation
for message in st.session_state.Chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
#user input

user_input = st.chat_input("You:")
if user_input is not None and user_input != "":
    st.session_state.Chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("Human"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        response=st.write_stream(get_response(user_input))
    st.session_state.Chat_history.append(AIMessage(content=response))    
