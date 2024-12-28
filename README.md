# LangChain-Function-Tool-Calling-Calculator
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
# Install required packages
!pip install langchain langchain-google-genai google-generativeai python-dotenv


# Initialize API key
GOOGLE_GEMINI_API_KEY = 'AIzaSyBy7C2FWsiLqWrpCAwqQODgol9ZhnzwK1c' # Replace with your actual API key
if not GOOGLE_GEMINI_API_KEY:
    GOOGLE_GEMINI_API_KEY = input("Please enter your Google Gemini API key: ")

# Configure Gemini
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

class Calculator:
    """Calculator class for performing arithmetic operations"""

    def __init__(self): # Corrected: __init__ instead of _init_
        self.allowed_operators = {'+', '-', '*', '/', '(', ')', '.'}

    def is_safe_expression(self, expression: str) -> bool:
        """Validate if expression contains only allowed characters"""
        cleaned = ''.join(expression.split())
        return all(c.isdigit() or c in self.allowed_operators for c in cleaned)

    def calculate(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            if not self.is_safe_expression(expression):
                return "Error: Invalid characters in expression"

            # Create a restricted environment for eval
            # Removed the now unnecessary restricted_globals
            result = eval(expression) # Directly use eval

            # Format the result nicely
            if isinstance(result, float):
                return f"{result:.6f}".rstrip('0').rstrip('.')
            return str(result)

        except Exception as e:
            return f"Error: {str(e)}"

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_GEMINI_API_KEY,
    temperature=0
)

# Create calculator instance
calculator_instance = Calculator()

# Define the calculator tool
def calculator_tool(expression: str) -> str:
    """Perform mathematical calculations"""
    return calculator_instance.calculate(expression)

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for performing mathematical calculations. Input should be a mathematical expression (e.g., '2 + 2', '10 * 5')"
    )
]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with ZERO_SHOT_REACT_DESCRIPTION
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

def process_calculation(expression: str) -> str:
    """Process calculation directly if it's a simple expression"""
    if calculator_instance.is_safe_expression(expression):
        return f"The result of {expression} is {calculator_instance.calculate(expression)}"
    return agent.run(expression)

def chat():
    print("Calculator Assistant: Hello! I can help you with calculations. What would you like to calculate?")
    print('(Type "exit" to end the conversation)\n')

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nCalculator Assistant: Goodbye! Have a great day!")
            break

        if not user_input:
            print("\nCalculator Assistant: Please enter a calculation or question.\n")
            continue

        try:
            response = process_calculation(user_input)
            print(f"\nCalculator Assistant: {response}\n")
        except Exception as e:
            print(f"\nCalculator Assistant: I encountered an error: {str(e)}\nPlease try again with a different calculation.\n")

if __name__ == "__main__": #Corrected: __name__ instead of _name_
    chat()
