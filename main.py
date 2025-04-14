"""
T-Shirt Inventory Q&A Chatbot powered by Google Gemini

This script implements a chatbot that answers questions about t-shirt inventory
using natural language processing and SQL queries. It uses Google's Generative AI
model to generate SQL queries and natural language responses.

Requirements:
- Python 3.7+
- Required packages: langchain, langchain_google_genai, langchain_community,
  python-dotenv, tenacity, pymysql

Usage:
1. Set up a .env file with the required environment variables.
2. Run the script and interact with the chatbot via the command line.
"""

from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration from environment variables
GOOGLE_API_KEY = "AIzaSyDTAXKiV33j1Y1b3QuHg-FFWsBYPRVjD2A"
db_user = "root"
db_password = "root"
db_host = "localhost:3307"
db_name = "atliq_tshirts"

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
# Initialize Google Generative AI model
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=GOOGLE_API_KEY)

# Custom prompt template
prompt_template = """
Given an input question, create a syntactically correct MySQL query to run. 
Do not include any markdown formatting or backticks in your response. 
Only provide the raw SQL query.

Here is the question:
{input}
"""

class SQLOutputParser(BaseOutputParser):
    """Custom output parser for SQL queries."""
    def parse(self, text):
        """Parse the output text."""
        return text

# Create SQL query chain
sql_chain = create_sql_query_chain(llm, db)

def clean_sql_query(query):
    """
    Clean the SQL query by removing markdown formatting and unnecessary characters.
    
    Args:
    query (str): The SQL query to clean.
    
    Returns:
    str: The cleaned SQL query.
    """
    cleaned_query = query.strip()
    if cleaned_query.startswith('```sql'):
        cleaned_query = cleaned_query[6:]
    if cleaned_query.endswith('```'):
        cleaned_query = cleaned_query[:-3]
    return cleaned_query.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_sql_query(question):
    """
    Generate an SQL query based on the given question.
    
    Args:
    question (str): The input question.
    
    Returns:
    str: The generated SQL query.
    """
    return sql_chain.invoke({"question": question})

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_response(query, result, question):
    """
    Generate a natural language response based on the SQL query and its result.
    
    Args:
    query (str): The SQL query.
    result (str): The result of the SQL query.
    question (str): The original input question.
    
    Returns:
    str: The generated natural language response.
    """
    response_prompt = PromptTemplate.from_template(
        "Based on the SQL query: {query}\n"
        "And its result: {result}\n"
        "Please provide a natural language answer to the question: {question}"
    )
    response_chain = response_prompt | llm | SQLOutputParser()
    return response_chain.invoke({"query": query, "result": result, "question": question})

def get_response(question):
    """
    Process the user's question and return a response.
    
    Args:
    question (str): The user's input question.
    
    Returns:
    str: The chatbot's response.
    """
    try:
        # Generate SQL query with retry
        sql_query = generate_sql_query(question)
        
        # Clean the SQL query
        cleaned_sql_query = clean_sql_query(sql_query)
        
        # Execute the cleaned query
        result = db.run(cleaned_sql_query)
        
        # Generate response based on the query result with retry
        response = generate_response(cleaned_sql_query, result, question)
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    """Main function to run the chatbot."""
    print("Welcome to the Ultimate T-Shirt Inventory Assistant, proudly powered by /'GEN ARCH/' – your smart solution for seamless inventory management!")
    print("Ask questions about t-shirts and discounts. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Thank you for using the T-Shirt Inventory Q&A Chatbot. Goodbye!")
            break
        
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()