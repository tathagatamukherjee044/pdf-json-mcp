import os
import json
from typing import Dict, Any

from langchain.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define the Ollama LLM
ollama_llm = Ollama(base_url='http://localhost:11434', model="llama3")

# Agent 1: PDF Table Extractor
def extract_table_count_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF and extracts the number of tables.
    """
    try:
        loader = PyPDFLoader(file_path=pdf_path)
        pages = loader.load_and_split()
        text = "".join([page.page_content for page in pages])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert PDF data extractor.  How many tables are present in the following document?"),
            ("user", "{input}")
        ])

        chain = prompt | ollama_llm

        result = chain.invoke({"input": text})
        return result
    except Exception as e:
        return f"Error extracting table count: {e}"

# Agent 2: JSON Comparison Agent
def compare_json_to_pdf_structure(json_data: str, pdf_structure_description: str) -> str:
    """
    Compares a JSON data structure to a description of the PDF structure.
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at comparing JSON data to PDF structure descriptions.  Does the following JSON match the PDF structure?"),
            ("user", "JSON:\n{json_data}\n\nPDF Structure Description:\n{pdf_structure_description}")
        ])

        chain = prompt | ollama_llm

        result = chain.invoke({"json_data": json_data, "pdf_structure_description": pdf_structure_description})
        return result
    except Exception as e:
        return f"Error comparing JSON to PDF structure: {e}"

# Agent 3: RAG-based PDF Structure Checker
def rag_check_pdf_structure(query: str, vectorstore: Chroma) -> str:
    """
    Uses a RAG system to check if a given query about PDF structure is found in the vectorstore.
    """
    try:
        retrieval_chain = create_retrieval_chain(
            vectorstore.as_retriever(),
            (PromptTemplate.from_template("Answer the query based on the context: {context}") | ollama_llm),
        )
        result = retrieval_chain.invoke(query)
        return result["answer"]
    except Exception as e:
        return f"Error during RAG check: {e}"

# Setup tools
tools = [
    Tool(
        name="pdf_table_extractor",
        func=extract_table_count_from_pdf,
        description="Useful for when you need to extract the number of tables from a PDF document. Input should be a valid file path to the PDF.",
    ),
    Tool(
        name="json_pdf_comparator",
        func=compare_json_to_pdf_structure,
        description="Useful for when you need to compare a JSON data structure to a description of a PDF's structure. Input should be a json string and a text description of the pdf structure.",
    ),
    Tool(
        name="rag_pdf_structure_checker",
        func=rag_check_pdf_structure,
        description="Useful for when you need to check the structure of a PDF against a knowledge base.  The input should be a query about the PDF structure.",
    ),
]

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful agent that can analyze PDFs, compare JSON data, and use a RAG system to verify information."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Memory
memory = ConversationSummaryBufferMemory(
    llm=ollama_llm,
    max_token_limit=1024,
    memory_key="chat_history",
    return_messages=True,
)

# Create agent
agent = create_react_agent(
    llm=ollama_llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)

# Example Usage (You'll need to create a vectorstore and have a PDF/JSON ready)
if __name__ == '__main__':
    # Dummy data and setup for demonstration
    pdf_file_path = "example.pdf"  # Replace with a real PDF file
    json_data = '{"table1": {"columns": ["A", "B"]}, "table2": {"columns": ["C", "D"]}}'  # Example JSON
    pdf_structure_description = "The PDF contains two tables. Table 1 has columns A and B, and Table 2 has columns C and D."

    # Create a dummy PDF (replace with actual PDF creation if needed)
    with open(pdf_file_path, "w") as f:
        f.write("This is a dummy PDF file for testing.")

    # Create a dummy vectorstore (replace with actual vectorstore creation)
    documents = [pdf_structure_description]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.create_documents(documents)
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings)

    # Example usage of the agent
    response = agent_executor.invoke({"input": f"How many tables are in {pdf_file_path}?"})
    print(f"Agent Response (Table Count): {response['output']}")

    response = agent_executor.invoke({"input": f"Does the following JSON match the PDF structure: {json_data}?"})
    print(f"Agent Response (JSON Comparison): {response['output']}")

    response = agent_executor.invoke({"input": "Check if the PDF contains two tables with specified columns."})
    print(f"Agent Response (RAG Check): {response['output']}")

    # Clean up dummy PDF
    os.remove(pdf_file_path)