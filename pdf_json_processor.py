import json
from typing import Dict, Any
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

class PDFJSONProcessor:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """Initialize the processor with Ollama LLM"""
        self.llm = Ollama(base_url=ollama_base_url, model="deepseek-r1")
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en")
        
    def load_pdf(self, pdf_path: str) -> str:
        """Load and extract text from PDF"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        return " ".join([page.page_content for page in pages])
    
    def load_json(self, json_path: str) -> Dict[str, Any]:
        """Load JSON configuration"""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def create_vector_relation(self, pdf_text: str, json_config: Dict[str, Any]) -> str:
        """Create vector relation between PDF and JSON using LLM"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Create a vector relation between the provided PDF content and JSON configuration.You will receive a JSON configuration and the content of a PDF document. Use this information to create a vector relation.This relation need to be used as a vector for further processing. So all the feaure including key mapping, embeddings, and any other relevant information should be included in the response."),
            ("user", "JSON Config:\n{json_config}\n\nPDF Content:\n{pdf_text}")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({
            "json_config": json.dumps(json_config, indent=2),
            "pdf_text": pdf_text[:2000]  # Limiting text length for prompt
        })
    
    def process_documents(self, pdf_path: str, json_path: str, output_dir: str) -> None:
        """Process PDF and JSON documents and save results"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load documents
        pdf_text = self.load_pdf(pdf_path)
        json_config = self.load_json(json_path)
        
        # Create embeddings for PDF
        pdf_embedding = self.embeddings.embed_query(pdf_text)
        
        # Create vector relation
        vector_relation = self.create_vector_relation(pdf_text, json_config)
        
        # Save results
        output = {
            "pdf_path": pdf_path,
            "json_path": json_path,
            "pdf_embedding": pdf_embedding,
            "json_config": json_config,
            "vector_relation": vector_relation
        }
        
        # Save to files
        base_name = Path(pdf_path).stem
        output_path = Path(output_dir) / f"{base_name}_processed.json"
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Processing complete. Results saved to: {output_path}")

def main():
    # Example usage
    processor = PDFJSONProcessor()
    
    pdf_path = "data/2.pdf"
    json_path = "data/1.json"
    output_dir = "output"
    
    processor.process_documents(pdf_path, json_path, output_dir)

if __name__ == "__main__":
    main()