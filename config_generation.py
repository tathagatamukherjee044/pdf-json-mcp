import json
from typing import Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery
import os
import instructor
from pydantic import BaseModel, Field
from anthropic import Anthropic
from models import ExtractionConfig

load_dotenv()


class PDFSimilarityProcessor:
    def __init__(self):
        """Initialize the processor with required components"""
        self.llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.7,
        )
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en")
        self.rag_collection_name = os.getenv("RAG_COLLECTION_NAME", "rag_prompt_db")

        # Initialize Weaviate client
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
            additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60)),
        )

    def load_and_embed_pdf(self, pdf_path: str) -> Tuple[str, list]:
        """Load PDF and create its embedding"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        pdf_text = " ".join([page.page_content for page in pages])
        embedding = self.embeddings.embed_query(pdf_text)
        return pdf_text, embedding

    def find_similar_document(self, pdf_embedding: list) -> Dict[str, Any]:
        """Search Weaviate for similar document using embedding"""
        collection = self.client.collections.get(self.rag_collection_name)

        response = collection.query.near_vector(
            near_vector=pdf_embedding,  # your query vector goes here
            return_metadata=MetadataQuery(distance=True),
            limit=1,
            return_properties=["rag_prompt", "json_config", "source"],
        )

        print(str(response))

        if response.objects:
            return response.objects[0].properties
        return None

    def generate_extraction_config(
        self, pdf_text: str, similar_doc: Dict[str, Any]
    ) -> ExtractionConfig:
        """Generate extraction configuration using LLM with structured output"""
        # Initialize instructor client
        client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
            mode=instructor.Mode.ANTHROPIC_TOOLS
        )

        # Create the prompt
        prompt = f"""You are an expert at analyzing documents and creating extraction configurations.
            Based on the example PDF, RAG prompt, and JSON configuration, create a new extraction configuration
            that can be used to extract similar information from the new PDF.
            Focus on maintaining the same structure while adapting to any specific differences.

        Example RAG Prompt: {similar_doc['rag_prompt']}
        Example JSON Config: {similar_doc['json_config']}
        New PDF Content: {pdf_text[:4000]}
        
            Create a new extraction configuration that follows the same pattern but is adapted for the new PDF. And retunn the configuration as a JSON string. Dont return any other text.""",

        # Get structured response
        response = client.chat.completions.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": prompt}],
            response_model=ExtractionConfig,
            max_tokens=4096,
            temperature=0.3
        )

        # Return the response directly (it's already the Pydantic model)
        return response

    def process_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """Process PDF and generate extraction configuration"""
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Load and embed PDF
            pdf_text, pdf_embedding = self.load_and_embed_pdf(pdf_path)

            # Find similar document
            similar_doc = self.find_similar_document(pdf_embedding)
            if not similar_doc:
                raise ValueError("No similar documents found in the database")

            # Generate new configuration
            new_config = self.generate_extraction_config(pdf_text, similar_doc)
            print(new_config)

            # Convert to dictionary
            config_dict = new_config.model_dump()

            # Save results to file (optional, for debugging)
            output = {
                # "pdf_path": pdf_path,
                # "similar_document_source": similar_doc["source"],
                # "example_rag_prompt": similar_doc["rag_prompt"],
                # "example_json_config": similar_doc["json_config"],
                "new_config": config_dict,
            }

            # Save to file
            base_name = Path(pdf_path).stem
            output_path = Path(output_dir) / f"{base_name}_extraction_config.json"

            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            print(f"Processing complete. Results saved to: {output_path}")
            
            # Return the config dictionary
            return config_dict

        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
        finally:
            self.client.close()

    def process_pdf_and_return_config(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF and return only the extraction configuration without saving to file"""
        try:
            # Load and embed PDF
            pdf_text, pdf_embedding = self.load_and_embed_pdf(pdf_path)

            # Find similar document
            similar_doc = self.find_similar_document(pdf_embedding)
            if not similar_doc:
                raise ValueError("No similar documents found in the database")

            # Generate new configuration
            new_config = self.generate_extraction_config(pdf_text, similar_doc)
            
            # Convert to dictionary and add debugging
            config_dict = new_config.model_dump()
            print(f"Generated config type: {type(config_dict)}")
            print(f"Config keys: {list(config_dict.keys()) if isinstance(config_dict, dict) else 'Not a dict'}")
            
            # Return the config dictionary
            return config_dict

        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
        finally:
            self.client.close()


def main():
    processor = PDFSimilarityProcessor()

    pdf_path = "data/TEST.pdf"
    output_dir = "output"

    processor.process_pdf(pdf_path, output_dir)


if __name__ == "__main__":
    main()
