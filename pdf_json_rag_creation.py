import json
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import weaviate
from weaviate.classes.init import Auth
from datetime import datetime, timezone
import weaviate.classes as wvc
from weaviate.classes.init import AdditionalConfig, Timeout
from uuid import uuid4
import os
import pdf2image
import numpy as np
from PIL import Image
import io

# Load environment variables
load_dotenv()


class PDFJSONProcessor:
    def __init__(self):
        """Initialize the processor with Anthropic's Claude"""
        self.llm = ChatAnthropic(
            model="claude-3-opus-20240229",  # or "claude-3-sonnet-20240229" for faster, cheaper option
            max_tokens=4096,
            temperature=0.7,
        )
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en")

        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.rag_collection_name = os.getenv("RAG_COLLECTION_NAME", "rag_prompt_db")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
            ),
        )

        self._create_schema()

    def load_pdf(self, pdf_path: str) -> str:
        """Load and extract text from PDF"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        return " ".join([page.page_content for page in pages])

    def load_json(self, json_path: str) -> Dict[str, Any]:
        """Load JSON configuration"""
        with open(json_path, "r") as f:
            return json.load(f)

    def rag_prompt_genration(self, pdf_text: str, json_config: Dict[str, Any]) -> str:
        """Create vector relation between PDF and JSON using Claude"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at analyzing documents and creating RAG prompts. 
            Create a relationship between the PDF and JSON file that depicts the structure of the PDF.
            Generate a RAG-optimized prompt that can be used for vector storage and retrieval without needing the original files.
            Focus on key structural relationships and semantic mappings.""",
                ),
                ("user", "JSON Config:\n{json_config}\n\nPDF Content:\n{pdf_text}"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke(
            {
                "json_config": json.dumps(json_config, indent=2),
                "pdf_text": pdf_text[:4000],  # Claude can handle longer contexts
            }
        )
        return str(response.content.strip())

    def _create_schema(self):
        """Create Weaviate schema if it doesn't exist"""
        # Check if collection exists
        if not self.client.collections.exists(self.rag_collection_name):
            try:
                # Create collection with schema
                collection = self.client.collections.create(
                    name=self.rag_collection_name,
                    description="Schema for a vector database",
                    generative_config=wvc.config.Configure.Generative.anthropic(),
                    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_cohere(
                        model="embed-multilingual-v2.0", vectorize_collection_name=True
                    ),
                    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                        distance_metric=wvc.config.VectorDistances.COSINE
                    ),
                    properties=[
                        wvc.config.Property(
                            name="json_config",
                            data_type=wvc.config.DataType.TEXT,  # Changed from 'object' to 'OBJECT' [1]
                            description="Json configuration of the vector",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                        ),
                        wvc.config.Property(
                            name="created_at",
                            data_type=wvc.config.DataType.DATE,  # Changed from 'date' to 'DATE' [1]
                            description="Timestamp when the vector was created",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                        ),
                        wvc.config.Property(
                            name="updated_at",
                            data_type=wvc.config.DataType.DATE,  # Changed from 'date' to 'DATE' [1]
                            description="Timestamp when the vector was last updated",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                        ),
                        wvc.config.Property(
                            name="source",
                            data_type=wvc.config.DataType.TEXT,  # Changed from 'string' to 'TEXT' [1]
                            description="Source of the vector data",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                            index_searchable=True,  # Changed from 'indexSearchable' to 'index_searchable' [2]
                        ),
                        wvc.config.Property(
                            name="tags",
                            data_type=wvc.config.DataType.TEXT_ARRAY,  # Changed from 'string' to 'TEXT_ARRAY' [1]
                            description="Tags for categorization",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                            index_searchable=True,  # Changed from 'indexSearchable' to 'index_searchable' [2]
                        ),
                        wvc.config.Property(
                            name="rag_prompt",
                            data_type=wvc.config.DataType.TEXT,  # Changed from 'text' to 'TEXT' [1]
                            description="RAG prompt for querying the vector database",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                            index_searchable=True,  # Changed from 'indexSearchable' to 'index_searchable' [2]
                        ),
                    ],
                )
                print(f"Created collection: {self.rag_collection_name}")
            except Exception as e:
                print(f"Error creating collection: {e}")
                raise
        else:
            print(f"Collection {self.rag_collection_name} already exists")
        """Create Weaviate schema if it doesn't exist"""
        # Check if collection exists

    def save_to_weaviate(self, output: Dict[str, Any]) -> str:
        """Save processed output to Weaviate"""
        try:
            # Get or create collection

            # Check if collection exists, if not create it
            if not self.client.collections.exists(self.rag_collection_name):
                collection = self.client.collections.create(
                    name=self.rag_collection_name,
                    description="Schema for a vector database",
                    generative_config=wvc.config.Configure.Generative.anthropic(),
                    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_cohere(
                        model="embed-multilingual-v2.0", vectorize_collection_name=True
                    ),
                    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                        distance_metric=wvc.config.VectorDistances.COSINE
                    ),
                    properties=[
                        wvc.config.Property(
                            name="json_config",
                            data_type=wvc.config.DataType.TEXT,  # Changed from 'object' to 'OBJECT' [1]
                            description="Json configuration of the vector",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                        ),
                        wvc.config.Property(
                            name="created_at",
                            data_type=wvc.config.DataType.DATE,  # Changed from 'date' to 'DATE' [1]
                            description="Timestamp when the vector was created",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                        ),
                        wvc.config.Property(
                            name="updated_at",
                            data_type=wvc.config.DataType.DATE,  # Changed from 'date' to 'DATE' [1]
                            description="Timestamp when the vector was last updated",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                        ),
                        wvc.config.Property(
                            name="source",
                            data_type=wvc.config.DataType.TEXT,  # Changed from 'string' to 'TEXT' [1]
                            description="Source of the vector data",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                            index_searchable=True,  # Changed from 'indexSearchable' to 'index_searchable' [2]
                        ),
                        wvc.config.Property(
                            name="tags",
                            data_type=wvc.config.DataType.TEXT_ARRAY,  # Changed from 'string' to 'TEXT_ARRAY' [1]
                            description="Tags for categorization",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                            index_searchable=True,  # Changed from 'indexSearchable' to 'index_searchable' [2]
                        ),
                        wvc.config.Property(
                            name="rag_prompt",
                            data_type=wvc.config.DataType.TEXT,  # Changed from 'text' to 'TEXT' [1]
                            description="RAG prompt for querying the vector database",
                            index_filterable=True,  # Changed from 'indexFilterable' to 'index_filterable' [2]
                            index_searchable=True,  # Changed from 'indexSearchable' to 'index_searchable' [2]
                        ),
                    ],
                )
            else:
                collection = self.client.collections.get(self.rag_collection_name)

            # Convert numpy array to list if necessary
            pdf_embedding = (
                output["pdf_embedding"].tolist()
                if hasattr(output["pdf_embedding"], "tolist")
                else output["pdf_embedding"]
            )

            # Prepare the data object
            vector_obj = {
                "json_config": str(output["json_config"]),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "source": f"pdf:{output['pdf_path']}",
                "tags": ["pdf", "json_config"],
                "rag_prompt": output["rag_prompt"],
            }

            # Generate a UUID for the object
            vector_id = str(uuid4())

            # Insert object with vector
            collection.data.insert(
                properties=vector_obj, uuid=vector_id, vector=pdf_embedding
            )

            return vector_id

        except Exception as e:
            print(f"Error saving to Weaviate: {e}")
            raise
        finally:
            # Don't close the client here as it might be needed for subsequent operations
            self.client.close()  # Free up resources, but keep the client open for further operations
            pass

    def process_documents(self, pdf_path: str, json_path: str, output_dir: str) -> None:
        """Process PDF and JSON documents and save results"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load JSON config
        json_config = self.load_json(json_path)

        # Convert PDF to images
        try:
            # Convert PDF to images (returns list of PIL Image objects)
            images = pdf2image.convert_from_path(pdf_path)
            
            # Combine all images into one
            if len(images) > 1:
                # Create a new image with combined height
                total_height = sum(img.height for img in images)
                max_width = max(img.width for img in images)
                combined_image = Image.new('RGB', (max_width, total_height))
                
                # Paste all images
                y_offset = 0
                for img in images:
                    combined_image.paste(img, (0, y_offset))
                    y_offset += img.height
            else:
                combined_image = images[0]

            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            combined_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create embeddings for the image
            pdf_embedding = self.embeddings.embed_query(str(img_byte_arr))

        except Exception as e:
            print(f"Error converting PDF to image: {e}")
            # Fallback to text extraction if image conversion fails
            pdf_text = self.load_pdf(pdf_path)
            pdf_embedding = self.embeddings.embed_query(pdf_text)

        # Create vector relation using the PDF text for RAG prompt
        pdf_text = self.load_pdf(pdf_path)
        rag_prompt = self.rag_prompt_genration(pdf_text, json_config)

        # Save results
        output = {
            "pdf_path": pdf_path,
            "json_path": json_path,
            "pdf_embedding": pdf_embedding,
            "json_config": json_config,
            "rag_prompt": rag_prompt,
        }

        # Save to files
        base_name = Path(pdf_path).stem
        output_path = Path(output_dir) / f"{base_name}_processed.json"

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Processing complete. Results saved to: {output_path}")

        vector_id = self.save_to_weaviate(output)
        print(f"Data saved to Weaviate with ID: {vector_id}")


def main():
    # Example usage
    processor = PDFJSONProcessor()

    pdf_path = "data/2.pdf"
    json_path = "data/2.json"
    output_dir = "output"

    processor.process_documents(pdf_path, json_path, output_dir)


if __name__ == "__main__":
    main()
