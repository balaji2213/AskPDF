# AskPDF
''' "AskPDF"  is an intelligent question-answering system designed to extract relevant information from PDF documents efficiently. The project leverages advanced natural language processing tools like Haystack, BM25Retriever, and the FARMReader to analyze document content and provide precise answers to user queries.'''
import os
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
import PyPDF2
import torch

# Step 1: Extracting text from PDF
def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF and associates it with page numbers.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dictionaries with text content and metadata for each page.
    """
    documents = []
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            if not reader.pages:
                print("The PDF contains no pages or is corrupted.")
                return []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():  # Ignore empty or whitespace-only pages
                        documents.append({"content": text, "meta": {"page": page_num}})
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {e}")
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF: {e}")
    except FileNotFoundError:
        print(f"File not found: {pdf_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return documents

# Step 2: Setting up Haystack pipeline
def setup_pipeline(documents):
    """
    Sets up the Haystack pipeline with a BM25 retriever and FARM reader.

    Args:
        documents (list): List of documents with content and metadata.

    Returns:
        ExtractiveQAPipeline: Configured pipeline for question answering.
    """
    # Initialize document store
    document_store = InMemoryDocumentStore(use_bm25=True)

    # Write documents to the store
    document_store.write_documents(documents)

    # Initialize retriever and reader
    retriever = BM25Retriever(document_store=document_store)

    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=use_gpu)

    # Set up pipeline
    pipeline = ExtractiveQAPipeline(reader, retriever)
    return pipeline

# Step 3: Querying the pipeline
def query_pipeline(pipeline, question):
    """
    Queries the pipeline with a question and retrieves the answer.

    Args:
        pipeline (ExtractiveQAPipeline): The QA pipeline.
        question (str): The question to ask.

    Returns:
        str: The best answer with metadata.
    """
    try:
        prediction = pipeline.run(
            query=question,
            params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}},
        )
        if prediction["answers"]:
            best_answer = prediction["answers"][0]
            return (
                f"Answer: {best_answer.answer}\n"
                f"Page: {best_answer.meta.get('page', 'Unknown')}\n"
                f"Confidence: {best_answer.score:.2f}"
            )
        else:
            return "No answer found."
    except Exception as e:
        return f"Error during query: {e}"

# Example usage
if __name__ == "__main__":
    # Path to your PDF
    pdf_path = "/content/The Unseen Gift.pdf"

    # Question to ask
    question = "Why did the traveler give Ravi a stone?"

    # Extract text from PDF
    print("Extracting text from the PDF...")
    documents = extract_pdf_text(pdf_path)

    if not documents:
        print("No content extracted from the PDF. Exiting.")
    else:
        # Set up the pipeline
        print("Setting up the QA pipeline...")
        pipeline = setup_pipeline(documents)

        # Query the pipeline
        print("Querying the pipeline...")
        answer = query_pipeline(pipeline, question)
        print("\nResult:")
        print(answer)

