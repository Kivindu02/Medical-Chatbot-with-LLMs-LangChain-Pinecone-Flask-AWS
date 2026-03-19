from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings


#Extract text from PDF files
def load_pdf_files(data):
  loader = DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader
                           )
  
  documents = loader.load()
  return documents




def filter_to_minimal_docs(docs: List[Document] ) -> List[Document] :
  """
  Given a list od document objects, return a new list of document objects containing only 'source' in metadata and the orginal page_content.
  """
  minimal_docs: List[Document] = []
  for doc in docs:
    src = doc.metadata.get("source")
    minimal_docs.append(
      Document(
        page_content=doc.page_content,
        metadata={"source": src}
      )
    )
  return minimal_docs



#Split the document into smaller chunks
def text_split(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
  )
  texts = text_splitter.split_documents(extracted_data)
  return texts


#Download the Embeddings from HuggingFace
def download_embeddings():
  """
  Download  and return the HuggingFace embedding model.
  """
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name
  )
  return embeddings

