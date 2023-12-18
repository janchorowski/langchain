from langchain.vectorstores import PathwayVectorServer

HOST = "127.0.0.1"
PORT = "8780"

# Example for document loading (from local folders), splitting, and creating vectorstore with Pathway
import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "sample_documents/", format="binary", mode="streaming", with_metadata=True
    )  # This creates a `pathway` connector that tracks all the files in the sample_documents directory
)

# Split
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=10,
    chunk_overlap=5,
    length_function=len,
    is_separator_regex=False,
)

# Embed
embeddings_model = OpenAIEmbeddings()

# Launch VectorDB
vector_server = PathwayVectorServer(
    *data_sources, embedder=embeddings_model, splitter=text_splitter
)
vector_server.run_server(host=HOST, port=PORT, threaded=True)