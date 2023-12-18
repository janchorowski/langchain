import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from langchain.vectorstores import PathwayVectorClient, PathwayVectorServer
from langchain.text_splitter import CharacterTextSplitter
import time
from multiprocessing import Process

HOST = "127.0.0.1"
PORT = 8784

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "tests/integration_tests/vectorstores/pathway/sample_documents/", format="binary", mode="streaming", with_metadata=True
    )
)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=10,
    chunk_overlap=5,
    length_function=len,
    is_separator_regex=False,
)

embeddings_model = FakeEmbeddings()

vector_server = PathwayVectorServer(
    *data_sources, embedder=embeddings_model, splitter=text_splitter
)

def pathway_server():
    thread = vector_server.run_server(host=HOST, port=PORT, threaded=True, with_cache=False)
    thread.join()

def run_app():
    p = Process(target=pathway_server)
    p.start()
    time.sleep(10)
    return p

def dummy_search():
    query = 'some text'
    results = client.similarity_search(query, k=2)
    return results


client = PathwayVectorClient(
    host=HOST,
    port=PORT,
)

def test_stats():
    process = run_app()
    
    stats_dict = client.get_vectorstore_statistics()
    assert stats_dict is not None
    assert 'last_modified' in stats_dict
    assert 'file_count' in stats_dict

    process.terminate()
    time.sleep(12)
    
def test_similarity_search_results():
    process = run_app()

    results = dummy_search()
    assert results is not None
    assert len(results) == 2

    process.terminate()
    time.sleep(12)

def test_similarity_search_metadata():
    process = run_app()

    results = dummy_search()
    metadata_dict = results[0].metadata
    assert metadata_dict is not None

    process.terminate()
    time.sleep(12)
