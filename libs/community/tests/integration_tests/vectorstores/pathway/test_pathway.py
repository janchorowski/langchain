from langchain.vectorstores import PathwayVectorClient
import subprocess

try:
    subprocess.run(['python', 'app.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

HOST = "127.0.0.1"
PORT = "8780"

client = PathwayVectorClient(
    host=HOST,
    port=PORT,
)

def dummy_search():
    query = 'some text'
    results = client.similarity_search(query, k=2)
    return results

def test_stats():
    stats_dict = client.get_vectorstore_statistics()
    assert stats_dict is not None
    assert 'last_modified' in stats_dict
    assert 'file_count' in stats_dict

def test_similarity_search_results():
    results = dummy_search()
    assert results is not None
    assert len(results) == 2

def test_similarity_search_metadata():
    results = dummy_search()
    metadata_dict = results[0].metadata
    assert metadata_dict is not None
    