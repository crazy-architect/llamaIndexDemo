import warnings

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank

# ignore wanrings
warnings.filterwarnings('ignore')

"""
Description：build index based on the input corpus
[input para] input_dir: the directory of the corpus
[input para] input_extension_list: the extension list of the corpus
[input para] is_recursive: whether to recursively traverse the subdirectories
[input para] persist_dir: the directory to store the index
[input para] need_show_progress: whether to show the progress
[input para] embed_model:  the model to embed the text
[input para] llm: the language model to query
[input para] chunk_size: the chunk size
"""
def build_index(input_dir=None,input_extension_list=None,is_recursive=None,persist_dir=None,need_show_progress=None,embed_model=None,llm=None,chunk_size=None):

    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=is_recursive,
        required_exts=input_extension_list
    )
    documents = reader.load_data(show_progress=need_show_progress)

    Settings.embed_model=embed_model
    Settings.llm=llm
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents)
    try:
        # index=VectorStoreIndex.from_documents(documents,show_progress=need_show_progress)
        index = VectorStoreIndex(nodes,show_progress=need_show_progress)
        # save the index
        index.storage_context.persist(persist_dir=persist_dir)
        print("Index built successfully.")
        return index
    except Exception as e:
        print(e)
        print("Failed to build index.")
        return None

"""
Description：query by index
[input para] persist_dir: the directory to store the index
[input para] query_str: the query string
"""
def query_by_index(persist_dir=None,query_str=None,embed_model=None,llm=None):

    Settings.embed_model=embed_model
    Settings.llm=llm

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    loaded_index = load_index_from_storage(storage_context)

    rerank_llm = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2",top_n=3)
  
    # query the index
    query_engine = loaded_index.as_query_engine(similiarity_top_k=3,node_postprocessor=[rerank_llm])
    query_response = query_engine.query(query_str)

    return query_response.response