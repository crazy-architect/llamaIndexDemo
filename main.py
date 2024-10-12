from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llamaindex_rag_util import build_index, query_by_index

def main():
    # set the directory to build the index
    input_dir="./data"
    input_extension_list=[".txt"]
    is_recursive=True
    # set the directory to store the index
    persist_dir="./rag_index"
    need_show_progress=True
    # select embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # select llm model
    llm = Ollama(model="gemma2:2b", request_timeout=600.0)
    chunk_size=1000
    build_index(input_dir, input_extension_list, is_recursive, persist_dir, need_show_progress,embed_model,llm,chunk_size)

    # query the index
    query_str="Please talk about your programming experience."
    print("The question to RAG is:"+query_str+"\n")
    result =query_by_index(persist_dir, query_str,embed_model,llm)
    print("The answer is:"+result)

if __name__ == '__main__':
    main()