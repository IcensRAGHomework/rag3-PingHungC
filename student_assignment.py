import datetime
import chromadb
import traceback
import pandas as pd

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() == 0:
        df = pd.read_csv("./COA_OpenData.csv")

        if df.empty:
            print("讀取失敗：DataFrame 為空")

        required_columns = {"Name", "Type", "Address", "Tel", "City", "Town", "CreateDate", "HostWords"}
        if not required_columns.issubset(df.columns):
            print("CSV 缺少必要的欄位")

        df["Date"] = pd.to_datetime(df["CreateDate"], errors="coerce").apply(lambda x: int(x.timestamp()) if pd.notnull(x) else 0)

        for i, row in df.iterrows():
            metadata = {
                "file_name": "COA_OpenData.csv",
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": row["Date"]
            }
            collection.add(
                ids=[str(i)],
                metadatas=[metadata],
                documents=[row["HostWords"]]
            )

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):

    collection = generate_hw01()

    results = collection.query(
        query_texts=[question],
        n_results=150
    )

    filtered_results = []
    for i, score in enumerate(results["distances"][0]):
        metadata = results["metadatas"][0][i]
        similarity = 1 - score
        if similarity < 0.80:
            continue
        if city and metadata["city"] not in city:
            continue
        if store_type and metadata["type"] not in store_type:
                continue
        if start_date and metadata["date"] < int(start_date.timestamp()):
            continue
        if end_date and metadata["date"] > int(end_date.timestamp()):
            continue
        filtered_results.append((metadata["name"], similarity))
    
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in filtered_results]
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()

    results = collection.query(
        query_texts=[store_name], 
        n_results=1
    )

    if results["metadatas"][0]:
        store_metadata = results["metadatas"][0][0]
        store_metadata["name"] = new_store_name
        collection.update(ids=[results["ids"][0][0]], metadatas=[store_metadata])

    results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )

    filtered_results = []
    for i, score in enumerate(results["distances"][0]):
        metadata = results["metadatas"][0][i]
        similarity = 1 - score
        if similarity < 0.80:
            continue

        filtered_results.append((metadata["name"], similarity))
    
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in filtered_results]
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

