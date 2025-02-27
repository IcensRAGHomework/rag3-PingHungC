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
    df = pd.read_csv("./COA_OpenData.csv")

    if df.empty:
        print("讀取失敗：DataFrame 為空")
    #else:
    #    print("讀取成功：共有", len(df), "筆資料")


    required_columns = {"Name", "Type", "Address", "Tel", "City", "Town", "CreateDate", "HostWords"}
    if not required_columns.issubset(df.columns):
        print("CSV 缺少必要的欄位")

    df["Date"] = pd.to_datetime(df["CreateDate"], errors="coerce").apply(lambda x: int(x.timestamp()) if pd.notnull(x) else 0)

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

    collection.add(
            ids=[str(i) for i in range(len(df))],
            metadatas=[
                {
                    "file_name": "COA_OpenData.csv",
                    "name": row["Name"],
                    "type": row["Type"],
                    "address": row["Address"],
                    "tel": row["Tel"],
                    "city": row["City"],
                    "town": row["Town"],
                    "date": row["Date"]
                }
                for _, row in df.iterrows()
            ],
            documents=df["HostWords"].fillna("").tolist()
        )
        
    print("資料已成功存入 ChromaDB")
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
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

#generate_hw01()
