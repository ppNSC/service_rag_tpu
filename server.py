import os
from pydantic import BaseModel
import re
import shutil
import time

import faiss
from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse
import nltk
import numpy as np
import pickle
from tqdm import tqdm
from uuid import uuid4

current_dir = os.path.dirname(os.path.abspath(__file__))
nltk_data_path = os.path.join(current_dir, './nltk_data/')
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

from doc_processor.knowledge_file import KnowledgeFile
from embedding import Word2VecEmbedding
from reranker import RerankerTPU


app = FastAPI()

class DelKnowledgeRequest(BaseModel):
    messageType: str
    id: str
    fileName: str

class SelectKnowledgeRequest(BaseModel):
    messageType: str
    id: str
    fileName: str

class PromptRetrievalRequest(BaseModel):
    messageType: str
    snippet_num: int
    prompt: str

class RAG_SERVER:
    _instance = None
    _initialized = False
    UPLOAD_PATH = './knowledge_base/uploaded_file'
    DATABASE_PATH =  './knowledge_base/vector_database'
    FILE_RECORDS_LAST_USED_DB = './knowledge_base/cur_knowledge_used.txt'
    SUPPORTED_EXT = ["pdf", "txt", "docx", "pptx", 'png', 'jpg', 'jpeg', 'bmp']
    def __init__(self) -> None:
        if not RAG_SERVER._initialized:
            self.cur_file_name = None
            self.cur_file_unique_id = None
            self.cur_vector_db = None
            self.cur_string_db = None
            self.model_status = 0
            if os.path.isfile(RAG_SERVER.FILE_RECORDS_LAST_USED_DB):
                with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'r') as record_id:
                    last_used_id = record_id.read().strip()
                    exist_knowledge_database = RAG_SERVER.get_vector_database_map()
                    if last_used_id in exist_knowledge_database.keys():
                        self.select_vector_database_to_use(last_used_id, exist_knowledge_database[last_used_id])
            try:
                self.embedding_machine = Word2VecEmbedding()
                self.reranker = RerankerTPU()
            except:
                self.model_status = -1
                return
            self.model_status = 1
            RAG_SERVER._initialized = True

    def __new__(cls):
        # 单例
        if cls._instance is None:
            cls._instance = super(RAG_SERVER, cls).__new__(cls)
        return cls._instance

    def docs2embedding(self, docs):
        emb = []
        for i in tqdm(range(len(docs) // 4)):
            emb += self.embedding_machine.embed_documents(docs[i * 4: i * 4 + 4])
        if len(docs) % 4 != 0:
            residue = docs[-(len(docs) % 4):] + [" " for _ in range(4 - len(docs) % 4)]
            emb += self.embedding_machine.embed_documents(residue)[:len(docs) % 4]
        return emb

    def retrieval_from_vector_db(self, query: str):
        # current batch size of reranker model if 3.
        k = 3
        query_embedding = self.embedding_machine.embed_query(query)
        _, i = self.cur_vector_db.search(x = np.array([query_embedding]), k = k)
        return [self.cur_string_db[ind] for ind in i[0]]

    def add_vector_database_for_file(self, file_path: str):
        last_file_unique_id = self.cur_file_unique_id
        last_file_name = self.cur_file_name
        last_string_db = self.cur_string_db
        last_vector_db = self.cur_vector_db
        self.cur_file_name = file_path.split("/")[-1]
        self.cur_file_unique_id = file_path.split("/")[-2]
        knowledge_file = KnowledgeFile(filename=file_path)
        try:
            last_model_status = self.model_status
            self.model_status = 2
            plain_doc = knowledge_file.docs2texts()
            emb = self.docs2embedding([item.page_content for item in plain_doc])
            emb = np.array(emb).astype(np.float32)
            if not emb.flags['C_CONTIGUOUS']:
                emb = np.ascontiguousarray(emb)
            self.embeddings_size = emb.shape[1]
            self.cur_vector_db = faiss.IndexFlatL2(self.embeddings_size)
            self.cur_vector_db.add(emb)
            os.makedirs(os.path.join(RAG_SERVER.DATABASE_PATH, self.cur_file_unique_id), exist_ok = True)
            faiss.write_index(self.cur_vector_db, os.path.join(RAG_SERVER.DATABASE_PATH, self.cur_file_unique_id, "db.index"))
            byte_stream = pickle.dumps(plain_doc)
            self.cur_string_db = byte_stream
            with open(os.path.join(RAG_SERVER.DATABASE_PATH, self.cur_file_unique_id, "db.string"), "wb") as file:
                file.write(byte_stream)
            with open(os.path.join(RAG_SERVER.DATABASE_PATH, self.cur_file_unique_id, "name.txt"), "w", encoding="utf-8") as file:
                file.write(self.cur_file_name)
            self.model_status = last_model_status 
            if os.path.isfile(RAG_SERVER.FILE_RECORDS_LAST_USED_DB):
                with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'w', encoding="utf-8") as file:
                    file.write(self.cur_file_unique_id)
            status = True
        except:
            if os.path.exists(os.path.join(RAG_SERVER.DATABASE_PATH, self.cur_file_unique_id)):
                shutil.rmtree(os.path.join(RAG_SERVER.DATABASE_PATH, self.cur_file_unique_id))
            self.cur_file_unique_id = last_file_unique_id
            self.cur_file_name = last_file_name
            self.cur_string_db = last_string_db
            self.cur_vector_db = last_vector_db
            status = False
        return status
    
    def select_vector_database_to_use(self, unique_id: str, file_name: str):
        self.cur_file_name = file_name
        self.cur_file_unique_id = unique_id
        self.cur_vector_db = faiss.read_index(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, 'db.index'))
        with open(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, 'db.string'), "rb") as file:
            byte_stream = file.read()
        self.cur_string_db = pickle.loads(byte_stream)
        if os.path.isfile(RAG_SERVER.FILE_RECORDS_LAST_USED_DB):
            with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'w', encoding="utf-8") as file:
                file.write(self.cur_file_unique_id)
        return True

    def del_vector_database_for_file(self, unique_id: str, file_name: str):
        if self.cur_file_unique_id == unique_id:
            self.cur_file_name = None
            self.cur_file_unique_id = None
            self.cur_string_db = None
            self.cur_vector_db = None
            if os.path.isfile(RAG_SERVER.FILE_RECORDS_LAST_USED_DB):
                with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'w', encoding="utf-8") as file:
                    file.write("")
        if os.path.exists(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id, file_name)):
            shutil.rmtree(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id))
        if os.path.exists(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id)):
            shutil.rmtree(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id))
        return True

    @staticmethod
    def get_vector_database_map()->dict[str, str]:
        knowledge_map = dict()
        if os.path.exists(RAG_SERVER.DATABASE_PATH):
            for folder_name in os.listdir(RAG_SERVER.DATABASE_PATH):
                folder_path = os.path.join(RAG_SERVER.DATABASE_PATH, folder_name)
                if os.path.isdir(folder_path):
                    name_file_path = os.path.join(folder_path, 'name.txt')
                    if os.path.isfile(name_file_path):
                        with open(name_file_path, 'r') as name_file:
                            file_name = name_file.read().strip()
                            knowledge_map[folder_name] = file_name
        return knowledge_map

@app.post("/add_knowledge")
async def add_knowledge(file: UploadFile = File(...)):
    # 检查文件是否接收成功
    if not file.filename:
        return JSONResponse(content={
            "code": 1,
            "message": "file uploaded failed.",
            "id": "None"
        })

    # 提取文件的扩展名
    extension = file.filename.split(".")[-1].lower()
    if extension not in RAG_SERVER.SUPPORTED_EXT:
        return JSONResponse(content={
            "code": 2,
            "message": "unsupported file format.",
            "id": "None"
        }) 

    # 通过uuid的前缀给当前上传文件生成唯一标识符
    unique_id = str(uuid4()).split('-')[0]
    while unique_id in RAG_SERVER.get_vector_database_map():
        unique_id = str(uuid4()).split('-')[0]

    # 获取RAG_SERVER单例
    rag_server = RAG_SERVER() 

    # 保存文件到指定目录
    os.makedirs(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id), exist_ok = True)
    file_path = os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except:
        return JSONResponse(content={
            "code": 3,
            "message": "failed to save the uploaded file.",
            "id": "None"
        })

    # 给上传的文件生成对应的向量数据库文件
    try:
        rag_server.add_vector_database_for_file(file_path)
    except:
        # 如果生成向量数据库失败，删除对应的文件
        if os.path.exists(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id)):
            shutil.rmtree(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id)) 
        return JSONResponse(content={
            "code": 4,
            "message": "failed to generate file's knowledge, file has been removed.",
            "id": "None"
        })

    # 返回正常响应状态，文件的唯一标识符
    return JSONResponse(content={
        "code": 0,
        "message": "",
        "id": unique_id
    })

@app.post("/del_knowledge")
async def del_knowledge(request: DelKnowledgeRequest):
    if request.messageType != "del_knowledge":
        return JSONResponse(content={
            "code": 1,
            "message": "messageType is illegal"
        })
    if RAG_SERVER.get_vector_database_map()[request.id] != request.fileName:
        return JSONResponse(content={
            "code":2,
            "message": "id and file name is not consistent"
        })
    RAG_SERVER().del_vector_database_for_file(request.id, request.fileName)
    return JSONResponse(content={
        "code": 0,
        "message": ""
    })

@app.post("/query_knowledge_used")
async def query_cur_knowledge():
    return JSONResponse(content={
        "code": 0,
        "message": "",
        "id": RAG_SERVER().cur_file_unique_id,
        "file_name": RAG_SERVER().cur_file_name
    })


@app.post("/mod_knowledge_used")
async def modify_knowledge_used(request: SelectKnowledgeRequest):
    if request.messageType != "mod_knowledge_used":
        return JSONResponse(content={
            "code": 1,
            "message": "messageType is illegal",
        })
    if request.id not in RAG_SERVER.get_vector_database_map().keys():
        return JSONResponse(content={
            "code":2,
            "message": "selected id is not exist",
        })
    if RAG_SERVER.get_vector_database_map()[request.id] != request.fileName:
        return JSONResponse(content={
            "code":3,
            "message": "id and file name is not consistent"
        })
    rag_server = RAG_SERVER()
    rag_server.select_vector_database_to_use(request.id, request.fileName)
    return JSONResponse(content={
        "code":0,
        "message": ""
    })

@app.post("/query_knowledge_base")
async def query_knowledge_base():
    knowledge_list = [{"id": unique_id, "fileName": file_name} for unique_id, file_name in RAG_SERVER.get_vector_database_map().items()]
    return JSONResponse(content={
      "code": 0,
      "message": "",
      "knowledge_list": knowledge_list
  })

@app.post("/prompt_retrieval")
async def prompt_retrieval(request: PromptRetrievalRequest):
    if request.messageType != "prompt_retrieval":
        return JSONResponse(content={
            "code": 1,
            "message": "messageType is illegal",
            "retrieval_snippets": []
        })
    rag_server = RAG_SERVER()
    if rag_server.cur_file_unique_id is None:
        return JSONResponse(content={
            "code": 2,
            "message": "No knowledge is selected",
            "retrieval_snippets": []
        })
    try:
        while rag_server.model_status == 2:
            time.sleep(3)
        retrieval_snippets = rag_server.retrieval_from_vector_db(request.prompt)
        reranker_snippets = rag_server.reranker.compress_documents(retrieval_snippets, request.prompt, request.snippet_num)
    except:
        return JSONResponse(content={
            "code": 3,
            "message": "There was a problem during retrieval",
            "retrieval_snippets": []
        }) 

    reference = [ {"text": re.sub(r'[^\x20-\x7E\u4E00-\u9FFF]+','',x.page_content), "score":round(x.metadata['relevance_score'].item(), 2)} for x in reranker_snippets ]
    # reference = [ {"text": x.page_content.encode('utf-8', 'ignore').decode('utf-8'), "score":round(x.metadata['relevance_score'].item(), 2)} for x in reranker_snippets ]
    return JSONResponse(content={
        "code": 0,
        "message": "",
        "retrieval_snippets": reference
    })

@app.post("/status")
async def query_status():
    if RAG_SERVER().model_status == -1:
        return JSONResponse(content={
            "code": 1,
            "message": "RAG service did not start properly",
            "status": -1
        })
    if RAG_SERVER().model_status == 0:
        return JSONResponse(content={
            "code": 0,
            "message": "embedding machine is loadding",
            "status": 0
        })
    if RAG_SERVER().model_status == 2:
        return JSONResponse(content={
            "code": 0,
            "message": "knowledge for uploaded file is generating",
            "status": 2
        })
    return JSONResponse(content={
        "code": 0,
        "message": "service has been started",
        "status": 1
    })


if __name__ == "__main__":
    rag_server = RAG_SERVER()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18082)