import argparse
import asyncio
from enum import Enum
import os
from pydantic import BaseModel
import re
import shutil
import sys
import threading
import time

import faiss
from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse
import nltk
import numpy as np
import pickle
from uuid import uuid4

current_dir = os.path.dirname(os.path.abspath(__file__))
# prepare Natural Language Toolkit
nltk_data_path = os.path.join(current_dir, './nltk_data/')
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

import SILK2.Tools.logger as Logger
from doc_processor.knowledge_file import KnowledgeFile
from embedding import Word2VecEmbedding
from reranker import RerankerTPU

class SelectKnowledgeRequest(BaseModel):
    messageType: str
    id: str
    fileName: str

class PromptRetrievalRequest(BaseModel):
    messageType: str
    snippet_num: int
    prompt: str

app = FastAPI()

class RAG_SERVER:
    _singleton = None
    _config_initialized = False

    class ServiceStatus(Enum):
        SERVICE_NOT_STARED = -1
        SERVICE_PREPARING = 0
        SERVICE_READY = 1
        SERVICE_ADDING_KNOWLEDGE = 2
        SERVICE_DELETING_KNOWLEDGE = 3
        SERVICE_QUERYING_CURRENT_KNOWLEDGE = 4
        SERVICE_MODIFYING_KNOWLEDGE = 5
        SERVICE_QUERYING_KNOWLEDGE_BASE = 6
        SERVICE_RETRIEVING = 7

    STATUS_LOCK = asyncio.Lock()

    UPLOAD_PATH = './knowledge_base/uploaded_file'
    DATABASE_PATH =  './knowledge_base/vector_database'
    FILE_RECORDS_LAST_USED_DB = './knowledge_base/cur_knowledge_used.txt'
    SUPPORTED_EXT = ["pdf", "txt", "docx", "pptx", 'png', 'jpg', 'jpeg', 'bmp']

    def __init__(self, args = None) -> None:
        # config
        if not RAG_SERVER._config_initialized:
            RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_NOT_STARED

            self.logger = Logger.Log("RAG_SERVER", args.log_level)
            self.logger.info(f"{Logger.file_lineno()} 日志系统启动，当前设置为:"+args.log_level)  
            self.logger.info(f"{Logger.file_lineno()} 当前embedding模型为:"+args.embedding_model_name)
            self.logger.info(f"{Logger.file_lineno()} 当前reranker模型为:"+args.reranker_model_name)

            self.embedding_model_paths = {
                'BM1688_bce_base_v1': "./models/BM1688/bce_embedding/bce-embedding-base_v1.bmodel",
                'BM1688_bce_base_v1_tokenizer': "./models/BM1688/bce_embedding/token_config",
                'BM1684X_bce_base_v1': "./models/BM1684X/bce_embedding/bce-embedding-base_v1.bmodel",
                'BM1684X_bce_base_v1_tokenizer': "./models/BM1684X/bce_embedding/token_config"
            }
            self.reranker_model_paths = {
                'BM1688_bce_base_v1': "./models/BM1688/bce_reranker/bce-reranker-base_v1.bmodel",
                'BM1688_bce_base_v1_tokenizer': "./models/BM1688/bce_reranker/token_config",
                'BM1684X_bce_base_v1': "./models/BM1684X/bce_reranker/bce-reranker-base_v1.bmodel",
                'BM1684X_bce_base_v1_tokenizer': "./models/BM1684X/bce_reranker/token_config"
            }
            self.embedding_model_name = args.embedding_model_name
            self.reranker_model_name = args.reranker_model_name
            self.dev_id = args.dev_id
            self.set_vector_db_empty()

            RAG_SERVER._config_initialized = True

    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super(RAG_SERVER, cls).__new__(cls)
        return cls._singleton

    def start_service(self,):
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_PREPARING
        # 获取并格式化当前时间，用于打印服务启动信息
        self.logger.info(f"{Logger.file_lineno()} RAG服务启动，模型启动中...")   
        self.logger.info(f"{Logger.file_lineno()} RAG模型状态:{RAG_SERVER.SERVICE_STATUS}，正在加载...")
        import sophon.sail as sail
        try:
            self.target = sail.Handle(self.dev_id).get_target()#获取设备型号
        except:
            self.logger.error(f"{Logger.file_lineno()} 设备{self.dev_id}不存在!")
            RAG_SERVER.SERVICE_STATUS= RAG_SERVER.ServiceStatus.SERVICE_NOT_STARED #模型加载失败
            sys.exit(1)
        try:
            embedding_bmodel_path = self.embedding_model_paths[self.target+'_'+self.embedding_model_name]
        except KeyError:
            self.logger.error(f"{Logger.file_lineno()} 模型{self.embedding_model_name}不存在!")
            RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_NOT_STARED #模型加载失败
            sys.exit(1)
        try:
            reranker_bmodel_path = self.reranker_model_paths[self.target+'_'+self.reranker_model_name]
        except KeyError:
            self.logger.error(f"{Logger.file_lineno()} 模型{self.reranker_model_name}不存在!")
            RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_NOT_STARED #模型加载失败
            sys.exit(1)

        curr_path = os.path.dirname(os.path.abspath(__file__))
        embedding_model_tokenizer_path = self.embedding_model_paths[self.target+'_'+self.embedding_model_name+'_tokenizer']
        embedding_model_tokenizer_path = os.path.abspath(os.path.join(curr_path, embedding_model_tokenizer_path))
        embedding_bmodel_path = os.path.abspath(os.path.join(curr_path, embedding_bmodel_path))
        reranker_model_tokenizer_path = self.reranker_model_paths[self.target+'_'+self.reranker_model_name+'_tokenizer']
        reranker_model_tokenizer_path = os.path.abspath(os.path.join(curr_path, reranker_model_tokenizer_path))
        reranker_bmodel_path = os.path.abspath(os.path.join(curr_path, reranker_bmodel_path))
        try:
            self.embedding_machine = Word2VecEmbedding(embedding_bmodel_path,
                                                       embedding_model_tokenizer_path,
                                                       256,
                                                       self.dev_id)
        except:
            RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_NOT_STARED
            self.logger.info(f"{Logger.file_lineno()} RAG模型状态:{RAG_SERVER.SERVICE_STATUS},embedding模型加载失败.")
            return False
        self.logger.info(f"{Logger.file_lineno()} RAG模型状态:{RAG_SERVER.SERVICE_STATUS},embedding模型加载完成.")
        try:
            self.reranker_machine = RerankerTPU(reranker_bmodel_path,
                                                reranker_model_tokenizer_path,
                                                self.dev_id)
        except:
            RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_NOT_STARED
            self.logger.info(f"{Logger.file_lineno()} RAG模型状态:{RAG_SERVER.SERVICE_STATUS},reranker模型加载失败.")
            return False
        self.logger.info(f"{Logger.file_lineno()} RAG模型状态:{RAG_SERVER.SERVICE_STATUS},reranker模型加载完成.")

        try:
            # read knowledge id last used
            if os.path.isfile(RAG_SERVER.FILE_RECORDS_LAST_USED_DB):
                with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'r') as record_id:
                    last_used_id = record_id.read().strip()
                    exist_knowledge_database = RAG_SERVER.get_vector_database_map()
                    if last_used_id in exist_knowledge_database.keys():
                        self.select_vector_database_to_use(last_used_id, exist_knowledge_database[last_used_id])
        except:
            # if file recorded knowledge last used was deleted, set vector database to empty
            self.logger.info(f"{Logger.file_lineno()} 未读取到上次退出前使用的知识库，当前使用的知识库设为空.")
            self.set_vector_db_empty()

        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return True

    @staticmethod
    def get_service_status_string(status: ServiceStatus) -> str:
        return status.name.lower()

    @staticmethod
    def get_vector_database_map()->dict[str, str]:
        # generate from DATABASE_PATH
        # add database to database_map if all three files exist
        knowledge_map = dict()
        if os.path.exists(RAG_SERVER.DATABASE_PATH):
            for folder_name in os.listdir(RAG_SERVER.DATABASE_PATH):
                folder_path = os.path.join(RAG_SERVER.DATABASE_PATH, folder_name)
                if os.path.isdir(folder_path):
                    files_name_need_check = ['name.txt', 'db.string', 'db.index']
                    files_path_need_check = [os.path.join(folder_path, file_name) for file_name in files_name_need_check]
                    for file_path in files_path_need_check:
                        if not os.path.isfile(file_path):
                            continue
                    try:
                        with open(os.path.join(folder_path, "name.txt"), 'r') as name_file:
                            file_name = name_file.read().strip()
                            knowledge_map[folder_name] = file_name
                    except:
                        pass
        return knowledge_map

    def set_vector_db_empty(self):
        self.cur_file_name = None
        self.cur_file_unique_id = None
        self.cur_vector_db = None
        self.cur_string_db = None 

    def retrieval_from_vector_db(self, query: str):
        # current batch size of reranker model if 3.
        k = 3
        try:
            query_embedding = self.embedding_machine.embed_query(query)
            _, i = self.cur_vector_db.search(x = np.array([query_embedding]), k = k)
        except:
            return []
        return [self.cur_string_db[ind] for ind in i[0]]

    def add_vector_database_for_file(self, file_path: str):
        last_file_unique_id = self.cur_file_unique_id
        last_file_name = self.cur_file_name
        last_string_db = self.cur_string_db
        last_vector_db = self.cur_vector_db
        unique_id = os.path.basename(os.path.dirname(file_path))
        try:
            knowledge_file = KnowledgeFile(filename=file_path)
            plain_doc = knowledge_file.docs2texts()
            emb = self.embedding_machine.docs2embedding([item.page_content for item in plain_doc])
            emb = np.array(emb).astype(np.float32)
            if not emb.flags['C_CONTIGUOUS']:
                emb = np.ascontiguousarray(emb)
            self.embeddings_size = emb.shape[1]
            self.cur_vector_db = faiss.IndexFlatL2(self.embeddings_size)
            self.cur_vector_db.add(emb)
            os.makedirs(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id), exist_ok = True)
            faiss.write_index(self.cur_vector_db, os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, "db.index"))
            byte_stream = pickle.dumps(plain_doc)
            self.cur_string_db = byte_stream
            with open(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, "db.string"), "wb") as file:
                file.write(byte_stream)
            with open(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, "name.txt"), "w", encoding="utf-8") as file:
                file.write(os.path.basename(file_path))
            self.cur_file_name = os.path.basename(file_path)
            self.cur_file_unique_id = unique_id
            with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'w', encoding="utf-8") as file:
                file.write(unique_id)
            status = True
        except:
            self.cur_file_unique_id = last_file_unique_id
            self.cur_file_name = last_file_name
            self.cur_string_db = last_string_db
            self.cur_vector_db = last_vector_db
            self.del_vector_database_for_file(file_path.split("/")[-2], file_path.split("/")[-1])
            status = False
        return status
    
    def select_vector_database_to_use(self, unique_id: str, file_name: str):
        if self.cur_file_unique_id == unique_id:
            return True
        last_file_unique_id = self.cur_file_unique_id
        last_file_name = self.cur_file_name
        last_string_db = self.cur_string_db
        last_vector_db = self.cur_vector_db
        try:
            self.cur_vector_db = faiss.read_index(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, 'db.index'))
            if os.path.isfile(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, 'db.string')):
                with open(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id, 'db.string'), "rb") as file:
                    byte_stream = file.read()
            self.cur_string_db = pickle.loads(byte_stream)
            self.cur_file_name = file_name
            self.cur_file_unique_id = unique_id
            with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'w', encoding="utf-8") as file:
                file.write(unique_id)
        except:
            self.cur_file_unique_id = last_file_unique_id
            self.cur_file_name = last_file_name
            self.cur_string_db = last_string_db
            self.cur_vector_db = last_vector_db
            return False
        return True

    def del_vector_database_for_file(self, unique_id: str, file_name: str):
        try:
            if self.cur_file_unique_id == unique_id:
                self.set_vector_db_empty()
                with open(RAG_SERVER.FILE_RECORDS_LAST_USED_DB, 'w', encoding="utf-8") as file:
                    file.write("")
            if os.path.exists(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id, file_name)):
                shutil.rmtree(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id))
            if os.path.exists(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id)):
                shutil.rmtree(os.path.join(RAG_SERVER.DATABASE_PATH, unique_id))
        except:
            return False
        return True


@app.post("/add_knowledge")
async def add_knowledge(file: UploadFile = File(...)):
    async with RAG_SERVER.STATUS_LOCK:
        if RAG_SERVER.SERVICE_STATUS != RAG_SERVER.ServiceStatus.SERVICE_READY:
            return JSONResponse(content={
                "code": 5,
                "message": "service is busy",
                "id": "None"
            })
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_ADDING_KNOWLEDGE

    # check if file received
    if not file.filename:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 1,
            "message": "file uploaded failed.",
            "id": "None"
        })

    # get extension of file
    extension = file.filename.split(".")[-1].lower()
    if extension not in RAG_SERVER.SUPPORTED_EXT:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 2,
            "message": "unsupported file format.",
            "id": "None"
        }) 

    # generate unique id via uuid4
    unique_id = str(uuid4()).split('-')[0]
    vector_db_map = RAG_SERVER.get_vector_database_map()
    while unique_id in vector_db_map:
        unique_id = str(uuid4()).split('-')[0]

    rag_server = RAG_SERVER() 

    # save file on server
    try:
        os.makedirs(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id), exist_ok = True)
        file_path = os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 3,
            "message": "failed to save the uploaded file.",
            "id": "None"
        })

    # generate vector database
    add_vector_database_success = await asyncio.to_thread(rag_server.add_vector_database_for_file, file_path) 
    if add_vector_database_success is False:
        # delete file if add vector database failed
        if os.path.exists(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id)):
            shutil.rmtree(os.path.join(RAG_SERVER.UPLOAD_PATH, unique_id)) 
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 4,
            "message": "failed to generate file's knowledge, file has been removed",
            "id": "None"
        })

    RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
    return JSONResponse(content={
        "code": 0,
        "message": "",
        "id": unique_id
    })

@app.post("/del_knowledge")
async def del_knowledge(request: SelectKnowledgeRequest):
    async with RAG_SERVER.STATUS_LOCK:
        if RAG_SERVER.SERVICE_STATUS != RAG_SERVER.ServiceStatus.SERVICE_READY:
            return JSONResponse(content={
                "code": 4,
                "message": "service is busy"
            })
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_DELETING_KNOWLEDGE
    if request.messageType != "del_knowledge":
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 1,
            "message": "messageType is illegal"
        })
    if request.id not in RAG_SERVER.get_vector_database_map():
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code":2,
            "message": "id is not recorded"
        })

    if RAG_SERVER.get_vector_database_map()[request.id] != request.fileName:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code":3,
            "message": "id and file name are not consistent"
        })
    if RAG_SERVER().del_vector_database_for_file(request.id, request.fileName) is True:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 0,
            "message": ""
        })
    else:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 4,
            "message": "delete file failed"
        })

@app.post("/query_knowledge_used")
async def query_cur_knowledge():
    async with RAG_SERVER.STATUS_LOCK:
        if RAG_SERVER.SERVICE_STATUS != RAG_SERVER.ServiceStatus.SERVICE_READY:
            return JSONResponse(content={
                "code": 1,
                "message": "service is busy",
                "id": "", 
                "fileName": ""
            })
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_QUERYING_CURRENT_KNOWLEDGE
    id = RAG_SERVER().cur_file_unique_id,
    file_name = RAG_SERVER().cur_file_name
    RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
    return JSONResponse(content={
        "code": 0,
        "message": "",
        "id": id,
        "fileName": file_name
    })

@app.post("/mod_knowledge_used")
async def modify_knowledge_used(request: SelectKnowledgeRequest):
    async with RAG_SERVER.STATUS_LOCK:
        if RAG_SERVER.SERVICE_STATUS != RAG_SERVER.ServiceStatus.SERVICE_READY:
            return JSONResponse(content={
                "code": 5,
                "message": "service is busy",
            })
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_MODIFYING_KNOWLEDGE
    if request.messageType != "mod_knowledge_used":
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 1,
            "message": "messageType is illegal",
        })
    if request.id not in RAG_SERVER.get_vector_database_map().keys():
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code":2,
            "message": "selected id is not exist",
        })
    if RAG_SERVER.get_vector_database_map()[request.id] != request.fileName:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code":3,
            "message": "id and file name are not consistent"
        })
    rag_server = RAG_SERVER()
    if rag_server.select_vector_database_to_use(request.id, request.fileName) is True:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code":0,
            "message": ""
        })
    else:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code":4,
            "message": "modify current knowledge failed"
        })

@app.post("/query_knowledge_base")
async def query_knowledge_base():
    async with RAG_SERVER.STATUS_LOCK:
        if RAG_SERVER.SERVICE_STATUS != RAG_SERVER.ServiceStatus.SERVICE_READY:
            return JSONResponse(content={
              "code": 1,
              "message": "service is busy",
              "knowledge_list": []
              })
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_QUERYING_KNOWLEDGE_BASE
    knowledge_list = [{"id": unique_id, "fileName": file_name} for unique_id, file_name in RAG_SERVER.get_vector_database_map().items()]
    RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
    return JSONResponse(content={
      "code": 0,
      "message": "",
      "knowledge_list": knowledge_list
      })

@app.post("/prompt_retrieval")
async def prompt_retrieval(request: PromptRetrievalRequest):
    async with RAG_SERVER.STATUS_LOCK:
        if RAG_SERVER.SERVICE_STATUS != RAG_SERVER.ServiceStatus.SERVICE_READY:
            return JSONResponse(content={
                "code": 4,
                "message": "service is busy",
                "retrieval_snippets": []
            })
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_RETRIEVING
    if request.messageType != "prompt_retrieval":
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 1,
            "message": "messageType is illegal",
            "retrieval_snippets": []
        })
    rag_server = RAG_SERVER()
    if rag_server.cur_file_unique_id is None:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 2,
            "message": "no knowledge is selected",
            "retrieval_snippets": []
        })
    try:
        retrieval_snippets = await asyncio.to_thread(rag_server.retrieval_from_vector_db, request.prompt)
        reranker_snippets = rag_server.reranker_machine.compress_documents(retrieval_snippets, request.prompt, request.snippet_num)
    except:
        RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
        return JSONResponse(content={
            "code": 3,
            "message": "there was a problem during retrieval",
            "retrieval_snippets": []
        }) 

    # only remains ASCII and Unicode
    reference = [ {"text": re.sub(r'[^\x20-\x7E\u4E00-\u9FFF]+','',x.page_content), "score":round(x.metadata['relevance_score'].item(), 2)} for x in reranker_snippets ]
    # reference = [ {"text": x.page_content.encode('utf-8', 'ignore').decode('utf-8'), "score":round(x.metadata['relevance_score'].item(), 2)} for x in reranker_snippets ]

    RAG_SERVER.SERVICE_STATUS = RAG_SERVER.ServiceStatus.SERVICE_READY
    return JSONResponse(content={
        "code": 0,
        "message": "",
        "retrieval_snippets": reference
    })

@app.post("/status")
async def query_status():
    message = RAG_SERVER.get_service_status_string(RAG_SERVER.SERVICE_STATUS)
    status = RAG_SERVER.SERVICE_STATUS.value
    return JSONResponse(content={
        "code": 0,
        "message": message, 
        "status": status
    })

def is_running(pid_file):
    """检查是否有其他实例在运行"""
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
            try:
                # 检查进程是否存在
                os.kill(pid, 0)
                return True  # 进程仍在运行
            except OSError:
                return False  # 进程不存在
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志等级,目前的等级有4个,从低到高依次是DEBUG、INFO、WARNING、ERROR,默认是INFO')
    # embedding model name
    parser.add_argument("--embedding_model_name", type=str, default="bce_base_v1", help="embedding model name")
    # reranker model name
    parser.add_argument("--reranker_model_name", type=str, default="bce_base_v1", help="reranker model name")
    # dev_id
    parser.add_argument("--dev_id", type=int, default=5, help="which TPU to load BModel")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    pid_file_path = '/tmp/arg_service.pid'
    if is_running(pid_file_path):
        current_time = time.localtime()
        formatted_time = time.strftime("%Y%m%d-%H:%M:%S", current_time)
        print(formatted_time,": RAG 服务已经在运行。")
        sys.exit(1)

    # 写入当前进程的 PID
    with open(pid_file_path, 'w') as f:
        f.write(str(os.getpid()))

    try:
        rag_server = RAG_SERVER(args)
        thread = threading.Thread(target=rag_server.start_service)
        thread.daemon = True
        thread.start()
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=18084)
    except KeyboardInterrupt:
        print("RAG 服务被中断。")
    finally:
        # 删除 PID 文件
        os.remove(pid_file_path)