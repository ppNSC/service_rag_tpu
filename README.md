# 检索增强生成（RAG）服务

## 目录
- [检索增强生成（RAG）服务](#检索增强生成rag服务)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 目录结构](#3-目录结构)
  - [4. 基础层使用说明](#4-基础层使用说明)
    - [4.1 依赖项](#41-依赖项)
    - [4.1.1 x86/arm PCIe平台](#411-x86arm-pcie平台)
    - [4.1.2 BM1684X SoC平台](#412-bm1684x-soc平台)
    - [4.1.3 BM1688 SoC平台](#413-bm1688-soc平台)
    - [4.2 接口说明](#42-接口说明)
    - [4.3 参数说明](#43-参数说明)
  - [5. 服务层使用说明](#5-服务层使用说明)
    - [5.1 依赖项](#51-依赖项)
    - [5.2 准备模型与数据](#52-准备模型与数据)
    - [5.3 添加知识库](#53-添加知识库)
    - [5.4 删除知识库](#54-删除知识库)
    - [5.5 查询当前使用的知识库](#55-查询当前使用的知识库)
    - [5.6 选择使用的知识库](#56-选择使用的知识库)
    - [5.7 查询知识库列表](#57-查询知识库列表)
    - [5.8 查询服务状态](#58-查询服务状态)
    - [5.9 检索增强prompt](#59-检索增强prompt)


## 1. 简介
该例程是一个基于BM1684X/BM1688构建的检索增强生成（Retrieval-Augmented Generation）服务。用户首先上传文档以生成文档相关的知识库，在问答阶段输入用户的提问和选定的知识库，在知识库中检索与用户提问相关的内容，然后对用户的提问进行增强，以提升LLM模型的生成质量。

## 2. 特性
* 支持BM1688(SoC)平台
* 支持BM1684X(PCIE、SOC)平台
* 支持如下RAG相关算法：
  * [bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)

## 3. 目录结构

## 4. 基础层使用说明

### 4.1 依赖项

### 4.1.1 x86/arm PCIe平台

### 4.1.2 BM1684X SoC平台

### 4.1.3 BM1688 SoC平台

### 4.2 接口说明
为了能够方便用户理解算法逻辑，实现快速定制化开发。该例程定义了RAG算法的一般接口，您可以通过实现这些接口来增加新的算法。

- 添加向量数据库

  ```python
      add_vector_database_for_file(file_path: str) -> bool
  ```

  - 输入参数说明:
    - file_path: 字符串类型，表示需要生成其向量数据库的目标文件的所在路径
  - 返回值说明, 返回一个布尔类型，表示是否添加成功

- 删除向量数据库

  ```python
      del_vector_database_for_file(uuid_prefix: str, file_name: str) -> bool
  ```

  - 输入参数说明:
    - uuid_prefix: 字符串类型，表示需要删除的向量数据库的标识符
    - file_name: 字符串类型，表示需要删除的向量数据库对应的文件的文件名
  - 返回值说明, 返回一个布尔类型，表示是否删除成功

- 返回已有知识库的列表

  ```python
      get_knowledge_map() -> Dict[str, str]
  ```
  - 返回值说明，返回一个字典，元素为已有的向量数据库的标识符和文件名对

- RAG检索增强

  ```python
      query_retrieval(query: str) -> List[str]
  ```

  - 输入参数说明:
    - query: 字符串类型，表示用户本次的提问内容
  - 返回值说明, 返回一个字符串构成的列表，表示检索到的若干相关片段

### 4.3 参数说明

## 5. 服务层使用说明

### 5.1 依赖项

### 5.2 准备模型与数据

### 5.3 添加知识库

- 发送请求URL: http://IP:PORT/add_knowledge

- 接受消息格式（表单模式）

|  字段名称 |  类型   |      参数值      |
| -------- | ------ | --------------- |
|   file   |  File  |   SE-9资料.txt   |

  -  参数说明
     -  file: 需要上传和生成其知识库的文件

- 返回的消息格式

  ```js
  {
      "code": 0,
      "message": "",
      "id": "4502"
  }
  ```

  -  参数说明
     -  code: code为0表示添加成功，为1时表示添加时发生错误
     -  message: 当code为1时的错误信息，成功时为空
     -  id: rag服务给上传的文件生成的唯一标识符

### 5.4 删除知识库

- 发送请求URL: http://IP:PORT/del_knowledge

- 接受消息格式

  ```js
  {
      "messageType": "del_knowledge",
      "id": "4502",
      "file_name": "SE-9资料.txt"
  }
  ```

  -  参数说明
     -  messageType: 固定为"del_knowledge"
     -  id: 添加知识库时给文件生成的唯一标识id
     -  file_name: 需要删除其向量数据库的文件名

- 返回的消息格式

  ```js
  {
      "code": 0,
      "message": ""
  }
  ```

  -  参数说明
     -  code: code为0表示删除成功，为1时表示删除时发生错误
     -  message: 当code为1时的错误信息，成功为空

### 5.5 查询当前使用的知识库


- 发送请求URL: http://IP:PORT/query_cur_knowledge

- 接受消息格式

  ```js
  {
      "messageType": "query_cur_knowledge",
  }
  ```

  -  参数说明
     -  messageType: 固定为"query_cur_knowledge"

- 返回的消息格式

  ```js
  {
      "code": 0,
      "message": "",
      "id": "4538cc02",
      "file_name": "SE9资料.pdf"
  }
  ```

  -  参数说明
     -  code: code为0表示删除成功，其他表示查询当前使用的知识库时发生错误
     -  message: 当code非0时的错误信息，查询成功为空
     -  id: 当前使用中的知识库的唯一标识符
     -  file_name: 当前使用的知识库对应的文件名

### 5.6 选择使用的知识库

- 发送请求URL: http://IP:PORT/select_knowledge

- 接受消息格式

  ```js
  {
      "messageType": "select_knowledge",
      "id": "4fg7o502",
      "file_name": "SE-9资料.txt"
  }
  ```

  -  参数说明
     -  messageType: 固定为"select_knowledge"
     -  id: 添加知识库时给文件生成的唯一标识id
     -  file_name: 将要选择的知识库对应的文件名

- 返回的消息格式

  ```js
  {
      "code": 0,
      "message": ""
  }
  ```

  -  参数说明
     -  code: code为0表示选择特定知识库成功，其他则表示有错误
     -  message: 当code非0时的错误信息，成功为空

### 5.7 查询知识库列表

- 发送请求URL: http://IP:PORT/query_knowledge_base

- 接受消息格式

  ```js
  {
      "messageType": "query_knowledge_base"
  }
  ```

  -  参数说明
     - messageType: 固定为"query_knowledge_base"

- 返回的消息格式

  ```js
  {
      "code": 0,
      "message": "",
      "knowledge_list": [
  		{
        "id": "4538cc02",
        "fileName": "SE-9资料.txt"
      },
  		{
        "id": "56f2g147",
        "fileName": "SM-7资料.txt"
      }
      ]
  }
  ```

  -  参数说明
     - code: code为0表示查询成功，为1时表示查询时发生错误
     - message: code为1时的错误信息
     - knowledge_list: 查询到的文件标识符和文件名对

### 5.8 查询服务状态

- 发送请求URL: http://IP:PORT/status

- 接受消息格式

  ```js
  {
      "messageType": "rag_status"
  }
  ```

  -  参数说明
     -  messageType: 固定为"rag_status"

- 返回的消息格式

  ```js
  {
      "code": 0,
      "message":"",
      "statu":0,
      "model_list":[
        "bce-embedding",
      ]
  }
  ```

  -  参数说明
     - code: code为0表示查询服务状态成功，为1时表示发生错误
     - message: code为1时的错误信息
     - statu: 0表示rag服务状态有异常，1表示rag服务状态正常
     - model_list: statu为0时，显示状态异常的模型。statu为1时为空

### 5.9 检索增强prompt

- 发送请求URL: http://IP:PORT/prompt_retrieval

- 接受消息格式

  ```js
  {
      "messageType": "prompt_retrieval",
      "snippet_num":2,
      "prompt": "BM1688的特点？"
  }
  ```

  - 参数说明
    - messageType: 固定为"rag_enhanced"
    - snippet_num: 设定需要的相关文本段数量上限
    - prompt: 用户的提问 

- 返回消息格式

  ```js
  {
      "code": 0,
      "message":"",
      "prompt": ["BM1688是端侧芯片，能耗比很好。",
                "BM1688的各项算力为..."]
  }
  ```

  - 参数说明
    - code：0代表rag增强成功，为1表示不成功
    - message: code为1时的错误信息
    - prompt: 检索到的相关文本段
