# -*- coding: utf-8 -*-
"""
cookrag

描述: 基于V1
日期: 2026-05-03

也可以 指定其他知识库，只要是md文件
python main.py --data my_recipes

"""
from pathlib import Path
import argparse
import torch
import glob
import os
from dotenv import load_dotenv
import os
from typing import Any
import uuid
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient,FieldSchema, CollectionSchema,DataType
from pymilvus import RRFRanker,AnnSearchRequest
from langchain_core.prompts import ChatPromptTemplate
from scipy.sparse import coo_array, coo_matrix
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

BASE_COLLECTION_NAME = 'cookrag_ct_v1' # 新的知识库

class DocumentsGenerator:
    """
    数据加载，数据分块
    """
    def __init__(self, paths: list[str]) -> None:
        """
        paths: 加载一些md文件
        """
        self.documents = []
        self.load_documents(paths=paths)
        self.splitter_documents()

    def load_documents(self, paths: list[str]) -> None :
        """ 将每个md文档加载为对应document结构 ，添加id辅助后面关联"""
        documents = []
        for md_file in paths:
            loader = TextLoader(file_path=md_file)
            data = loader.load()
            data = data[0]
            data.metadata['id'] = str(uuid.uuid4())
            documents.append(data)
        self.documents.extend(documents)       

    def splitter_documents(self) -> None:
        """ 对一些document分块 """
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators = ["\n\n", "\n", "。", "，", " ", ""],  # 分隔符优先级
            chunk_size = 200,
            chunk_overlap=10
        )
        all_sections = []

        # 由load传来的全文文档        
        for doc in self.documents:
            sections  = markdown_splitter.split_text(doc.page_content)
            for section in sections:
                section.metadata['parent_id'] = doc.metadata['id']
            all_sections.extend(sections)

        chunks = text_splitter.split_documents(all_sections)

        # page_content会丢失层级信息，只保留内容. 
        # 这导致一勺盐不知道是红烧肉还是豆腐
        # 需要手动为其注入
        for chunk in chunks:
            prefix = ""
            h1 = chunk.metadata.get('Header 1', '无')
            h2 = chunk.metadata.get('Header 2', '无')
            h3 = chunk.metadata.get('Header 3', '无')
            prefix = f"主题: {h1} > 章节: {h2} > 细节: {h3}\n内容: "
            chunk.page_content = prefix + chunk.page_content 
        self.documents.extend(chunks) # 原全文文档+ 分割产生的
    

class KnowledgeDB:
    def __init__(self, collection_name:str, documents:list[Document]) -> None:
        """ 初始化结构 """
        self.client = MilvusClient(uri='milvus_cookrag.db')
        self.embed_encoder = BGEM3EmbeddingFunction( 
            model_name='BAAI/bge-small-zh-v1.5', # Specify the model name
            device='cpu', # Specify the device to use
            use_fp16=False # 使用16位精度，加快速度
            )
        self.collection_name = collection_name
        self.build_collection(documents)
    def build_collection(self, documents:list[Document]):
        print('开始构建知识库')
        if self.client.has_collection(self.collection_name):
            # self.client.drop_collection(self.collection_name) 
            print('知识库已经存在')
            return
        # 定义数据库格式
        self._setup_collection()
        self.insert_collection(documents)
    def _setup_collection(self):
        """ 创建集合结构 """
        if self.client.has_collection(self.collection_name):
            # self.client.drop_collection(self.collection_name) 
            return
        # 定义数据库格式
        fileds = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True,auto_id = True),
            FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name='metadata', dtype=DataType.JSON),
            FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=self.embed_encoder.dim['dense']),
        ]
        schema = CollectionSchema(fields=fileds, description='cookrag')
        self.client.create_collection(collection_name=self.collection_name, schema=schema)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name='dense_vector',
            index_type='IVF_FLAT', # 索引类型
            metric_type='IP',  # 
        )
        self.client.create_index(collection_name=self.collection_name, index_params=index_params)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name='sparse_vector',
            index_type='SPARSE_INVERTED_INDEX', # 索引类型
            metric_type='IP',  # 必须显式指定为内积

        )
        self.client.create_index(collection_name=self.collection_name, index_params=index_params)
    
    def _format_sparse_vector_to_dict(self, coo_arr: coo_array) -> dict:
        """ 
        将 coo_array 数组 转换为 Milvus 稀疏向量字段要求的字典格式: 
        {index: value}
        """
        # 1D coo_array 的坐标在 coords[0] 中
        indices = coo_arr.coords[0] 
        return dict(zip(indices, coo_arr.data))
    
    def insert_collection(self, documents:list[Document])->None:
        """ 把docuemnts 放到向量数据库 """
        data_to_insert = []

        for doc in documents:
            embeddings = self.embed_encoder([doc.page_content])
            data_to_insert.append({
                'metadata': doc.metadata,
                'dense_vector': embeddings['dense'][0],
                'sparse_vector': self._format_sparse_vector_to_dict(embeddings['sparse'][0]),
                'content': doc.page_content
            })
        self.client.insert(
            collection_name=self.collection_name,
            data = data_to_insert
        )


    def knowledge_search(self, queries: list[str]) -> list[dict[str, Any]]:
        """ 混合检索 稀疏向量和稠密向量, 多路搜索, 只返回父文档 """
        hybrid_results = []
        for query in queries:
            query_embeddings = self.embed_encoder([query])
            dense_vec, sparse_vec = query_embeddings['dense'][0], self._format_sparse_vector_to_dict(query_embeddings['sparse'][0])
            # RRF
            rerank = RRFRanker(k=60) # 会融合两个结果，得到, k=60表示平滑参数（就是两个结果平衡参数）
            dense_req = AnnSearchRequest(
                [dense_vec],
                anns_field='dense_vector',
                limit=30,
                param={"metric_type": "IP"}
            )
            sparse_req = AnnSearchRequest(
                [sparse_vec],
                anns_field='sparse_vector',
                limit=30,
                param={"metric_type": "IP"}
            )
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=rerank,
                limit=30,
                output_fields=['content','metadata']
            )[0]
            for r in results:
                hybrid_results.append(r)

        # 只返回父亲主页文档，否则导致知识重复，浪费上下文。
        need_ids = set()
        for r in hybrid_results:
            parent_id = r['metadata'].get('parent_id', '')
            # 如果由parentid 那就是子chunk
            if parent_id:
                need_ids.add(parent_id)
            # 如果有id 那就是父亲
            id = r['metadata'].get('id', '')
            if id:
                need_ids.add(id)
        final_results = []
        if need_ids:
            pid_list = list(need_ids)

            # 准确查询
            parent_results = self.client.query(
                collection_name=self.collection_name,
                filter=f"metadata['id'] in {pid_list}",
                output_fields=['content','metadata']

            )
            # 可以考虑不要格式化，因为现在都是返回父chunK,
            formatted_parents = []
            for p in parent_results:
                formatted_parents.append({
                    'id': p['id'],
                    'distance': 0.0, # 补齐字段，防止后续代码报错
                    'entity': {
                        'content': p['content'],
                        'metadata': p['metadata']
                    }
                })
            final_results.extend(formatted_parents)
        return final_results
class Rag:
    def __init__(self, data_folder:str) -> None:

        paths = glob.glob(os.path.join(data_folder, '*.md'))
        # 比如data/cooking
        folder_name = os.path.basename(os.path.normpath(data_folder))

        self.doc_generator = DocumentsGenerator(paths=paths) # 从这里读取知识
        print('知识文件加载成功')
        self.knowledge = KnowledgeDB(BASE_COLLECTION_NAME+folder_name, self.doc_generator.documents)
        self.llm = ChatOpenAI(
            api_key= SecretStr(os.getenv('OPENROUTER_API_KEY') or ""), 
            base_url="https://openrouter.ai/api/v1",
            model='tencent/hy3-preview:free',
        ) 
    def query_rewrite(self, user_query: str):
        #
        chat_template = ChatPromptTemplate.from_messages([
            ("system", (
                """
                你是一个烹饪饮食方面的智能查询分析助手。请根据用户输入的查询，进行重写。
                重写要求：
                - 如果用户查询已经含义明确：不需要扩展重写。
                - 如果用户查询模糊简短：从多个方面进行扩展，扩展出多个明确的查询。

                重写原则：
                - 保持原意不变
                - 保持简洁性

                请输出最终查询,如多个查询按照换行分割:
                """
            )),
            ("human", "原始问题：{user_query}")
        ])
        
        # 链式调用
        chain = chat_template | self.llm | StrOutputParser()
        
        response = chain.invoke({"user_query": user_query})
        # 处理结果
        rewritten_queries = [q.strip() for q in response.split('\n') if q.strip()]
        return list(set(rewritten_queries))
    
    def build_context(self, knowledges: list[dict[str, Any]])->str:
        """ 知识库返回结果转换位上下文 """
        context = ""
        for kd in knowledges:
            context+= kd['entity']['content']
            # print(f'distance:{kd['distance']:.4f}, content:{kd['entity']['content']}')
        return context
    def rag(self, query:str)->str:
        """ 用户接口 """
        template = ChatPromptTemplate(
            messages=[
                ('system', '你是一个专业的烹饪饮食专家, 严格根据上下文信息回答，如果不知道就说不知道'),
                ('human', """
                上下文信息：{context} 
                        
                用户输入:{query}
                """)
                    ]
        )
        chain = template | self.llm | StrOutputParser()  # 为了方便 重写和llm 都使用了同一个llm
        rewrite_queries = self.query_rewrite(user_query=query)
        knowledges = self.knowledge.knowledge_search(queries=rewrite_queries)
        context = self.build_context(knowledges=knowledges)
        response = chain.invoke({'context': context, 'query':query})
        return response
    def close(self):
        self.knowledge.client.close()
    def __enter__(self):
        # with语句进入触发
        return self
    def __exit__(self, exc_type, exc, tb):
        # with语句退出触发
        self.close()

def main():
    cuda_available = torch.cuda.is_available()
    load_dotenv()
    print(f'GPU加速: {cuda_available}')

    parser = argparse.ArgumentParser(description="RAG 知识库问答系统")
    parser.add_argument(
        '--data', 
        type=str, 
        default='data', # 默认值还是 data 目录
        help='指定存放 .md 文件的目录路径 (默认为当前目录下的 data)'
    )
    args = parser.parse_args()
    # 如果用户传的是绝对路径就用绝对路径，否则拼接到当前路径
    data_folder = args.data if os.path.isabs(args.data) else os.path.join(os.getcwd(), args.data)
    print(f'rag将从{data_folder}构建')
    paths = glob.glob(os.path.join(data_folder, '*.md'))
    if not paths:
        print(f"未在 {data_folder} 找到任何文件，请检查路径。")
    else:
        with Rag(data_folder=data_folder) as ragobj:
            while True:
                user_input = input('请输入您的查询（输入 exit 退出）：').strip() 
                if not user_input:
                    continue
                if user_input.lower() == 'exit':
                    print("程序已退出。")
                    break
                res = ragobj.rag(user_input)
                print("-" * 20)
                print(f"回答：\n{res}")

if __name__ == "__main__":
    main()