import torch
import time
import glob
import os
from dotenv import load_dotenv
import os
from typing import Any
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import json
from langchain_core.output_parsers import JsonOutputParser
cuda_available = torch.cuda.is_available()
print(f"是否支持 CUDA (GPU加速): {cuda_available}")
load_dotenv()

llm = ChatOpenAI(
    api_key= 'ollama',
    base_url="http://localhost:11434/v1",
    model='qwen2.5:7b-instruct',
    temperature=0, # 更加严谨冷酷统一
) 
def graph_extractor(text: str) ->dict:
    """
    """
    chat_template = ChatPromptTemplate.from_messages([
        ("system", (
            """
            角色
            你是一个非常专业的知识图谱构建助手，擅长从非结构化文本中提取实体和逻辑关系。

            任务
            阅读提供的文档，根据以下定义的 Schema 提取知识三元组。

            自定义 Schema 
            1. 节点标签 (Labels):
            - Dish: 菜名：文档起始标题名字，去掉"的做法"。 
            - Ingredient: 食材名字如鸡蛋
            - CookingMethod: 烹饪技法包括：炒, 炸, 炖, 蒸, 煮, 拌, 烤, 焖, 炸, 煎
            - Category: 菜谱分类包括: 素菜/荤菜/水产/早餐/主食/汤类/甜品/饮料/调料
            - DifficultyLevel: 烹饪难度:1星、2星...、8星

            2. 关系类型 (Relationships):
            - REQUIRES_INGREDIENT: 菜品包含某种食材
            - USE_COOKING_METHOD: 菜品使用了某种技法
            - BELONGS_CATEGORY: 属于什么分类
            - HAS_DIFFICULTY_LEVEL: 难度等级
            - SUBSTITUTE_FOR:食材替代 

            要求：
            请尽可能详尽地提取，不要遗漏任何微小的关系。
            不要过度推理，只使用文档内容。
            推理只限于语言简单替换。
            涉及实体名字如：食材分类、烹饪方法、食材名字、菜谱分类等，必须准确简短，不能模糊
            content属性 必须完整且能重构节点，高度压缩但信息丰满

            示例输出：
            {{
            'nodes':[
                {{
                    'label': 'Dish',
                    ''name':'拔丝土豆',
                    "properties":{{
                            'category':['素菜'],
                        'steps': '1.土豆切块, 2...',
                        'note': '注意提示、附加内容',
                        'content':'拔丝土豆是一道经典的甜口素菜，通过炸制土豆挂糖浆制成，烹饪难度2星。其核心特征是外脆内软、金黄拉丝，适合作为甜点。'
                    }}
                }},
                {{
                    'name': '土豆',
                    'label': 'Ingredient',
                    "properties":{{
                            'category':'食材类别（蔬菜/调料/蛋白质/淀粉类/其他)',
                        'content':'节点内容'
                    }}

                }}
            ],
            'relationships':[
                {{
                    'source_label':'Dish'
                    'source_name':'拔丝土豆',
                    "target_name": "土豆",
                    "target_label": "Ingredient",
                    "type": "HAS_INGREDIENT",
                    "properties": {{ "amount": "2个" }}
                }}
            ]

            }}

            输出格式为json形式, 必须包括节点(名字、标签、属性), 关系(起始标签和名字、终止标签和名字、关系类型、关系属性)

            待处理文本
            {text}
            """
        )),
        ("human", "原始文本：{text}")
    ])
    
    # 链式调用
    chain = chat_template | llm| JsonOutputParser()
    
    response = chain.invoke({"text": text})
    # 处理结果
    return response


class GraphDBClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embed_encoder =  HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-zh-v1.5',
        )
        self.init_db()
        self.clear_database()

    def clear_database(self):
        with self.driver.session() as session:
            # 删除所有节点及其关系
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
            pass
    def close(self):
        self.driver.close()

    def save(self, content: dict):
        """
        content 应该是 LLM 返回的那个包含 'nodes' 和 'relationships' 的字典
        """
        with self.driver.session() as session:
            tx = session.begin_transaction()
            try:
                self._create_graph(tx, content)
                tx.commit() 
                print("Successfully committed.")
            except Exception as e:
                tx.rollback()
                print(f"Error: {e}")

    def init_db(self):
        """ 初始化向量索引 """
        index_query = """
        CREATE VECTOR INDEX `dish_vector_index` IF NOT EXISTS
        FOR (n:Dish) ON (n.dense_vector)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 512,
            `vector.similarity_function`: 'cosine'
            }}
        """
        with self.driver.session() as session:
            session.run(index_query)
            print("向量索引 `dish_vector_index` ")


    def _create_graph(self, tx, data):
        # 1. 插入节点
        # 标签名（Label）和关系类型（Type）不支持参数化，不能$
        # 使用 MERGE 确保节点唯一性（基于 name） ，，SET n += props 动态更新属性
        if "nodes" in data:
            for node in data["nodes"]: 
                # 对content产生向量嵌入
                embeddings = self.embed_encoder.embed_query(node['properties']['content'])
                node['properties']['dense_vector'] = embeddings

                query = f"""
                MERGE (n:{node['label']} {{name: $name}}) 
                SET n += $props
                """
                result = tx.run(query, name=node['name'], 
                    props=node.get('properties', {}))
        # 2. 插入关系
        if "relationships" in data:
            for rel in data["relationships"]:
                # 匹配 source 和 target 节点，然后建立有向关系
                # 同样，关系类型 type 需要拼接字符串
                # 起始节点设置为MERGE方便找错误
                query = f"""
                MERGE (a:{rel['source_label']}  {{name: $source_name}})
                MERGE (b:{rel['target_label']}  {{name: $target_name}})
                MERGE (a)-[r:{rel['type']}]->(b)
                SET r += $props
                """
                result = tx.run(query, 
                       source_name=rel['source_name'], 
                       target_name=rel['target_name'], 
                       props=rel.get('properties', {}))

    def execute_query(self, cypher: str, parameters=None) :
        """ 执行cypher 查询语句 返回查询结果的字符串列表 """
        results_as_strings = []
        
        with self.driver.session() as session:
            # 执行查询
            result = session.run(cypher,  parameters or {})
            result = result.data()
            return result
    def text2cypher(self, text:str) -> str:
        """
        text转为cypher: llm会识别为向量搜证或者正常搜索
        """
        chat_template = ChatPromptTemplate.from_messages([
            ("system", (
                """
                你是ne04j cypher的专家。根据下面信息将用户问题转为cypher语句。

                neo4j Schema 
                1. 节点标签 (Labels):
                - Dish: 菜名：文档起始标题名字，去掉"的做法"。 
                - Ingredient: 食材名字如鸡蛋
                - CookingMethod: 烹饪技法包括：炒, 炸, 炖, 蒸, 煮, 拌, 烤, 焖, 炸, 煎
                - Category: 菜谱分类包括: 素菜/荤菜/水产/早餐/主食/汤类/甜品/饮料/调料
                - DifficultyLevel: 烹饪难度:1星、2星...、8星

                2. 关系类型 (Relationships):
                - REQUIRES_INGREDIENT: 菜品包含某种食材
                - USE_COOKING_METHOD: 菜品使用了某种技法
                - BELONGS_CATEGORY: 属于什么分类
                - HAS_DIFFICULTY_LEVEL: 难度等级
                - SUBSTITUTE_FOR:食材替代 

                示例节点和关系数据：
                {{
                'nodes':[
                    {{
                        ''name':'拔丝土豆',
                        'label': 'Dish',
                        'properties':{{
                            'path':'',
                            'category':['素菜'],
                            'steps': '1.土豆切块, 2...',
                            'note': '放一些附加内容或者未注意到的文本内容'
                        }}
                    }},
                    {{
                        'name': '土豆',
                        'label': 'Ingredient',
                        'properties':{{
                            'category':'食材类别（蔬菜/调料/蛋白质/淀粉类/其他)'
                        }}
                    }}
                ],
                'relationships':[
                    {{
                        'source_label':'Dish'
                        'source_name':'拔丝土豆',
                        "target_name": "土豆",
                        "target_label": "Ingredient",
                        "type": "HAS_INGREDIENT",
                        "properties": {{ "amount": "2个" }}
                    }}
                ]

                }}

                要求：
                1. 只返回cypher语句，不要包含任何解释标签
                2. 确保cypher语法正确
                3. 使用上下文提供标签、关系、属性等
                4. RETURN 返回需要进行AS易懂的字段描述
                5. 当问题涉及模糊语义（如口感、场景、功效）时，请使用以下语法：
                    CALL db.index.vector.queryNodes('dish_vector_index', 5, $vec) YIELD node AS d
                    后续可以继续match和return

                用户问题
                {text}
                """
            )),
            ("human", "原始文本：{text}")
        ])
        
        # 链式调用
        chain = chat_template | llm|  StrOutputParser()
        
        response = chain.invoke({"text": text})
        # 处理结果
        return response

    def search(self, query:str):
        """
        用户查询接口：
        
        """
        cypher = self.text2cypher(text='我想吃点清爽解腻的，最近胃口不好')
        print(f'search: {query}\n{cypher}')
        # cypher里面有占位，以供embedd
        params = {}
        if "$vec" in cypher:
            # 你的 Python 代码在这里偷偷调用 Embedding 模型
            user_query_vector = self.embed_encoder.embed_query(query)
            params["vec"] = user_query_vector  # 把真正的向量数组放进字典
            params["k"] = 5 # 也可以让 LLM 指定 k

        # 3. 执行查询
        final_data = self.execute_query(cypher, parameters=params)
        return final_data

def load_files(file_paths:list[str]):
    """ 加载到图库， fangbianceshi """
    for path in file_paths:
        with open(path, mode= 'r', encoding='utf-8') as f:
            text = f.read()
            graph_content = graph_extractor(text=text)
            client.save(graph_content)
            print(f"{path} done. ..")
            time.sleep(10)

client = GraphDBClient("bolt://localhost:7687", "neo4j", "12345678")

def main():
    DATA_FOLDER = os.path.join(os.getcwd(), 'data')
    file_paths = glob.glob(os.path.join(DATA_FOLDER,  '*.md'))
    load_files(file_paths)

    result = client.search("我想吃点辣的东西")
main()