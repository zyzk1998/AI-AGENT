import time
import sqlite3
import asyncio
import aiohttp
from functools import wraps
from typing import Dict, List, Callable, Any

# 全局配置：本地Ollama服务地址（与Docker端口映射一致）
OLLAMA_BASE_URL = "http://localhost:11435"
# 选择容器内已有的模型（必须与 `ollama list` 显示一致）
SELECTED_MODEL = "llama3:latest"
# 天气API Key（需替换为你的真实Key，否则天气查询功能不可用）
WEATHER_API_KEY = "your_weather_api_key_here"

# ------------------------------
# 1. 记忆管理模块（存储用户查询历史）
# ------------------------------
class MemoryManager:
    """管理用户查询历史的数据库模块"""
    def __init__(self, db_path: str = "customer_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                query TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def save_query(self, customer_id: str, query: str):
        """保存用户查询到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customer_queries (customer_id, query)
            VALUES (?, ?)
        """, (customer_id, query))
        conn.commit()
        conn.close()

    def get_last_query(self, customer_id: str) -> str:
        """获取用户最后一次查询"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query FROM customer_queries
            WHERE customer_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (customer_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "未查询到历史记录"

# 初始化记忆管理器（全局单例）
memory_manager = MemoryManager()

# ------------------------------
# 2. 性能监控装饰器（统计函数运行时间）
# ------------------------------
def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = round(time.time() - start_time, 2)
        print(f"\n✅ 【{func.__name__}】运行完成，耗时 {elapsed} 秒")
        return result
    return wrapper

# ------------------------------
# 3. Ollama服务调用模块（核心：连接本地Docker内的LLM）
# ------------------------------
class OllamaManager:
    """管理与本地Ollama Docker服务的交互"""
    def __init__(self, model_name: str = SELECTED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.session = None  # 异步会话延迟初始化

    async def __aenter__(self):
        """通过 async with 自动创建会话"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """通过 async with 自动关闭会话"""
        await self.close()

    async def close(self):
        """关闭异步会话"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """
        调用Ollama生成中文回答
        :param prompt: 用户输入的提示词
        :param max_tokens: 最大生成token数（控制回答长度）
        :return: 中文回答文本
        """
        if not self.session:
            return "❌ Ollama会话未初始化，无法生成回答"

        # 强制添加中文指令，确保输出为中文
        chinese_prompt = f"""
        请用简洁、专业的中文回答以下问题，避免使用英文。
        若问题涉及投资建议，需基于保守、稳健的原则；若涉及概念解释，需通俗易懂。
        
        问题：{prompt}
        """

        try:
            response = await self.session.post(
                url=f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": chinese_prompt,
                    "max_tokens": max_tokens,
                    "stream": False,  # 关闭流式输出，适合批量处理
                    "temperature": 0.7  # 控制随机性（0.7为平衡值）
                }
            )
            response.raise_for_status()  # 触发HTTP错误（如404、500）
            data = await response.json()
            return data.get("response", "❌ 未获取到模型回答，请检查Ollama服务")

        except aiohttp.ClientError as e:
            return f"❌ 调用Ollama失败：网络错误（{str(e)}）"
        except Exception as e:
            return f"❌ 调用Ollama异常：{str(e)}"

# ------------------------------
# 4. LLM语义检索模块（替代原sentence-transformers）
# ------------------------------
class LLMSemanticSearch:
    """使用LLM判断语义相关性，实现知识库检索"""
    def __init__(self, ollama_manager: OllamaManager):
        self.ollama_manager = ollama_manager
        # 金融知识库（可根据需求扩展）
        self.knowledge_base = [
            "股票市场是风险较高的投资渠道，适合能承受短期波动的投资者。",
            "债券投资风险较低、收益稳定，适合保守型投资者。",
            "外汇市场波动剧烈，对专业知识要求高，不适合新手。",
            "货币基金流动性强、风险极低，适合存放短期备用资金。",
            "指数基金通过跟踪大盘分散风险，适合长期定投。"
        ]

    async def _score_relevance(self, query: str, knowledge: str) -> float:
        """
        让LLM给“查询-知识库条目”的相关性打分（0-10分）
        :param query: 用户查询
        :param knowledge: 知识库条目
        :return: 相关性分数（0=完全不相关，10=高度相关）
        """
        score_prompt = f"""
        请仅返回一个0-10的数字，用于表示“用户查询”与“知识库条目”的语义相关性：
        - 0分：完全不相关（如查询天气 vs 投资知识）
        - 5分：部分相关（如查询“短期理财” vs “货币基金”）
        - 10分：高度相关（如查询“保守投资” vs “债券投资”）
        
        用户查询：{query}
        知识库条目：{knowledge}
        相关性分数：
        """

        score_str = await self.ollama_manager.generate_text(score_prompt, max_tokens=10)
        try:
            # 提取数字（处理可能的多余字符，如“分数：8”→8）
            score = float([c for c in score_str if c.isdigit() or c == "."][0])
            return max(0.0, min(10.0, score))  # 限制分数在0-10之间
        except:
            return 3.0  # 解析失败时返回默认分数

    async def search(self, query: str) -> str:
        """
        检索知识库中与查询最相关的条目
        :param query: 用户查询
        :return: 最相关的知识库条目
        """
        if not self.knowledge_base:
            return "❌ 知识库为空，无法检索"

        print("\n🔍 正在进行知识库语义检索...")
        # 批量计算每个条目的相关性分数
        relevance_scores = []
        for idx, knowledge in enumerate(self.knowledge_base, 1):
            score = await self._score_relevance(query, knowledge)
            relevance_scores.append((score, knowledge))
            print(f"   条目{idx}：{knowledge[:20]}... 相关性分数：{score:.1f}")

        # 返回分数最高的条目
        best_match = max(relevance_scores, key=lambda x: x[0])[1]
        print(f"✅ 检索完成，最相关条目：{best_match}")
        return best_match

# ------------------------------
# 5. 外部API调用模块（天气查询）
# ------------------------------
class APIManager:
    """管理外部API调用（如天气查询）"""
    @staticmethod
    async def get_weather(city: str) -> str:
        """
        调用天气API获取城市温度（需替换WEATHER_API_KEY）
        :param city: 城市名称（如“北京”“上海”）
        :return: 天气信息
        """
        if WEATHER_API_KEY == "your_weather_api_key_here":
            return "❌ 天气查询功能未启用：请替换代码中的 WEATHER_API_KEY（可从weatherapi.com获取免费Key）"

        try:
            async with aiohttp.ClientSession() as session:
                response = await session.get(
                    url=f"http://api.weatherapi.com/v1/current.json",
                    params={
                        "key": WEATHER_API_KEY,
                        "q": city,
                        "aqi": "no"  # 不返回空气质量数据
                    }
                )
                response.raise_for_status()
                data = await response.json()
                temp_c = data["current"]["temp_c"]
                condition = data["current"]["condition"]["text"]
                return f"🌤️ {city}当前天气：{condition}，气温 {temp_c}℃"
        except Exception as e:
            return f"❌ 天气查询失败：{str(e)}"

# ------------------------------
# 6. 核心智能体模块（整合所有功能）
# ------------------------------
class FinancialAgent:
    """金融智能体：整合记忆、LLM、检索、API功能"""
    def __init__(self, ollama_manager: OllamaManager, semantic_search: LLMSemanticSearch):
        self.ollama_manager = ollama_manager
        self.semantic_search = semantic_search

    @performance_monitor
    async def handle_user_query(self, customer_id: str, query: str) -> None:
        """
        处理用户查询的主入口
        :param customer_id: 用户ID（用于记忆跟踪）
        :param query: 用户输入的查询文本
        """
        # 1. 保存查询历史
        memory_manager.save_query(customer_id, query)
        print(f"\n📌 正在处理用户「{customer_id}」的查询：{query}")

        # 2. 分支1：天气查询
        if any(keyword in query for keyword in ["天气", "气温", "温度"]):
            # 提取城市名（简单规则：取查询最后2-3个汉字，如“北京天气”→“北京”）
            city = "".join([c for c in query if '\u4e00' <= c <= '\u9fff'])[-3:] or "北京"
            weather_info = await APIManager.get_weather(city)
            print(f"\n📊 天气查询结果：{weather_info}")

        # 3. 分支2：投资相关查询（触发知识库检索）
        elif any(keyword in query for keyword in ["投资", "理财", "基金", "债券", "股票"]):
            # 检索知识库最相关条目
            relevant_knowledge = await self.semantic_search.search(query)
            # 基于知识库生成增强回答
            final_answer = await self.ollama_manager.generate_text(
                prompt=f"基于以下知识库信息，回答用户问题：\n知识库：{relevant_knowledge}\n用户问题：{query}"
            )
            print(f"\n📊 投资建议结果：\n{final_answer}")

        # 4. 分支3：通用问题（直接调用LLM）
        else:
            general_answer = await self.ollama_manager.generate_text(prompt=query)
            print(f"\n📊 通用问题回答：\n{general_answer}")

# ------------------------------
# 7. 主程序入口
# ------------------------------
async def main():
    print(f"=====================================")
    print(f"🚀 金融智能体启动（使用模型：{SELECTED_MODEL}）")
    print(f"📌 Ollama服务地址：{OLLAMA_BASE_URL}")
    print(f"=====================================")

    # 初始化核心组件（通过async with自动管理会话生命周期）
    async with OllamaManager() as ollama_manager:
        semantic_search = LLMSemanticSearch(ollama_manager)
        agent = FinancialAgent(ollama_manager, semantic_search)

        # 测试用例（可替换为实际用户查询）
        test_cases = [
            ("customer_001", "我是保守型投资者，该选什么理财方式？"),
            ("customer_001", "解释一下什么是复利？"),
            ("customer_002", "北京今天的天气怎么样？"),
            ("customer_002", "货币基金和债券基金有什么区别？")
        ]

        # 执行测试用例
        for customer_id, query in test_cases:
            await agent.handle_user_query(customer_id, query)
            print("\n" + "-"*50)  # 分隔符

    print("✅ 所有查询处理完成，程序退出")

if __name__ == "__main__":
    # 解决Windows/Linux异步事件循环差异
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # 适配Jupyter等已有事件循环的环境
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
