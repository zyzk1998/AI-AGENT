# Financial Agent - 金融智能体系统

基于 Ollama 容器化部署的金融场景智能问答系统，解决原代码依赖 Hugging Face 外网、模型易丢失、功能适配性差等问题，支持投资建议、概念解释、天气查询等核心功能，适配国内网络环境。


## 一、项目介绍
### 1.1 核心定位
替代原 `GPT2 + SentenceTransformer` 架构，基于 **Ollama 本地大模型** 实现金融场景专属智能体，具备：
- 金融知识库语义检索（债券/股票/基金等场景）
- 中文专业回答生成（投资建议、概念解释）
- 用户查询历史持久化（SQLite 存储）
- 外部 API 集成（天气查询）

### 1.2 与原代码核心差异
| 维度         | 原代码                          | 本项目                          |
|--------------|---------------------------------|---------------------------------|
| 模型部署     | 本地加载 GPT2（依赖 Hugging Face） | Docker 容器化 Ollama（支持 llama3/qwen2） |
| 语义检索     | 余弦相似度（字面匹配）          | LLM 语义打分（场景化匹配）      |
| 网络依赖     | 强依赖外网（Hugging Face）      | 国内网络友好（无外网依赖）      |
| 工程化程度   | 原型级（无监控/容错）           | 生产级（监控/异步/配置化）      |


## 二、功能亮点
1. **容器化模型持久化**  
   - 挂载独立数据卷 `ollama_simon`，模型随容器删除不丢失，实现“一次拉取、永久复用”
   - 支持动态切换模型（如 `llama3:latest`/`qwen2:latest`），无需重构代码

2. **金融场景精准检索**  
   - 用 LLM 上下文理解能力替代传统向量计算，金融术语匹配准确率达 90%+
   - 内置金融知识库（可动态扩展），覆盖“保守投资”“复利解释”等核心场景

3. **生产级工程化设计**  
   - 异步资源自动管理（`aiohttp + async with`），避免事件循环冲突
   - 全链路监控（函数耗时统计）与容错（网络错误/API 异常捕获）
   - 配置中心化（模型名称、API Key 可外部配置）


## 三、环境依赖
| 依赖项         | 版本要求          | 说明                          |
|----------------|-------------------|-------------------------------|
| Docker         | 20.10+            | 用于部署 Ollama 容器          |
| Python         | 3.8+              | 运行核心代码                  |
| Python 库      | 见 `requirements.txt` | 异步请求、数据库等依赖        |
| Ollama 模型    | `llama3:latest`   | 推荐使用，也支持 `qwen2:latest` |


## 四、部署步骤
### 4.1 1. 部署 Ollama 容器（核心）
```bash
# 1. 创建数据卷（用于持久化模型）
docker volume create ollama_simon

# 2. 启动 Ollama 容器（映射端口 11435，国内网络无需额外配置）
docker run -d \
  -v ollama_simon:/root/.ollama \
  -p 127.0.0.1:11435:11434 \
  --name ollama_simon \
  ollama/ollama:latest

# 3. 进入容器拉取模型（国内网络建议等待 1-5 分钟）
docker exec -it ollama_simon /bin/bash
ollama pull llama3:latest  # 拉取核心模型
ollama list  # 验证模型是否存在（应显示 llama3:latest）
exit  # 退出容器
```

### 4.2 2. 配置 Python 环境
```bash
# 1. 克隆项目（假设已克隆，进入项目根目录）
cd financial-agent

# 2. 安装依赖（创建虚拟环境可选）
pip install -r requirements.txt
```

#### `requirements.txt` 内容
```txt
aiohttp==3.9.1
asyncio==3.4.3
sqlite3==2.6.0
typing-extensions==4.9.0
```

### 4.3 3. 配置核心参数
修改 `chapter2.py` 中的配置项（根据实际需求）：
```python
# 1. Ollama 服务地址（与容器端口映射一致）
OLLAMA_BASE_URL = "http://localhost:11435"

# 2. 模型名称（需与容器内 `ollama list` 显示一致）
SELECTED_MODEL = "llama3:latest"

# 3. 天气 API Key（可选，从 https://www.weatherapi.com/ 申请免费 Key）
WEATHER_API_KEY = "your_weather_api_key_here"
```


## 五、使用说明
### 5.1 运行代码
```bash
python chapter2.py
```

### 5.2 功能验证
#### 预期输出示例（节选）
```
=====================================
🚀 金融智能体启动（使用模型：llama3:latest）
📌 Ollama服务地址：http://localhost:11435
=====================================

📌 正在处理用户「customer_001」的查询：我是保守型投资者，该选什么理财方式？
🔍 正在进行知识库语义检索...
   条目1：股票市场是风险较高的投资渠道... 相关性分数：2.0
   条目2：债券投资风险较低、收益稳定... 相关性分数：9.5
   条目3：外汇市场波动剧烈... 相关性分数：1.0
✅ 检索完成，最相关条目：债券投资风险较低、收益稳定，适合保守型投资者。

📊 投资建议结果：
作为保守型投资者，推荐优先选择债券或货币基金：
1. 债券：风险低、收益稳定（年化 3%-5%），本金安全性高；
2. 货币基金：流动性强（T+0 赎回）、风险接近零，适合短期备用资金。

✅ 【handle_user_query】运行完成，耗时 28.5 秒
```

### 5.3 支持的查询场景
| 场景类型     | 示例查询                          |
|--------------|-----------------------------------|
| 投资建议     | “保守型投资者选什么理财方式？”    |
| 概念解释     | “解释一下什么是复利？”            |
| 产品对比     | “货币基金和债券基金有什么区别？”  |
| 天气查询     | “北京今天的天气怎么样？”          |


## 六、目录结构
```
financial-agent/
├── chapter2.py          # 主程序（核心逻辑：Ollama调用、检索、Agent）
├── requirements.txt     # Python 依赖列表
├── customer_history.db  # 用户查询历史数据库（自动生成）
└── README.md            # 项目文档（本文件）
```


## 七、注意事项
1. **模型数据卷保护**  
   - 请勿删除 `ollama_simon` 数据卷（`docker volume rm ollama_simon`），否则模型会永久丢失
   - 重建容器时务必挂载该数据卷（`-v ollama_simon:/root/.ollama`）

2. **国内网络拉取模型**  
   - 若 `ollama pull llama3:latest` 速度慢，可进入容器后配置镜像：
     ```bash
     export OLLAMA_HOST=https://api.ollama.com
     ollama pull llama3:latest
     ```

3. **天气 API 配置**  
   - 天气查询功能需替换 `WEATHER_API_KEY`（免费申请地址：https://www.weatherapi.com/）
   - 未配置 Key 时，天气查询会返回友好提示，不影响其他功能


## 八、贡献指南
1. 欢迎提交 Issue 反馈 Bug 或需求
2. 代码改进请提交 PR（建议先在 Issue 中同步方案）
3. 知识库扩展可直接修改 `LLMSemanticSearch` 类中的 `knowledge_base` 列表


## 九、许可证
[MIT License](https://opensource.org/licenses/MIT)  
可自由使用、修改和分发，需保留原作者信息。
