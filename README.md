# NewsPulse — 多视角新闻推荐系统 Demo

> *"Breaking your filter bubble, one headline at a time."*

NewsPulse 是一个打破信息茧房的智能新闻推荐系统演示项目。它展示同一新闻事件在 **左派（Progressive）、中立（Center）、右派（Traditional）** 三种政治倾向下的不同报道框架，帮助用户建立多元化的信息消费习惯。

---

## 项目简介

### 基本功能

- **多视角新闻浏览**：每篇文章同时呈现三种政治倾向的报道版本，让用户对比不同立场的叙事差异
- **个性化推荐**：基于用户阅读偏好，从 MIND 数据集中智能匹配相关内容
- **重排序引擎**：支持 MMR（Maximal Marginal Relevance）等多样性增强策略，在相关性与信息多样性间取得平衡
- **冷启动推荐**：4 阶段渐进式策略（静态画像 → Bandit 探索 → 兴趣捕捉 → 完全个性化），解决新用户推荐问题
- **阅读平衡度仪表盘**：实时可视化 P/B/T 分布，鼓励多维度新闻消费
- **AI 驱动内容分析**：通过 DeepSeek 大模型自动完成新闻摘要、立场分类与可信度评估

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | 原生 HTML/CSS/JS | 零框架 SPA，108KB 单文件应用 |
| 后端 A | Python 3.11 + Flask | 完整 API，NumPy 向量计算 |
| 后端 B | Node 18 + Express | 同功能移植，纯 JS 手工实现向量运算 |
| 数据 | MIND 数据集 | 微软新闻推荐数据集（5万+ 文章、15万+ 行为） |
| 模型 | PyTorch | NRMS / NAML / LSTUR / NPA 四种深度推荐模型 |
| AI | DeepSeek Chat API | 新闻摘要、立场分析、可信度评估 |
| 部署 | Docker + Nginx | 多方案容器化部署，支持 Lighthouse |

### 核心特性

- **双后端可互换**：Python/Flask 与 Node.js/Express 两套完全对等的后端，可按环境灵活切换
- **实体级 MMR 重排序**：基于新闻实体（人物、地名、机构等）的多样性优化，奖励引入新信息的文章
- **三种 Bandit 算法**：冷启动支持 Epsilon-Greedy / UCB1 / Thompson Sampling 策略
- **无框架依赖**：前端纯原生 JS，Node.js 后端仅依赖 Express + CORS
- **完整 ML 训练管线**：超参数网格搜索、早停、模型集成、checkpoint 管理

---

## Demo 目的

### 演示目标

1. **展示"去信息茧房"的产品理念**：通过并排对比同一事件的多视角报道，让用户直观感受不同立场媒体的叙事差异
2. **验证 MMR 多样性重排序的实际效果**：对比"纯相关性排序"与"多样性增强排序"在推荐质量和信息广度上的差异
3. **演示冷启动推荐流程**：模拟新用户注册 → 偏好选择 → 逐步个性化推荐的完整路径
4. **提供一个可落地的新闻推荐系统原型**：包含数据管道、模型训练、API 服务、前端展示的完整闭环

### 核心逻辑

```
用户注册/选择 → 生成基准推荐（协同过滤 + 内容匹配）
                        ↓
              MMR 多样性重排序（实体/语义/校准策略）
                        ↓
              冷启动干预（4 阶段渐进策略）
                        ↓
              多视角内容增强（P/B/T 三版本）
                        ↓
              前端渲染 & 平衡度仪表盘
                        ↓
              用户交互反馈 → 闭环优化
```

### 预期效果

- 用户能够在同一页面阅读同一事件在 Progressive、Center、Traditional 三种立场下的不同报道
- 阅读平衡度仪表盘实时反映用户的信息消费倾向，引导打破过滤气泡
- MMR 重排序确保推荐列表在保持相关性的同时具备充分的信息多样性
- 冷启动用户在 8+ 次交互后获得与存量用户一致的个性化推荐体验

---

## 快速开始

### 环境要求

- Python 3.11+ 或 Node.js 18+
- Docker（可选，用于容器化部署）

### 本地运行

```bash
# 方式一：Python/Flask 后端
pip install flask flask-cors numpy
python backend/server.py
# 访问 http://localhost:8096

# 方式二：Node.js 后端
cd backend-node && npm install && npm start
# 访问 http://localhost:8097

# 方式三：开发代理（静态文件 + API 代理）
python dev_proxy.py
# 访问 http://localhost:8080
```

### Docker 部署

```bash
# Python + Nginx + Supervisor
docker build -t news-reco -f Dockerfile .
docker run -d -p 80:80 news-reco

# Node.js 独立部署
docker build -t news-backend-node -f Dockerfile.node .
docker run -d -p 8097:8097 news-backend-node
```

---

## 数据管道

```bash
# 配置 .env 文件中的 API 密钥
# DEEPSEEK_API_KEY=your_key
# NEWSAPI_KEY=your_key

# 运行数据抓取与分析管道
python run_pipeline.py
# 或
python news_fetcher/pipeline.py --interval 7 --max-articles 50
```

---

## 项目结构

```
.
├── index.html                  # Demo 主应用（SPA）
├── landing.html                # 产品首页
├── backend/                    # Python/Flask 后端
│   ├── server.py               # API 服务（18 个端点）
│   ├── mind_adapter.py         # MIND 数据集适配器
│   ├── reranker.py             # MMR 多样性重排序
│   └── coldstart.py            # 冷启动推荐引擎
├── backend-node/               # Node.js/Express 后端（同功能移植）
│   ├── server.js               # Express API 服务
│   ├── data.js                 # 数据加载层
│   ├── reranker.js             # 纯 JS 重排序引擎
│   └── coldstart.js            # 纯 JS 冷启动引擎
├── news_fetcher/               # AI 数据管道
│   ├── pipeline.py             # 管道编排器
│   ├── news_scraper.py         # 新闻抓取器
│   ├── deepseek_client.py      # DeepSeek API 客户端
│   └── frontend_formatter.py   # 前端格式转换
├── Recommender-Project-MIND/   # ML 模型训练
│   └── mind_rec/               # NRMS / NAML / LSTUR / NPA
├── deploy_pkg/                 # 生产部署包
├── Dockerfile                  # Python 部署方案
├── Dockerfile.node             # Node.js 部署方案
└── nginx.conf                  # Nginx 反向代理配置
```
