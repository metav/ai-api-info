# AI API Info - 聚合器比价与性能监测平台

## 项目目标
打造 AI API 聚合器层的**实时比价 + 延迟监测 + 稳定性追踪**平台。

## 监测目标：AI API 聚合器/网关清单

### 🥇 第一梯队（必须覆盖）
| 聚合器 | 官网 | 价格页面 | 特点 | API获取价格 |
|--------|------|----------|------|-------------|
| **OpenRouter** | openrouter.ai | /models (有API) | 最大的聚合器，300+模型，多provider路由 | ✅ `/api/v1/models` |
| **Together AI** | together.ai | /pricing | 自建算力+第三方，开源模型为主 | ✅ `/v1/models` |
| **Groq** | groq.com | /pricing | 自研LPU芯片，极致速度 | ✅ `/v1/models` |
| **SiliconFlow** | siliconflow.cn | /pricing | 国内领先，自建算力，开源模型 | ✅ API可查 |
| **OhMyGPT** | ohmygpt.com | /models | 面向国内用户，支持支付宝/微信 | ⚠️ 需爬取 |

### 🥈 第二梯队（优先覆盖）
| 聚合器 | 官网 | 特点 |
|--------|------|------|
| **AIML API** | aimlapi.com | 200+模型，开发者友好 |
| **Novita AI** | novita.ai | GPU云+模型API，图像生成强 |
| **Helicone** | helicone.ai | 可观测性网关，带代理功能 |
| **Portkey** | portkey.ai | 企业级AI网关，智能路由 |
| **Unify AI** | unify.ai | 智能路由，自动选最优provider |

### 🥉 第三梯队（后续扩展）
| 聚合器 | 特点 |
|--------|------|
| **ZenMux** | 企业级网关，LLM保险功能 |
| **LiteLLM** | 开源自建网关方案 |
| **Fireworks AI** | 高速推理，自研基础设施 |
| **Deepinfra** | GPU云+推理服务 |
| **Replicate** | 模型即服务，按次计费 |

## 数据采集方案

### Phase 1: 价格采集
- OpenRouter: 官方 API `/api/v1/models` 返回完整价格
- Together AI: 官方 API 返回模型列表+价格
- Groq: 官方 API
- 其他: 爬取价格页面 或 查文档

### Phase 2: 延迟拨测
- 从多地区发送标准化请求，记录 TTFT 和总延迟

### Phase 3: 稳定性监测
- 定时请求，记录成功率/错误率/超时率

## 技术栈（建议）
- 采集: Python (requests/aiohttp)
- 存储: SQLite → PostgreSQL
- 展示: 静态网页 (Next.js/Astro) 或 GitHub Pages
- 调度: cron / GitHub Actions
