# 模块4：智能体应用开发 — 基础知识

> 面向金砖赛"人工智能应用创新"赛道——智能体应用开发入门。

## 目录

1. [什么是 OpenClaw Skill](#1-什么是-openclaw-skill)
   - [1.1 定义](#11-定义)
   - [1.2 核心设计理念](#12-核心设计理念)
   - [1.3 文件结构](#13-文件结构)
   - [1.4 技术要点](#14-技术要点)
   - [1.5 SKILL.md 规范说明](#15-skillmd-规范说明)
   - [1.6 入门案例：天气查询 Skill](#16-入门案例天气查询-skill)
2. [Python + OpenClaw 控制](#2-python--openclaw-控制)
   - [2.1 定义](#21-定义)
   - [2.2 整体架构](#22-整体架构)
   - [2.3 核心技术点](#23-核心技术点)
   - [2.4 技术要点](#24-技术要点)
   - [2.5 入门案例：文件批量处理脚本](#25-入门案例文件批量处理脚本)
3. [Dify 智能体](#3-dify-智能体)
   - [3.1 什么是 Dify](#31-什么是-dify)
   - [3.2 Dify 工作流架构](#32-dify-工作流架构)
   - [3.3 关键组件](#33-关键组件)
   - [3.4 技术要点](#34-技术要点)
   - [3.5 入门案例：智能翻译工作流](#35-入门案例智能翻译工作流)
4. [LangChain/LangGraph 智能体](#4-langchainlanggraph-智能体)
   - [4.1 什么是 LangChain/LangGraph](#41-什么是-langchainlanggraph)
   - [4.2 LangChain vs LangGraph](#42-langchain-vs-langgraph)
   - [4.3 为什么用 LangGraph 构建智能体](#43-为什么用-langgraph-构建智能体)
   - [4.4 LangGraph 核心概念](#44-langgraph-核心概念)
   - [4.5 技术要点](#45-技术要点)
   - [4.6 入门案例：计算器智能体](#46-入门案例计算器智能体)
5. [四种技能对比与学习路径](#5-四种技能对比与学习路径)
   - [5.1 四种技能对比总结](#51-四种技能对比总结)
   - [5.2 学习路径建议](#52-学习路径建议)

---

## 1. 什么是 OpenClaw Skill

### 1.1 定义

**OpenClaw Skill** 是一种由 **YAML 元数据 + Markdown 正文** 组成的纯文本文件（`SKILL.md`），存放在 `skills/<skill-name>/` 目录下。它充当智能体的"技能说明书"，定义了：

- 技能能做什么（name、description）
- 需要什么运行环境（requires）
- 用户自然语言到 Shell 命令的映射规则

OpenClaw 启动时会自动扫描 `skills/` 目录，加载全部 SKILL.md 文件。当用户输入自然语言时，大模型根据 Skill 内容选择最匹配的命令来执行。

> **关键理解**：Skill 文件本身不执行操作，它只是"说明书"——告诉智能体遇到某种用户意图时应该执行哪条命令。真正的执行由 Shell 脚本和程序完成。

### 1.2 核心设计理念

整个链路分为三层——**理解层**（自然语言→命令）→**封装层**（Shell 初始化环境）→**执行层**（Python 业务逻辑），各层职责清晰分离。

![OpenClaw Skill 三层架构](images/2082034950332592130.png)

### 1.3 文件结构

```
skills/
└── weather-query/               ← 技能目录名
    └── SKILL.md                 ← 唯一必要的文件
    └── references/              ← 可选：参考文档、图片等
```

### 1.4 技术要点

| 要点 | 说明 |
|------|------|
| **YAML 元数据** | 必须包含 `name`、`description`；可选 `metadata.openclaw.requires` 声明运行依赖 |
| **指令映射表** | 核心价值——将"查天气/算算术/读文件"等自然语言意图映射到具体 shell 命令 |
| **自然语言匹配规则** | Skill 中用 `**用户意图** → 执行命令` 的格式描述，大模型据此匹配 |
| **环境依赖声明** | `requires.env` 确保所需环境变量已设置（如 API Key） |
| **路径规范** | 统一使用 `/workspace/scripts/...` 绝对路径，避免相对路径歧义 |
| **失败提示** | Skill 中应包含常见错误提示，帮助用户排错 |

### 1.5 SKILL.md 规范说明

OpenClaw 遵循 [AgentSkills](https://agentskills.io) 规范，每个 Skill 是一个文件夹，核心文件为 `SKILL.md`（Markdown + YAML Frontmatter）。

#### 文件结构

```
skills/<skill-name>/
├── SKILL.md              ← 必需：技能定义文件（YAML + Markdown）
├── .clawhubignore        ← 可选：发布忽略规则
├── .gitignore            ← 可选：同样被遵守
└── references/           ← 可选：参考文档、图片等辅助文件
```

#### YAML Frontmatter 完整字段参考

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `name` | string | ✅ | 技能名称，小写字母+数字+连字符，如 `weather-query` |
| `description` | string | ✅ | 技能简述，在 UI/搜索中作为摘要显示 |
| `version` | string | | 语义化版本号，如 `1.0.0` |
| `metadata.openclaw.requires.env` | string[] | | 运行前必须存在的环境变量 |
| `metadata.openclaw.requires.bins` | string[] | | 运行前必须安装的 CLI 二进制文件 |
| `metadata.openclaw.requires.anyBins` | string[] | | 至少一个必须存在的 CLI 二进制文件 |
| `metadata.openclaw.requires.config` | string[] | | 技能读取的配置文件路径 |
| `metadata.openclaw.primaryEnv` | string | | 主要凭证环境变量名（用于安全扫描匹配） |
| `metadata.openclaw.envVars` | array | | 环境变量声明列表（含 `name`、`required`、`description`） |
| `metadata.openclaw.always` | boolean | | 设为 `true` 时始终激活，无需显式安装 |
| `metadata.openclaw.emoji` | string | | 技能显示图标 |
| `metadata.openclaw.homepage` | string | | 技能主页/文档 URL |
| `metadata.openclaw.os` | string[] | | 操作系统限制，如 `["linux"]`、`["macos"]` |
| `metadata.openclaw.install` | array | | 依赖安装规格（支持 `brew`/`node`/`go`/`uv`） |

#### 完整示例：标准 SKILL.md 模板

````markdown
---
name: <skill-name>
description: <一句话描述技能功能>
version: 1.0.0
metadata:
  openclaw:
    requires:
      env:
        - <REQUIRED_ENV_VAR>
      bins:
        - <required-binary>
    primaryEnv: <MAIN_CREDENTIAL_VAR>
    emoji: "🔧"
    homepage: https://github.com/example/<skill-name>
---

# <技能标题>

## 前置条件

- 已配置 `<REQUIRED_ENV_VAR>` 环境变量
- 系统已安装 `<required-binary>`

## 使用方法

本技能通过以下 Shell 命令执行：

```bash
<command-example>
```

## 命令映射表

| 用户意图 | 执行命令 | 效果 |
|----------|----------|------|
| <意图A> | `<command-a>` | <效果描述> |
| <意图B> | `<command-b>` | <效果描述> |

## 自然语言 → 命令规则

当用户说出以下意图时，按规则选择命令：

- **包含"<关键词>"** → 执行 `<command>`
- **默认行为** → <默认处理>

## 注意事项

- <错误处理提示 1>
- <错误处理提示 2>
````

#### 加载流程

OpenClaw 从多个来源加载 Skills（**高优先级同名覆盖低优先级**），经门控过滤和 Allowlist 控制后注入运行时环境。

![OpenClaw Skill 加载流程](images/2082034950601027585.png)

> **安全提醒**：将第三方 Skill 视为不受信任的代码。frontmatter 中声明的 `requires` 会被安全扫描校验——代码中引用的环境变量必须在 frontmatter 中声明，否则会标记为元数据不匹配。

### 1.6 入门案例：天气查询 Skill

**场景**：用户对 OpenClaw 说"今天北京天气怎么样"，智能体自动查询并返回天气。

**`skills/weather-query/SKILL.md`**

````markdown
---
name: weather-query
description: >-
  查询指定城市的天气信息。
  支持自然语言提问。
metadata:
  openclaw:
    emoji: "🌤️"
    requires:
      env:
        - WEATHER_API_KEY
---

# 天气查询

## 前置条件

- 已配置 `WEATHER_API_KEY` 环境变量
- Python 3 已安装，`requests` 库可用

## 使用方法

```bash
python3 /workspace/scripts/weather.py "<城市名>"
```

### 命令映射表

| 用户意图 | 执行命令 | 效果 |
|----------|----------|------|
| 查天气 / 今天天气 | `python3 scripts/weather.py "北京"` | 返回指定城市天气 |
| 天气预报 | `python3 scripts/weather.py "上海" --forecast 3` | 查询未来 3 天预报 |

### 自然语言 → 命令规则

当用户说出以下意图时，按规则选择命令：

- **包含"天气/气温"** → 提取城市名，执行 `python3 scripts/weather.py "<城市>"`
- **包含"预报/未来几天"** → 提取天数，执行 `python3 scripts/weather.py "<城市>" --forecast <天数>`
- **未指定城市** → 默认查询"北京"

## 注意事项

- 若 API 调用失败，提示用户检查 API Key 是否正确
- 若未安装 requests 库，提示 `pip install requests`
````

> **设计原则**：职责分离——Skill = 说明书，Shell = 启动器，Python = 业务逻辑。

---

## 2. Python + OpenClaw 控制

### 2.1 定义

Python 程序是整个智能体系统的**核心执行层**。智能体通过 OpenClaw Skill 调用 Shell 脚本，Shell 脚本加载运行环境并调用 Python 程序，Python 程序完成实际的业务逻辑处理。本模块通过编写 Python 脚本，掌握命令行参数解析、外部 API 调用、文件 I/O 等核心能力。

### 2.2 整体架构

```
用户输入
    │
    ▼
OpenClaw Skill ──► 匹配意图 → 选择命令
    │
    ▼
Shell 脚本 ──► 初始化环境变量、激活虚拟环境
    │
    ▼
Python 程序 ──► 解析参数 → 执行逻辑 → 输出结果
```

### 2.3 核心技术点

| 概念 | 说明 |
|------|------|
| **命令行参数解析** | 使用 `sys.argv` 或 `argparse` 接收用户输入 |
| **API 调用** | 使用 `requests` 库调用外部 HTTP API |
| **文件 I/O** | 读写本地文件，如批量重命名、内容搜索替换 |
| **进程调用** | 使用 `subprocess` 调用外部命令 |

### 2.4 技术要点

| 要点 | 说明 |
|------|------|
| **参数解析** | `sys.argv` 适合简单场景，`argparse` 适合复杂命令行工具 |
| **错误处理** | 使用 `try/except` 捕获异常，给出友好提示而非裸露报错 |
| **返回值** | 程序用 `sys.exit(0)` 表示成功，`sys.exit(1)` 表示失败 |
| **输出格式** | 终端输出用 `print`，结构化数据用 JSON 输出便于下游解析 |
| **环境变量** | 敏感信息（如 API Key）通过环境变量传入，不硬编码 |

### 2.5 入门案例：文件批量处理脚本

**场景**：编写一个 Python 脚本，支持批量重命名文件、统计文件信息。

**`scripts/file_tool.py`**

```python
#!/usr/bin/env python3
"""
file_tool.py — 文件批量处理工具

用法:
  # 批量重命名：将所有 .txt 文件加前缀 "done_"
  python3 file_tool.py rename --dir ./data --type txt --prefix done

  # 统计目录下各类型文件数量
  python3 file_tool.py stats --dir ./data
"""
import sys
import os
import argparse
from pathlib import Path


def rename_files(directory: str, file_type: str, prefix: str):
    """批量重命名：给指定类型文件添加前缀"""
    path = Path(directory)
    if not path.exists():
        print(f"[ERROR] 目录不存在: {directory}")
        sys.exit(1)

    files = list(path.glob(f"*.{file_type}"))
    if not files:
        print(f"[INFO] 未找到 .{file_type} 文件")
        return

    for f in files:
        new_name = f.parent / f"{prefix}_{f.name}"
        f.rename(new_name)
        print(f"  {f.name} → {new_name.name}")

    print(f"[DONE] 共重命名 {len(files)} 个文件")


def count_files(directory: str):
    """统计目录下各类型文件数量"""
    path = Path(directory)
    if not path.exists():
        print(f"[ERROR] 目录不存在: {directory}")
        sys.exit(1)

    stats = {}
    for f in path.iterdir():
        if f.is_file():
            ext = f.suffix or "无后缀"
            stats[ext] = stats.get(ext, 0) + 1

    if not stats:
        print("[INFO] 目录为空")
        return

    print(f"\n📁 {directory} 文件统计:")
    total = 0
    for ext, count in sorted(stats.items()):
        print(f"  {ext:>8}: {count} 个")
        total += count
    print(f"  {'总计':>8}: {total} 个\n")


def main():
    parser = argparse.ArgumentParser(description="文件批量处理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 子命令: rename
    rename_parser = subparsers.add_parser("rename", help="批量重命名")
    rename_parser.add_argument("--dir", required=True, help="目标目录")
    rename_parser.add_argument("--type", default="txt", help="文件类型（如 txt, jpg）")
    rename_parser.add_argument("--prefix", required=True, help="前缀文本")

    # 子命令: stats
    stats_parser = subparsers.add_parser("stats", help="文件统计")
    stats_parser.add_argument("--dir", required=True, help="目标目录")

    args = parser.parse_args()

    if args.command == "rename":
        rename_files(args.dir, args.type, args.prefix)
    elif args.command == "stats":
        count_files(args.dir)


if __name__ == "__main__":
    main()
```

**对应 SKILL.md 映射片段：**

```markdown
### 命令映射表

| 用户意图 | 执行命令 |
|----------|----------|
| 批量重命名 / 加前缀 | `python3 scripts/file_tool.py rename --dir ./data --type txt --prefix done` |
| 统计文件 / 看目录 | `python3 scripts/file_tool.py stats --dir ./data` |
```

---

## 3. Dify 智能体

### 3.1 什么是 Dify

**Dify** 是一个**低代码 LLMOps 平台**，提供可视化的 AI 应用开发界面。它降低了大模型应用开发的门槛，让非专业程序员也能通过拖拽的方式构建智能体。

| 特性 | 说明 |
|------|------|
| **可视化编排** | 拖拽节点搭建工作流，无需写前端代码 |
| **LLM 集成** | 支持 OpenAI、通义千问、DeepSeek 等大模型 |
| **工具/插件** | 支持自定义 API 工具、代码执行节点 |
| **知识库** | 可上传文档作为 RAG 知识源 |
| **一键发布** | 内置 Web 应用界面，可直接分享使用 |

### 3.2 Dify 工作流架构

![Dify Chatflow 工作流架构](images/2082034950869463041.png)

### 3.3 关键组件

| 组件 | 作用 |
|------|------|
| **Chatflow 工作流** | Dify 核心编排单元，定义"用户输入→LLM→工具→输出"的完整链路 |
| **LLM 节点** | 调用大模型进行意图识别、推理、生成回复 |
| **HTTP 请求节点** | 向外部 API 发送请求，扩展智能体能力边界 |
| **代码节点** | 在 Dify 沙箱中执行 Python/JS 代码，用于数据预处理 |
| **知识库** | 上传参考资料，让 LLM 自动检索并用于回答 |
| **变量** | 工作流中传递的数据，如 `{{query}}`、`{{result}}` |

### 3.4 技术要点

| 要点 | 说明 |
|------|------|
| **意图识别** | 在 LLM 节点中使用 System Prompt 让模型输出结构化 JSON |
| **工具注册** | 在 Dify 中注册自定义 API 工具，定义其 URL、方法、参数 Schema |
| **变量传递** | 使用 `{{节点名.输出字段}}` 语法在工作流节点间传递数据 |
| **条件分支** | 根据 LLM 输出内容走不同分支（如翻译→调翻译 API，计算→调代码节点） |
| **知识库** | 上传参考文档后，LLM 自动检索相关内容增强回答质量 |

### 3.5 入门案例：智能翻译工作流

**场景**：用户在 Dify 中输入"把'你好世界'翻译成英文"，工作流自动调用翻译 API 返回结果。

#### 工作流编排

```
[开始] → [LLM 节点] → [条件判断] → [HTTP 请求] → [回答]
                         │
                         ├── 意图=translate → 调用翻译 API
                         └── 意图=chat     → 直接回答
```

#### LLM 节点 System Prompt

```
你是一个多功能助手。根据用户输入，输出一个 JSON 对象。

输出格式必须是严格的 JSON：

1. 翻译请求: {"intent": "translate", "text": "你好世界", "target_lang": "英文"}
2. 普通聊天: {"intent": "chat", "reply": "你的回复内容"}

示例:
用户: "把'Hello'翻译成中文" → {"intent": "translate", "text": "Hello", "target_lang": "中文"}
用户: "帮我翻译'I love AI'为日语" → {"intent": "translate", "text": "I love AI", "target_lang": "日语"}
用户: "你好" → {"intent": "chat", "reply": "你好！有什么可以帮你的？"}
```

#### HTTP 请求节点配置

| 配置项 | 值 |
|--------|-----|
| 请求方法 | POST |
| URL | `https://api.example.com/translate` |
| Headers | `Content-Type: application/json` |
| Body | `{"text": "{{LLM.text}}", "target": "{{LLM.target_lang}}"}` |

#### 代码节点（替代 HTTP，用于简单逻辑）

如果不想依赖外部 API，也可以用代码节点实现简单查表：

```python
def main(text: str, target_lang: str) -> dict:
    """简单翻译查表（仅作演示）"""
    lookup = {
        "你好": {"英文": "Hello", "日语": "こんにちは", "韩语": "안녕하세요"},
        "谢谢": {"英文": "Thank you", "日语": "ありがとう", "韩语": "감사합니다"},
        "再见": {"英文": "Goodbye", "日语": "さようなら", "韩语": "안녕히 가세요"},
    }
    result = lookup.get(text, {}).get(target_lang, f"（未找到 '{text}' 的{target_lang}翻译）")
    return {"translated": result}
```

---

## 4. LangChain/LangGraph 智能体

### 4.1 什么是 LangChain/LangGraph

**LangChain** 是一个 LLM 应用开发框架，提供链式调用、工具集成、记忆管理等基础设施。**LangGraph** 是 LangChain 生态系统中的图计算引擎，专为构建**有状态、多角色、支持循环的智能体**而设计。

### 4.2 LangChain vs LangGraph

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| **定位** | LLM 应用开发框架 | 有状态、多角色图计算引擎 |
| **核心抽象** | Chain（链式调用：A→B→C） | StateGraph（状态图：可循环、可分支） |
| **控制流** | 线性：固定顺序执行 | 循环+条件：A→B→C→A（闭环） |
| **记忆** | 会话记忆（ConversationBufferMemory） | 全局 State（TypedDict 定义，节点间共享） |
| **适合场景** | 简单问答、RAG、工具调用 | 多步骤智能体、循环决策 |
| **关系** | LangGraph 是 LangChain 生态的一部分 | 可独立使用或与 LangChain 集成 |

### 4.3 为什么用 LangGraph 构建智能体

智能体的核心是 **思考→行动→观察** 循环（ReAct 模式）：

![LangGraph StateGraph 架构](images/2082034951133704194.png)

选择 LangGraph 的理由：

| 需求 | LangGraph 如何满足 |
|------|-------------------|
| **循环推理** | 天然支持环路（Conditional Edge），Chain 无法实现 |
| **状态持久化** | State 在节点间传递和更新，智能体可"记住"中间结果 |
| **动态路由** | 根据推理结果动态选择下一步（继续计算 / 输出结果） |
| **可中断/可恢复** | 支持 checkpoint，可暂停和恢复推理流程 |
| **节点解耦** | 思考、行动、观察三个节点独立开发和测试 |

### 4.4 LangGraph 核心概念

| 概念 | 说明 | 代码对应 |
|------|------|----------|
| **State** | 全局状态对象，TypedDict 定义，在节点间传递和累加 | `class AgentState(TypedDict)` |
| **Node** | 处理函数，接收 State 并返回 State 的部分更新 | `def think(state) -> dict` |
| **Edge** | 普通边：A 完成后一定去 B | `.add_edge("think", "act")` |
| **Conditional Edge** | 条件边：根据 state 值决定跳转目标 | `.add_conditional_edges("act", router, {...})` |
| **Graph** | 包含所有节点和边的有向图 | `StateGraph(AgentState)` |
| **Compile** | 将图编译为可执行的 Runnable | `.compile(checkpointer=...)` |
| **Checkpoint** | 每次节点执行后自动保存 State 快照 | `MemorySaver()` |

### 4.5 技术要点

| 要点 | 说明 |
|------|------|
| **State 定义** | 用 `TypedDict` 精确定义状态字段和类型，保证类型安全 |
| **节点设计** | 每个节点是纯函数 `(state) → update_dict`，只负责自己的范围 |
| **条件路由** | 根据 state 中的值决定下一步跳转路径 |
| **工具定义** | 使用 LangChain `@tool` 装饰器将普通函数包装为 LLM 可调用的工具 |
| **循环终止** | 条件边判断 `should_continue` → "continue"回到思考，"end"退出 |

### 4.6 入门案例：小乌龟画正方形

**场景**：在 ROS2 的经典 `turtlesim` 仿真环境中，用 LangGraph 控制小乌龟自动画出一个正方形（直走 → 转弯 → 直走 → 转弯……循环 4 次）。

> **前置条件**：终端先启动 `ros2 run turtlesim turtlesim_node`，看到小乌龟窗口后再运行本脚本。

```python
#!/usr/bin/env python3
"""
turtle_agent.py — 基于 LangGraph 的 ROS2 小乌龟控制智能体

运行方式:
    终端1: ros2 run turtlesim turtlesim_node
    终端2: python3 turtle_agent.py

架构:
    START → [perceive] → [decide] → [act] → [条件]
                 ▲                          │
                 └──────── continue ────────┘
                              │
                            end → END
"""
from typing import TypedDict, Annotated
import operator
import math
import time

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


# ============================================================
# 1. ROS2 共享节点（桥接 LangGraph 与 turtlesim）
# ============================================================

_shared_node = None


class TurtleBridge(Node):
    """共享节点：订阅小乌龟位姿、发布速度指令"""

    def __init__(self):
        super().__init__("turtle_bridge")
        self.latest_pose = None
        self.cmd_pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.create_subscription(Pose, "/turtle1/pose", self._pose_cb, 10)

    def _pose_cb(self, msg: Pose):
        self.latest_pose = msg

    def publish_cmd(self, linear: float, angular: float):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)


# ============================================================
# 2. 定义 State
# ============================================================

class TurtleState(TypedDict):
    """小乌龟智能体状态"""
    # 传感器数据
    pose_x: float          # 当前位置 x
    pose_y: float          # 当前位置 y
    pose_theta: float      # 当前朝向 (rad)

    # 任务参数
    side_count: int        # 已完成的边数（目标：4 条边 = 正方形）
    step: str              # 当前阶段: "forward" | "turn"

    # 控制
    linear_x: float        # 当前线速度
    angular_z: float       # 当前角速度
    step_start_time: float # 当前阶段开始时间

    # 日志
    log: Annotated[list[str], operator.add]


# ============================================================
# 3. 配置参数
# ============================================================

CFG = {
    "forward_speed": 2.0,     # 前进速度
    "forward_time": 1.8,      # 前进持续时间（秒），≈ 边长
    "turn_speed": 1.57,       # 转弯角速度 (≈ π/2 rad/s)
    "turn_time": 1.0,         # 转弯持续时间（秒），≈ 转 90°
    "total_sides": 4,         # 总边数
}


# ============================================================
# 4. 节点函数
# ============================================================

def perceive_node(state: TurtleState) -> dict:
    """感知节点：读取小乌龟当前位置"""
    global _shared_node

    if _shared_node is None or _shared_node.latest_pose is None:
        return {"log": ["[PERCEIVE] 等待 turtlesim 数据..."]}

    p = _shared_node.latest_pose
    return {
        "pose_x": p.x,
        "pose_y": p.y,
        "pose_theta": p.theta,
        "log": [f"[PERCEIVE] 位姿 x={p.x:.2f} y={p.y:.2f} θ={math.degrees(p.theta):.0f}°"],
    }


def decide_node(state: TurtleState) -> dict:
    """决策节点：判断该直走还是转弯"""
    side_count = state.get("side_count", 0)

    # 已完成 4 条边 → 停止
    if side_count >= CFG["total_sides"]:
        return {"step": "done", "log": ["[DECIDE] 正方形完成，准备停止"]}

    # 每条边先直走后转弯
    step = state.get("step", "forward")
    if step == "forward":
        return {"step": "turn", "log": [f"[DECIDE] 第 {side_count + 1} 条边 → 直走"]}
    else:
        return {"step": "forward", "log": [f"[DECIDE] 第 {side_count + 1} 条边 → 转弯"]}


def act_node(state: TurtleState) -> dict:
    """行动节点：根据决策发布速度指令"""
    global _shared_node

    step = state.get("step", "forward")
    now = time.time()
    step_start = state.get("step_start_time", now)
    elapsed = now - step_start
    side_count = state.get("side_count", 0)

    twist = Twist()

    if step == "forward":
        if elapsed < CFG["forward_time"]:
            # 还没走够 → 继续直走
            twist.linear.x = CFG["forward_speed"]
            twist.angular.z = 0.0
            _shared_node.publish_cmd(CFG["forward_speed"], 0.0)
            return {
                "log": [f"[ACT] 直走 {elapsed:.1f}s / {CFG['forward_time']}s"],
                "linear_x": CFG["forward_speed"],
                "angular_z": 0.0,
            }
        else:
            # 走够了 → 记录完成一条边
            _shared_node.publish_cmd(0.0, 0.0)
            return {
                "step_start_time": now,         # 重置计时
                "linear_x": 0.0,
                "angular_z": 0.0,
                "log": [f"[ACT] 直走完成 ✓ 边 {side_count + 1}/{CFG['total_sides']}"],
            }

    elif step == "turn":
        if elapsed < CFG["turn_time"]:
            # 还没转够 → 继续转弯
            _shared_node.publish_cmd(0.0, CFG["turn_speed"])
            return {
                "log": [f"[ACT] 转弯 {elapsed:.1f}s / {CFG['turn_time']}s"],
                "linear_x": 0.0,
                "angular_z": CFG["turn_speed"],
            }
        else:
            # 转够了 → 进入下一条边
            _shared_node.publish_cmd(0.0, 0.0)
            return {
                "side_count": side_count + 1,
                "step_start_time": now,         # 重置计时
                "linear_x": 0.0,
                "angular_z": 0.0,
                "log": [f"[ACT] 转弯完成 ✓ 进入边 {side_count + 2}"],
            }

    elif step == "done":
        # 停车
        _shared_node.publish_cmd(0.0, 0.0)
        return {
            "linear_x": 0.0,
            "angular_z": 0.0,
            "log": ["[ACT] 停车，任务完成 ✓"],
        }

    return {}


# ============================================================
# 5. 条件路由
# ============================================================

def route_after_act(state: TurtleState) -> str:
    """行动后：继续循环还是结束"""
    step = state.get("step", "")
    if step == "done":
        return "end"
    return "continue"


# ============================================================
# 6. 构建 StateGraph
# ============================================================

def build_turtle_graph() -> StateGraph:
    """
    构建小乌龟控制 StateGraph。

    图结构:
        START → perceive → decide → act → [条件] → perceive (继续)
                                            └→ END (停止)
    """
    graph = StateGraph(TurtleState)

    graph.add_node("perceive", perceive_node)
    graph.add_node("decide", decide_node)
    graph.add_node("act", act_node)

    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "decide")
    graph.add_edge("decide", "act")

    graph.add_conditional_edges(
        "act",
        route_after_act,
        {"continue": "perceive", "end": END},
    )

    return graph.compile(checkpointer=MemorySaver())


# ============================================================
# 7. 主函数
# ============================================================

def main():
    global _shared_node

    rclpy.init()
    _shared_node = TurtleBridge()

    # 后台线程持续更新传感器数据
    import threading
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(_shared_node), daemon=True
    )
    spin_thread.start()
    time.sleep(1.0)  # 等待 ROS2 连接就绪

    graph = build_turtle_graph()

    initial_state: TurtleState = {
        "pose_x": 0.0, "pose_y": 0.0, "pose_theta": 0.0,
        "side_count": 0,
        "step": "turn",           # 从直走开始（"turn" 下一步会变 "forward"）
        "linear_x": 0.0, "angular_z": 0.0,
        "step_start_time": time.time(),
        "log": [],
    }

    print("=" * 50)
    print("LangGraph 小乌龟正方形控制")
    print("架构: [perceive] → [decide] → [act] → [循环]")
    print("=" * 50)

    try:
        for event in graph.stream(initial_state):
            node_name = list(event.keys())[0]
            data = event[node_name]
            if "log" in data:
                for line in data["log"]:
                    print(f"  {line}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        _shared_node.publish_cmd(0.0, 0.0)
        _shared_node.destroy_node()
        rclpy.shutdown()
        print("正方形绘制完成 ✓")


if __name__ == "__main__":
    main()
```

> **关键理解**：这个例子展示了 LangGraph 的两种核心模式——**循环**（perceive→decide→act→perceive）加**条件路由**（完成则 END，未完成则继续）。每次 act 用计时器判断当前动作是否完成，彻底避免了手动写 `while` 循环，这就是 StateGraph 区别于普通 Chain 的关键能力。

---

## 5. 四种技能对比与学习路径

### 5.1 四种技能对比总结

| 维度 | OpenClaw Skill | Python 控制 | Dify 智能体 | LangChain/LangGraph |
|------|:---:|:---:|:---:|:---:|
| **开发难度** | ⭐ 极低 | ⭐⭐ 中等 | ⭐ 低 | ⭐⭐⭐ 较高 |
| **代码量** | ~50 行 Markdown | ~80 行 Python | ~30 行 Python + GUI | ~150 行 Python |
| **自然语言交互** | ✅ 原生支持 | ❌ 需自己解析 | ✅ 可视化编排 | ✅ 可集成 LLM |
| **循环控制** | 间接（重复调脚本） | ✅ 直接 | 间接（条件分支） | ✅ 节点内实现 |
| **可视化调试** | ❌ | ❌ | ✅ 工作流视图 | ✅ 图可视化 |
| **适合场景** | 简单命令映射 | 数据处理、文件操作 | 快速原型、低代码 | 复杂多步推理 |
| **扩展性** | 加映射行 | 加函数 | 加工作流节点 | 加图节点+边 |
| **外部 API 调用** | ❌（通过 Shell） | ✅ | ✅ | ✅ |
| **学习价值** | 理解智能体交互 | 理解程序控制 | 理解 LLMOps | 理解 AI Agent 架构 |

### 5.2 学习路径建议

```
入门阶段
  │
  ├─ 1. OpenClaw Skill → 理解"用户→智能体→命令"链路
  ├─ 2. Python 脚本编写 → 理解参数解析、文件 I/O、API 调用
  │
进阶阶段
  │
  ├─ 3. Dify 工作流 → 理解低代码编排、LLM 意图识别
  │
高阶阶段
  │
  └─ 4. LangGraph 智能体 → 理解状态图 + 循环推理架构
```

> **建议顺序**：从左到右、从简到繁逐步深入。OpenClaw Skill 帮助快速建立"智能体"的直觉，Python 脚本深入理解程序执行机制，Dify 体验低代码智能体编排，LangGraph 掌握专业的 AI Agent 架构设计。

| 阶段 | 技能 | 核心收获 | 建议用时 |
|:---:|------|------|:---:|
| 入门 | OpenClaw Skill | 理解自然语言到命令的映射；掌握 SKILL.md 编写 | 30 分钟 |
| 入门 | Python 脚本控制 | 理解命令行参数、文件操作、API 调用 | 1 小时 |
| 进阶 | Dify 智能体 | 理解工作流编排、意图识别、工具注册 | 1 小时 |
| 高阶 | LangGraph 智能体 | 理解 StateGraph、条件路由、循环推理 | 2 小时 |

---

<!-- edu-oss-embedded:2082034950332592130,2082034950601027585,2082034950869463041,2082034951133704194 -->
