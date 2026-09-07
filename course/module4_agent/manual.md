# 模块4 智能体应用开发 · 实操手册

适用镜像：**`ydy-eia-jzfull`**（在 jzbase 基础上预装 ROS 2 Humble、Dify、OpenClaw，开机自动拉起）。
仿真对象：**ROS 2 turtlesim**。本模块侧重智能体应用开发，不要求重新安装平台组件。

---

## 任务准备

### 1. 环境确认


| 项目     | 说明                                                                     |
| -------- | ------------------------------------------------------------------------ |
| 平台镜像 | `ydy-eia-jzfull`                                                         |
| 桌面账号 | `ubuntu` / `ubuntu`                                                      |
| 已预装   | MATE、VS Code、Epiphany、Docker、ROS 2 Humble、turtlesim、Dify、OpenClaw |
| Dify     | http://127.0.0.1:3000/install →`/signin`                                |
| OpenClaw | http://127.0.0.1:18789/ ，Token**`jzfull-lab-token`**                    |
| ROS      | `source /opt/ros/humble/setup.bash`                                      |

进入桌面后执行：

```bash
cat ~/Desktop/实训说明.txt
source /opt/ros/humble/setup.bash
ros2 pkg executables turtlesim | head -3
curl -s -o /dev/null -w "dify:%{http_code}\n" http://127.0.0.1:3000/signin
curl -s -o /dev/null -w "openclaw:%{http_code}\n" http://127.0.0.1:18789/
```

若 Dify / OpenClaw 暂未就绪，等待约 1～2 分钟后重试（首次拉起 compose 可能较慢）。

### 2. 启动仿真（全程保持）

新开终端：

```bash
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtlesim_node
```

后续各任务默认该窗口保持运行。

### 3. 文件服务器

```bash
export FILE_SERVER=http://192.168.0.105:8088
```

若无法访问，以平台「实训资源 → 资源下载」页复制的链接为准。

服务器目录约定：

```text
$FILE_SERVER/module4/
├── docs/                 # RAG 文档
├── skills/               # OpenClaw Skill
├── langchain/            # Function Calling 示例与依赖
├── frontend/             # 前端原型
└── workspace.tgz         # 可选整包
```

### 4. 下载实训资源

```bash
export FILE_SERVER=http://192.168.0.105:8088
mkdir -p ~/lab-m4 && cd ~/lab-m4

# 整包（若有）
# wget -c "$FILE_SERVER/module4/workspace.tgz" -O workspace.tgz && tar xf workspace.tgz

# 或按目录分别下载（以资源页文件名为准）
# mkdir -p docs skills langchain frontend
# wget -c "$FILE_SERVER/module4/docs/robot-ops-manual.md" -O docs/robot-ops-manual.md
# wget -c "$FILE_SERVER/module4/skills/turtlesim-patrol.tgz" -O /tmp/turtlesim-patrol.tgz
# tar xf /tmp/turtlesim-patrol.tgz -C skills/
```

无服务器文件时，可直接使用下文内嵌步骤与代码完成本地练习。

### 5. 架构示意

```text
自然语言
   ├─► Dify 应用（提示词 / 记忆 / RAG）     …… 任务 1、2
   ├─► OpenClaw + Skills → shell/python     …… 任务 3
   └─► LangChain Agent + Tools → Twist      …… 任务 4
              └─► 前端页面串联以上能力        …… 任务 5
                        ▼
              /turtle1/cmd_vel → turtlesim
```

### 6. 本手册任务一览


| 任务   | 内容                               |
| ------ | ---------------------------------- |
| 任务 1 | Dify 智能体：提示词与多轮记忆      |
| 任务 2 | Dify 知识库 RAG                    |
| 任务 3 | OpenClaw Skills：自动巡检          |
| 任务 4 | LangChain Function Calling 控 ROS2 |
| 任务 5 | 智能体前端对接与联调               |

---

## 任务 1　Dify 智能体：提示词与多轮记忆

### 目标

在 Dify 中创建聊天助手，编写系统提示词，配置对话记忆，完成多轮自然语言交互验证。

### 操作步骤

**1）打开并初始化**

1. 桌面浏览器（Epiphany）打开 http://127.0.0.1:3000/install
2. 若已初始化则打开 http://127.0.0.1:3000/signin
3. 注册 / 登录管理员账号

**2）配置 DeepSeek**

进入「设置 → 模型供应商」，添加 DeepSeek（或 OpenAI-Compatible）：


| 字段                | 填写                            |
| ------------------- | ------------------------------- |
| API Key             | 教师发放                        |
| API Base / Base URL | 如`https://api.deepseek.com/v1` |
| 模型名              | 如`deepseek-chat`               |

保存后做一次连通测试（若界面有）。

**3）创建聊天助手**

1. 工作室 → 创建应用 → **聊天助手**
2. **系统提示词**示例（可改）：

```text
你是具身智能助手「小龟教练」。
职责：用简洁中文指导用户操作 turtlesim 仿真海龟。
规则：
1. 多轮对话中记住用户目标与已执行动作；
2. 不编造未提供的传感器数据；
3. 涉及运动时，说明应执行的口令：前进/后退/左转/右转/停。
```

3. 开启 **对话记忆** / 设置上下文轮数（如最近 10～20 轮）
4. 发布应用，打开调试对话：

```text
用户：我等会要画一个正方形。
助手：……（确认目标）
用户：第一步做什么？
助手：……（应仍记得「正方形」目标）
```

### 预期结果

- 应用可正常对话
- 第二轮仍能引用第一轮目标（记忆生效）

---

## 任务 2　Dify 知识库 RAG

### 目标

创建知识库并上传操作手册与场景规范，配置分段与 Top-K，关联至任务 1 应用，验证 RAG 增强问答。

### 操作步骤

**1）准备文档**

从文件服务器下载，或本地新建：

```bash
mkdir -p ~/lab-m4/docs && cd ~/lab-m4/docs

# 若服务器已有：
# curl -fLO "$FILE_SERVER/module4/docs/robot-ops-manual.md"
# curl -fLO "$FILE_SERVER/module4/docs/scene-spec.md"

cat > robot-ops-manual.md <<'EOF'
# turtlesim 操作手册

## 基本口令
- 前进：线速度沿 X 轴正方向
- 后退：线速度沿 X 轴负方向
- 左转 / 右转：角速度绕 Z 轴
- 停：线速度与角速度均为 0

## ROS 话题
- 速度指令：`/turtle1/cmd_vel`（geometry_msgs/Twist）
- 位姿：`/turtle1/pose`

## 安全规范
- 连续运动指令应设置超时并自动刹车
- 联调前确认 turtlesim 窗口已打开
EOF

cat > scene-spec.md <<'EOF'
# 场景规范

## 仿真环境
- 发行版：ROS 2 Humble
- 仿真器：turtlesim
- 默认海龟：turtle1

## 验收场景
1. 前进 2 秒后停止
2. 左转约 90 度后停止
3. 自然语言经智能体触发上述动作之一
EOF
```

**2）在 Dify 建知识库**

1. 知识库 → 创建 → 上传 `robot-ops-manual.md`、`scene-spec.md`
2. 分段：按标题 / 长度适中；Top-K 先用 3～5
3. 索引完成后，回到任务 1 应用 → 添加「知识库」引用
4. 提问：「速度指令发到哪个话题？」「联调前要确认什么？」

回答应能体现手册内容（可带引用片段）。

### 预期结果

- 文档已完成索引
- 应用已关联知识库
- 提问能命中手册要点

---

## 任务 3　OpenClaw Skills：自动巡检

### 目标

编写本地控龟脚本与 OpenClaw Skill，配置触发条件与执行逻辑，用自然语言触发「自动巡检」。

### 操作步骤

**1）编写本地控龟脚本**

```bash
mkdir -p ~/lab-m4/scripts && cd ~/lab-m4/scripts

cat > drive_turtle.py <<'PY'
#!/usr/bin/env python3
import sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

MAP = {
    "前进": (1.0, 0.0, 2.0),
    "后退": (-1.0, 0.0, 2.0),
    "左转": (0.0, 1.5, 1.2),
    "右转": (0.0, -1.5, 1.2),
    "停": (0.0, 0.0, 0.3),
    "巡检": (0.8, 0.6, 6.0),  # 简单绕行示意
}

class Driver(Node):
    def __init__(self):
        super().__init__("m4_drive_turtle")
        self.pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)

    def run(self, text: str):
        key = text.strip()
        for k, v in MAP.items():
            if k in key:
                lx, az, dur = v
                break
        else:
            print("unknown:", key); return
        msg = Twist(); msg.linear.x = lx; msg.angular.z = az
        end = time.time() + dur
        while time.time() < end:
            self.pub.publish(msg); time.sleep(0.1)
        stop = Twist()
        for _ in range(5):
            self.pub.publish(stop); time.sleep(0.05)

def main():
    rclpy.init()
    n = Driver(); n.run(" ".join(sys.argv[1:]) or "巡检")
    n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
PY

cat > drive_turtle.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
exec python3 "$(cd "$(dirname "$0")" && pwd)/drive_turtle.py" "$@"
EOF
chmod +x drive_turtle.sh drive_turtle.py

# 先不经过 OpenClaw 验证：
# ./drive_turtle.sh 前进
```

**2）编写 Skill（SKILL.md）**

```bash
mkdir -p ~/lab-m4/skills/turtlesim-patrol
cat > ~/lab-m4/skills/turtlesim-patrol/SKILL.md <<'EOF'
---
name: turtlesim-patrol
description: 控制 ROS2 turtlesim 海龟完成前进/转向/巡检
metadata:
  openclaw:
    emoji: "🐢"
---

# turtlesim 巡检技能

## 前置
- 已运行：`ros2 run turtlesim turtlesim_node`
- 脚本路径：`/home/ubuntu/lab-m4/scripts/drive_turtle.sh`

## 何时使用
用户提到：前进、后退、左转、右转、停、巡检、自动巡检、绕一圈。

## 如何执行
在终端执行（优先）：

```bash
/home/ubuntu/lab-m4/scripts/drive_turtle.sh "巡检"
```

其他口令把参数换成「前进」「左转」等。

## 注意

- 不要并行启动多个长时间控龟进程
- 执行后用一句话反馈结果
  EOF

```

按 OpenClaw 版本将 Skill 安装到 Agent 工作区（常见为复制到 Agent 的 `skills/` 目录，或以 `openclaw skills install` 安装）。**以桌面内 `openclaw --help` / 教师补充说明为准**。

若服务器已有包：

```bash
wget -c "$FILE_SERVER/module4/skills/turtlesim-patrol.tgz" -O /tmp/turtlesim-patrol.tgz
tar xf /tmp/turtlesim-patrol.tgz -C ~/lab-m4/skills/
```

**3）在 OpenClaw 中验证**

1. 打开 http://127.0.0.1:18789/
2. 粘贴 Token：`jzfull-lab-token` → Connect
3. 确认 DeepSeek 已配置
4. 发送：「执行自动巡检」或「让海龟前进」
5. 观察 turtlesim 是否动作；看 OpenClaw 是否引用了 Skill / 执行了脚本

### 预期结果

- Skill 文件已就位且可被 Agent 使用
- 自然语言能触发控龟
- 「巡检」口令有可观察运动

---

## 任务 4　LangChain Function Calling 控 ROS2

### 目标

使用 LangChain 集成 DeepSeek，定义运动控制工具，通过 Function Calling 将自然语言转为 turtlesim 动作并执行。

### 操作步骤

**1）准备依赖**

```bash
mkdir -p ~/lab-m4/langchain && cd ~/lab-m4/langchain

# 有网：
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install langchain langchain-openai langchain-core

# 离线（教师投放后）：
# wget -c "$FILE_SERVER/module4/langchain/wheels.tgz" && tar xf wheels.tgz
# pip install --no-index --find-links=./wheels -r requirements.txt
```

> 推荐：Tool 只调用 shell 脚本 `drive_turtle.sh`，避免 venv 与系统 `rclpy` 冲突。

**2）编写示例脚本**

```bash
cat > ~/lab-m4/langchain/cmd_vel_agent_demo.py <<'PY'
#!/usr/bin/env python3
"""LangChain 工具调用 → drive_turtle.sh → turtlesim"""
import os
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DRIVE = os.path.expanduser("~/lab-m4/scripts/drive_turtle.sh")


@tool
def move_turtle(command: str) -> str:
    """控制 turtlesim。command 取：前进、后退、左转、右转、停、巡检。"""
    r = subprocess.run([DRIVE, command], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return f"失败: {r.stderr or r.stdout}"
    return f"已执行: {command}"


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请 export DEEPSEEK_API_KEY=...")
    llm = ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0,
    )
    tools = [move_turtle]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是海龟控制助手。用户要运动时必须调用 move_turtle 工具。"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    ex = AgentExecutor(agent=agent, tools=tools, verbose=True)
    q = os.environ.get("USER_QUERY", "请让海龟前进，然后停")
    print(ex.invoke({"input": q}))


if __name__ == "__main__":
    main()
PY
```

**3）运行**

```bash
# 确认 turtlesim 已开（任务准备第 2 步）
export DEEPSEEK_API_KEY=sk-...   # 教师提供
cd ~/lab-m4/langchain
source .venv/bin/activate
python3 cmd_vel_agent_demo.py
```

### 预期结果

- Agent 日志中出现 tool call（`move_turtle`）
- turtlesim 出现对应运动

---

## 任务 5　智能体前端对接与联调

### 目标

编写（或使用原型）智能体前端，对接后台服务，实现对话、知识库问答、工具调用与具身交互可视化。

### 操作步骤

**1）获取或创建前端**

```bash
mkdir -p ~/lab-m4/frontend && cd ~/lab-m4/frontend
# wget -c "$FILE_SERVER/module4/frontend/frontend.tgz" && tar xf frontend.tgz
```

若暂无包，可用最小静态页联调 Dify（需在 Dify 创建 API Key）：

```bash
cat > index.html <<'EOF'
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8" />
  <title>海龟助手</title>
  <style>
    body { font-family: sans-serif; max-width: 720px; margin: 2rem auto; }
    #log { border: 1px solid #ccc; min-height: 240px; padding: 1rem; white-space: pre-wrap; }
    input, button { font-size: 1rem; padding: 0.4rem 0.6rem; }
  </style>
</head>
<body>
  <h1>海龟助手（对接 Dify）</h1>
  <p>在下方填写 Dify API Base、Key、App 后发送消息。</p>
  <div>
    API Base <input id="base" size="40" value="http://127.0.0.1:3000/v1" />
    App ID <input id="app" size="28" placeholder="应用 ID" />
    API Key <input id="key" size="36" type="password" />
  </div>
  <div id="log"></div>
  <input id="msg" size="48" placeholder="输入问题" />
  <button id="send">发送</button>
  <script>
    const log = (t) => { document.getElementById('log').textContent += t + '\n'; };
    document.getElementById('send').onclick = async () => {
      const base = document.getElementById('base').value.replace(/\/$/, '');
      const key = document.getElementById('key').value.trim();
      const query = document.getElementById('msg').value.trim();
      log('> ' + query);
      const resp = await fetch(base + '/chat-messages', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer ' + key,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: {},
          query,
          response_mode: 'blocking',
          user: 'lab-student',
        }),
      });
      const data = await resp.json();
      log(data.answer || JSON.stringify(data, null, 2));
    };
  </script>
</body>
</html>
EOF

python3 -m http.server 5173
# 浏览器打开 http://127.0.0.1:5173/
```

**2）功能要求**

页面或联调结果中应能体现：

1. **对话**（Dify 或自建后端）
2. **知识库问答**（走已绑 RAG 的应用）
3. **工具调用**（展示 LangChain / OpenClaw 执行结果，或提供「前进」按钮调 `drive_turtle.sh`）
4. **具身反馈**：至少文字说明海龟动作；进阶可读 `/turtle1/pose` 显示坐标

按钮直连示例（可选）：

```bash
cat > ~/lab-m4/frontend/move_api.py <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess, urllib.parse, os
DRIVE = os.path.expanduser("~/lab-m4/scripts/drive_turtle.sh")

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.urlparse(self.path)
        if q.path != "/move":
            self.send_response(404); self.end_headers(); return
        cmd = urllib.parse.parse_qs(q.query).get("cmd", ["停"])[0]
        subprocess.run([DRIVE, cmd], timeout=30)
        self.send_response(200); self.end_headers()
        self.wfile.write(f"ok:{cmd}".encode())

HTTPServer(("127.0.0.1", 5174), H).serve_forever()
PY
# python3 move_api.py  &
# 前端可 fetch('http://127.0.0.1:5174/move?cmd=前进')
```

### 预期结果

- 浏览器可打开前端
- 能完成对话（及知识库问答）
- 能触发或展示工具 / 运动结果

---

## 附录 A　常用地址与命令


| 服务 / 用途 | 地址或命令                                        |
| ----------- | ------------------------------------------------- |
| 文件服务器  | http://192.168.0.105:8088/                        |
| Dify        | http://127.0.0.1:3000/install 、 /signin          |
| OpenClaw    | http://127.0.0.1:18789/ ，Token`jzfull-lab-token` |
| 起仿真      | `ros2 run turtlesim turtlesim_node`               |
| 速度话题    | `/turtle1/cmd_vel` · `geometry_msgs/Twist`       |
| 位姿        | `ros2 topic echo /turtle1/pose`                   |
| 本地控龟    | `~/lab-m4/scripts/drive_turtle.sh 前进`           |

---

## 附录 B　故障排查


| 现象                    | 处理                                              |
| ----------------------- | ------------------------------------------------- |
| Dify 502 / 打不开       | 等 compose 就绪；`docker ps` 看 nginx / api       |
| OpenClaw 要 Token       | 填`jzfull-lab-token`                              |
| 模型无输出              | 检查 DeepSeek Key / Base URL / 余额               |
| Skill 不跑脚本          | 先手动跑`drive_turtle.sh`；检查绝对路径与权限     |
| LangChain 与 rclpy 冲突 | Tool 只调 shell，勿在 venv 里硬 import 系统 rclpy |
| 8088 下不了包           | 改用手册内嵌步骤；或向教师要拷贝                  |

---

## 附录 C　教师说明

- 本模块默认绑定镜像 **`ydy-eia-jzfull`（平台镜像 id 5202）**
- 将 docs / skills / langchain wheels / frontend 放到 `$FILE_SERVER/module4/`
- DeepSeek Key 课上分发，禁止放进公开文件服务器
- 若误开 `ydy-eia-jzbase`，须先按《模块3 实操手册》装完 ROS、Dify、OpenClaw 再继续