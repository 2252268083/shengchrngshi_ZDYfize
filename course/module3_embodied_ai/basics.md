# 模块3：具身智能平台部署与运维 — 基础知识

---

## 目录

1. [什么是具身智能](#1-什么是具身智能)
2. [ROS2 机器人操作系统](#2-ros2-机器人操作系统)
3. [Gazebo 机器人仿真](#3-gazebo-机器人仿真)
4. [Docker 容器技术](#4-docker-容器技术)
5. [Docker Compose 多容器编排](#5-docker-compose-多容器编排)
6. [code-server 浏览器 IDE](#6-code-server-浏览器-ide)
7. [rosbridge 与 WebSocket 通信](#7-rosbridge-与-websocket-通信)
8. [机器人导航与控制基础](#8-机器人导航与控制基础)
9. [OpenClaw 智能体网关](#9-openclaw-智能体网关)
10. [Dify LLM 应用平台](#10-dify-llm-应用平台)
11. [Linux 系统部署基础](#11-linux-系统部署基础)


---

## 1. 什么是具身智能

### 1.1 定义

**具身智能**（Embodied Artificial Intelligence, EAI）是人工智能的一个重要分支，强调智能体必须具备物理或仿真的"身体"，通过与环境的实时交互来感知、推理和行动。与传统的"离身"AI（如纯图像分类、文本生成）不同，具身智能关注的是 **"感知—决策—行动"的闭环**。

> **核心理念**：智能源于身体与环境的持续互动，而非孤立的信息处理。

### 1.2 技术范畴

具身智能属于以下领域的交叉：

```
┌─────────────────────────────────────────────────┐
│                  具身智能                        │
│          (Embodied AI)                           │
│    ┌──────────────┐    ┌──────────────────┐      │
│    │  机器人学      │    │  仿真与数字孪生    │      │
│    │  (Robotics)   │    │  (Simulation)     │      │
│    └──────┬───────┘    └──────────────────┘      │
│           │                                      │
│    ┌──────┴───────┐    ┌──────────────────┐      │
│    │  深度学习      │    │  云原生/容器化     │      │
│    │ (Deep         │◄──►│  (Cloud Native)   │      │
│    │  Learning)    │    │                   │      │
│    └──────────────┘    └──────────────────┘      │
│           │                                      │
│    ┌──────┴───────┐    ┌──────────────────┐      │
│    │  LLM 智能体   │    │  边缘计算          │      │
│    │ (LLM Agent)  │    │  (Edge Computing) │      │
│    └──────────────┘    └──────────────────┘      │
└─────────────────────────────────────────────────┘
```

### 1.3 本模块的具身智能体系

本模块构建的"仓库小车仿真平台"是一个典型的具身智能系统，包含以下层次：

![具身智能五层架构](/resource/oss/download/2082034945467199490)

### 1.4 具身智能的核心挑战

| 挑战 | 说明 | 本模块应对策略 |
|:---|:---|:---|
| **仿真与现实差距（Sim-to-Real Gap）** | 仿真环境与真实物理世界存在差异 | Gazebo 提供高保真物理引擎 |
| **实时性要求** | 机器人控制需毫秒级响应 | ROS2 实时通信框架 |
| **多组件协同** | 感知、规划、控制、可视化需协同工作 | Docker Compose 统一编排 |
| **环境一致性** | 不同机器环境差异导致"跑不起来" | Docker 容器化，一次构建到处运行 |
| **人机交互** | 自然语言到机器人指令的映射 | OpenClaw 智能体网关 + LLM |

### 1.5 具身智能的发展历程

| 阶段 | 年代 | 主要特征 | 代表性工作 |
|:---|:---|:---|:---|
| 传统机器人期 | 1990s~2010 | 硬编码规则、PID 控制 | ROS1 生态 |
| 深度学习融合期 | 2015~2020 | 端到端学习、DRL 控制 | OpenAI Gym, ROS2 |
| 大模型智能体期 | 2022~至今 | LLM + 机器人，自然语言控车 | SayCan, RT-2, OpenClaw |

---

## 2. ROS2 机器人操作系统

### 2.1 什么是 ROS2

**ROS2**（Robot Operating System 2）是一个用于机器人软件开发的**开源中间件框架**。它不是传统意义上的操作系统，而是运行在 Linux/Windows 之上的分布式通信框架，提供了一套标准化的机器人软件开发工具和库。

> **一句话理解**：ROS2 是机器人世界的"安卓系统"——提供标准 API 和通信机制，让不同模块能即插即用地协同工作。

<img src="images/ros-bp.svg" width="80" alt="ROS"/>

### 2.2 ROS1 vs ROS2

| 特性 | ROS1 | ROS2 |
|:---|:---|:---|
| **发布年代** | 2007 | 2017 |
| **通信协议** | 自定义 TCP/UDP（TCPROS） | DDS（Data Distribution Service） |
| **操作系统** | 仅 Ubuntu | Ubuntu / Windows / macOS |
| **实时性** | 不支持 | 支持实时控制 |
| **多机器人** | 需要额外配置 | 原生支持 |
| **安全性** | 无 | DDS-Security 加密 |
| **Python 版本** | Python 2 | Python 3 |
| **本模块使用** | ❌ | ✅ ROS2 Humble |

### 2.3 ROS2 核心概念

#### 节点（Node）

节点是 ROS2 中最小的执行单元，每个节点负责一个独立功能：

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 激光雷达  │    │  控制器   │    │  摄像头   │
│  节点     │    │  节点     │    │  节点     │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              ┌──────┴──────┐
              │  ROS2 通信层  │
              └──────────────┘
```

#### 话题（Topic）

话题是节点间**异步通信**的数据通道，采用**发布/订阅（Pub/Sub）**模式：

```
发布者 (Publisher)              订阅者 (Subscriber)
     │                                │
     │  publish("cmd_vel", msg)       │
     ├──────────────┬─────────────────┤
     │              ▼                  │
     │     ┌────────────────┐         │
     │     │   /cmd_vel      │         │
     │     │  (Twist 消息)   │         │
     │     └────────┬───────┘         │
     │              │                  │
     │              └──── callback(msg)
     │                                  │
     ▼                                  ▼
  控制节点                           底盘驱动节点
```

本模块涉及的关键话题：

| 话题名称 | 消息类型 | 说明 |
|:---|:---|:---|
| `/bcr_bot/cmd_vel` | `geometry_msgs/Twist` | **速度控制指令**：线速度 + 角速度 |
| `/bcr_bot/scan` | `sensor_msgs/LaserScan` | **激光雷达数据**：障碍物距离信息 |
| `/bcr_bot/odom` | `nav_msgs/Odometry` | **里程计**：机器人位置和速度估计 |

#### 服务（Service）

服务是**同步的请求/响应**通信模式（一问一答），适用于需要立即返回结果的操作。

#### 动作（Action）

动作是**带反馈的长时间任务**，适用于导航、抓取等需要持续跟踪进度的操作。

```
话题 vs 服务 vs 动作：

话题 (Topic)：  发布者 ────→ 订阅者         （持续推送，如传感器数据）
服务 (Service)： 客户端 ──→←── 服务端        （单次问答，如"当前位置？"）
动作 (Action)：  客户端 ──→←── 服务端 + 反馈   （长任务，如"导航到(x,y)"）
```

![ROS2 话题通信架构](/resource/oss/download/2082034945744023554)

### 2.4 DDS 通信协议

ROS2 底层使用 **DDS（Data Distribution Service）** 协议，这是一个工业级的分布式通信标准：

```
节点A                                    节点B
  │                                        │
  ├─ Publisher                             ├─ Subscriber
  │   (DataWriter)                         │   (DataReader)
  │       │                                │       │
  │       └──────── DDS Global ────────────┘       │
  │              Data Space                        │
  │                                                │
  │   自动发现 (Discovery) · QoS 策略 · 实时传输    │
```

DDS 的关键特性：

| 特性 | 说明 |
|:---|:---|
| **自动发现** | 节点启动后自动发现网络中其他节点，无需手动配置 IP |
| **QoS 策略** | 可配置可靠性（RELIABLE/BEST_EFFORT）、持久性、截止时间等 |
| **去中心化** | 无中心节点（Master），天然支持分布式部署 |
| **实时性** | 支持 RTPS（Real-Time Publish-Subscribe）协议 |

### 2.5 ROS2 常用命令行工具

```bash
# 查看所有运行中的节点
ros2 node list

# 查看所有话题
ros2 topic list

# 监听某个话题的数据
ros2 topic echo /bcr_bot/cmd_vel

# 向话题发布消息
ros2 topic pub /bcr_bot/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.0}}"

# 查看话题的详细信息（类型、发布者、订阅者）
ros2 topic info /bcr_bot/scan

# 查看节点信息
ros2 node info /bcr_bot_controller
```

---

## 3. Gazebo 机器人仿真

### 3.1 什么是 Gazebo

**Gazebo** 是一个开源的三维机器人仿真环境，提供高保真的物理引擎、传感器模拟和逼真的渲染效果。它是 ROS 生态中最常用的仿真工具。

> **一句话理解**：Gazebo 是机器人的"虚拟训练场"——在没有真实硬件的情况下，模拟机器人在真实世界中的行为。

### 3.2 Gazebo 的核心能力

```
┌─────────────────────────────────────────────┐
│                Gazebo 仿真世界                │
│                                              │
│  ┌──────────┐     ┌──────────┐               │
│  │  物理引擎  │     │  渲染引擎  │               │
│  │  (ODE/    │     │  (OGRE2)  │               │
│  │  Bullet)  │     │           │               │
│  └─────┬────┘     └─────┬─────┘               │
│        │                │                     │
│  ┌─────┴────────────────┴─────┐               │
│  │       传感器模拟              │               │
│  │  激光雷达 · 相机 · IMU · GPS  │               │
│  └─────────────┬───────────────┘               │
│                │                               │
│  ┌─────────────┴───────────────┐               │
│  │       机器人模型 (SDF/URDF)   │               │
│  │   轮式 · 四足 · 机械臂 · 无人机│              │
│  └─────────────────────────────┘               │
└─────────────────────────────────────────────┘
```

![Gazebo 仿真世界](/resource/oss/download/2082034946008264706)

### 3.3 Gazebo 与本模块的关系

本模块使用 Gazebo 模拟了一个**仓库场景**，其中包含：

| 仿真元素 | 说明 |
|:---|:---|
| **仓库环境** | 墙壁、货架、障碍物等静态场景 |
| **差分驱动机器人** | 两轮驱动 + 万向轮，模拟物流小车 |
| **激光雷达（LiDAR）** | 360° 扫描，检测障碍物距离 |
| **相机** | 模拟 RGB 摄像头，提供视觉感知 |
| **物理引擎** | 模拟重力、碰撞、摩擦力等真实物理效果 |

```
仓库仿真场景示意图：

    ┌─────────────────────────────────────┐
    │  货架A      货架B      货架C         │
    │                                     │
    │           ┌───┐                     │
    │           │🤖 │  ← 仓库小车          │
    │           └───┘                     │
    │                                     │
    │  货架D      货架E      货架F         │
    └─────────────────────────────────────┘
         ↑ 激光雷达扫描范围（360°）
```

### 3.4 URDF 与 SDF 模型格式

ROS2/Gazebo 使用两种格式描述机器人模型：

| 格式 | 全称 | 用途 | 特点 |
|:---|:---|:---|:---|
| **URDF** | Unified Robot Description Format | 描述机器人结构 | XML 格式，定义连杆(joint)、关节(link) |
| **SDF** | Simulation Description Format | 描述仿真世界 | 功能更强，支持光照、物理属性等 |

```xml
<!-- URDF 示例：简单轮式机器人 -->
<robot name="bcr_bot">
  <!-- 车体 -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.4 0.3 0.2"/>
      </geometry>
    </visual>
  </link>
  
  <!-- 左轮 -->
  <link name="left_wheel"/>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
  </joint>
  
  <!-- 激光雷达传感器 -->
  <gazebo reference="base_link">
    <sensor type="ray" name="laser">
      <ray>
        <scan><horizontal><samples>360</samples></horizontal></scan>
      </ray>
    </sensor>
  </gazebo>
</robot>
```

---

## 4. Docker 容器技术

### 4.1 什么是 Docker

**Docker** 是一个开源的**容器化平台**，它将应用程序及其所有依赖打包到一个轻量级、可移植的**容器**中，确保应用在任何 Linux 环境中都能一致运行。

> **一句话理解**：Docker 就像"集装箱"——把软件和它的所有依赖装进一个标准化的箱子，到哪都能直接用。

<img src="images/docker-bp.svg" width="80" alt="Docker"/>

### 4.2 容器 vs 虚拟机

![容器 vs 虚拟机](/resource/oss/download/2082034946268311554)

| 特性 | 虚拟机（VM） | Docker 容器 |
|:---|:---|:---|
| **启动速度** | 分钟级 | 秒级 |
| **资源占用** | GB 级内存/磁盘 | MB 级 |
| **隔离级别** | 完整 OS 隔离 | 进程级隔离（共享宿主机内核） |
| **镜像大小** | 通常数 GB | 通常数百 MB |
| **性能** | 有虚拟化损耗 | 接近原生性能 |
| **可移植性** | 依赖 Hypervisor | 任意 Linux 宿主机 |

### 4.3 Docker 核心三要素

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   镜像 (Image)    ──→    容器 (Container)         │
│   ┌─────────┐            ┌─────────┐             │
│   │ 只读模板  │    run     │ 运行实例  │             │
│   │          │  ───────→  │          │             │
│   │ Layer 3  │            │ 可读写层  │             │
│   │ Layer 2  │            ├─────────┤             │
│   │ Layer 1  │            │  Layer 3 │             │
│   └─────────┘            │  Layer 2 │             │
│                           │  Layer 1 │             │
│       │                   └─────────┘             │
│       │ save/load                                    │
│       ▼                                            │
│   ┌─────────┐                                      │
│   │  Registry │  ← 镜像仓库（Docker Hub / 私有仓库）│
│   │  ┌─────┐ │                                      │
│   │  │ tar │ │  ← 离线场景：docker save/load       │
│   │  └─────┘ │                                      │
│   └─────────┘                                      │
└──────────────────────────────────────────────────┘
```

#### 镜像（Image）

镜像是一个**只读的模板**，采用**分层存储（Layered Filesystem）**结构：

```
镜像分层结构：

┌──────────────────────┐
│  应用层（Python依赖）  │  ← 自定义：pip install ...
├──────────────────────┤
│  中间件层（ROS2）      │  ← apt install ros-humble-*
├──────────────────────┤
│  基础层（Ubuntu）      │  ← FROM ubuntu:22.04
└──────────────────────┘

每层只读，共享复用 → 节省磁盘空间
```

#### 容器（Container）

容器是镜像的**运行实例**，在镜像的只读层之上添加一个**可读写层**：

- 容器停止后，可读写层的数据**丢失**（除非挂载到宿主机）
- 同一镜像可启动多个相互隔离的容器

#### 仓库（Registry）

仓库是镜像的**存储和分发中心**：

| 仓库类型 | 说明 | 本模块使用 |
|:---|:---|:---|
| **Docker Hub** | 官方公共仓库 | ❌ 离线环境不使用 |
| **本地 tar 文件** | 离线导出/导入 | ✅ `docker save` / `docker load` |

### 4.4 Docker 常用命令

```bash
# ── 镜像操作 ──
docker images                    # 列出本地所有镜像
docker pull <image>              # 从仓库拉取镜像
docker load -i <file>.tar        # 从 tar 文件导入镜像（离线）
docker save -o <file>.tar <image># 导出镜像为 tar 文件
docker rmi <image>               # 删除镜像

# ── 容器操作 ──
docker ps                        # 列出运行中的容器
docker ps -a                     # 列出所有容器（包括已停止）
docker run -d --name <name> <image>  # 后台启动容器
docker exec -it <container> bash     # 进入容器终端
docker logs -f <container>           # 实时查看容器日志
docker stop <container>              # 停止容器
docker rm <container>                # 删除容器

# ── 系统操作 ──
docker system prune -a           # 清理未使用的镜像/容器/网络（⚠️ 会删除数据）
docker stats                     # 查看容器资源占用
```

### 4.5 容器网络与端口映射

Docker 容器默认运行在独立的虚拟网络中。要与宿主机通信，需要**端口映射**：

```
宿主机                      Docker 网络
┌──────────────────┐      ┌──────────────────┐
│  localhost:18080 │ ───→ │ 容器:18080 (code-server) │
│  localhost:8888  │ ───→ │ 容器:8888  (仿真 Web)     │
│  localhost:28789 │ ───→ │ 容器:28789 (OpenClaw)     │
│  localhost:19090 │ ───→ │ 容器:19090 (rosbridge)    │
└──────────────────┘      └──────────────────┘

-p 宿主机端口:容器端口  (port mapping)
```

本模块端口映射一览：

| 宿主机端口 | 容器内端口 | 服务 |
|:---|:---|:---|
| 18080 | 18080 | code-server 浏览器 IDE |
| 8888 | 8888 | 仿真 Web 仪表盘 |
| 28789 | 28789 | OpenClaw 智能体面板 |
| 19090 | 19090 | rosbridge WebSocket |

### 4.6 数据持久化：挂载（Volume / Bind Mount）

容器停止后内部数据会丢失。为了让代码持久保存，使用**挂载**将宿主机目录映射到容器内：

```
宿主机                           容器
┌──────────────────┐           ┌──────────────────┐
│ robot-code/      │ ──bind──→ │ /workspace       │
│  ├── scripts/    │           │  ├── scripts/    │
│  ├── drive.sh    │           │  ├── drive.sh    │
│  └── ...         │           │  └── ...         │
└──────────────────┘           └──────────────────┘
       ↑                                ↑
  保存到磁盘                        容器内编辑

修改实时同步，重启容器不丢失
```

---

## 5. Docker Compose 多容器编排

### 5.1 什么是 Docker Compose

**Docker Compose** 是 Docker 官方的**多容器编排工具**，通过一个 YAML 文件定义多个容器的配置和依赖关系，实现一键启动整个应用栈。

> **本模块场景**：一个仿真系统需要同时运行 Gazebo 容器、Web 仪表盘容器等，Docker Compose 将它们作为一个整体管理。

### 5.2 `docker-compose.yml` 核心结构

```yaml
version: '3.8'

services:
  # ── 主开发容器 ──
  vehicle-dev:
    image: ydy-eia/vehicle-agent:latest
    container_name: ydy-eia-vehicle-dev
    ports:
      - "18080:18080"   # code-server
      - "19090:19090"   # rosbridge
      - "28789:28789"   # OpenClaw
    volumes:
      - ../robot-code:/workspace     # 代码挂载
      - ../:/project                 # 项目挂载
    environment:
      - DISPLAY=${DISPLAY}
      - CODE_SERVER_PORT=18080
    network_mode: host

  # ── 仿真 Web 容器 ──
  sim-web:
    image: ydy-eia/sim-web:latest
    container_name: ydy-eia-sim-web
    ports:
      - "8888:8888"
    network_mode: host
```

### 5.3 Docker Compose 常用命令

```bash
docker compose up -d            # 后台启动所有服务
docker compose down             # 停止并移除所有服务
docker compose ps               # 查看服务状态
docker compose logs -f <service># 查看指定服务日志
docker compose restart          # 重启所有服务
```

![Docker Compose 多容器架构](/resource/oss/download/2082034946532552705)

### 5.4 环境变量配置（`.env` 文件）

Docker Compose 支持通过 `.env` 文件管理配置参数：

```bash
# docker/.env 示例
CODE_SERVER_PORT=18080
SIM_WEB_PORT=8888
OPENCLAW_PORT=28789
ROSBRIDGE_PORT=19090
# DEEPSEEK_API_KEY=   ← 离线考场留空
```

在 `docker-compose.yml` 中引用：

```yaml
ports:
  - "${CODE_SERVER_PORT}:18080"
```

---

## 6. code-server 浏览器 IDE

### 6.1 什么是 code-server

**code-server** 是 **VS Code** 的 Web 版本，由 Coder 公司开源维护。它将 VS Code 的编辑体验完整搬到了浏览器中，用户无需安装任何本地 IDE 即可在浏览器中编写、调试代码。

> **在本模块中的作用**：学生通过浏览器直接操作容器内的代码，无需 SSH 或远程桌面。

<img src="images/vscode-bp.svg" width="60" alt="VS Code"/>

### 6.2 code-server 架构

![code-server 架构图](/resource/oss/download/2082034946796793857)

### 6.3 关键特性

| 特性 | 说明 |
|:---|:---|
| **零安装** | 浏览器打开即用，无需安装 VS Code |
| **集成的终端** | 内置终端直接在容器内执行命令 |
| **扩展支持** | 支持大部分 VS Code 扩展（Python、Docker 等） |
| **容器原生** | 代码直接操作容器内文件，无传输延迟 |
| **多用户** | 支持多用户同时访问不同工作区 |

---

## 7. rosbridge 与 WebSocket 通信

### 7.1 什么是 rosbridge

**rosbridge** 是一个 ROS 生态中的通信桥接工具，它在 ROS2 的话题/服务系统和外部 Web 应用之间建立桥梁，使得浏览器中的 JavaScript 代码也能与 ROS2 节点通信。

### 7.2 为什么需要 rosbridge

![rosbridge WebSocket 通信](/resource/oss/download/2082034947065229314)

### 7.3 rosbridge 通信流程

```
1. 浏览器 JavaScript 建立 WebSocket 连接：
   new WebSocket('ws://localhost:19090')

2. 订阅 ROS2 话题：
   → {"op": "subscribe", "topic": "/bcr_bot/scan"}

3. 接收传感器数据（JSON）：
   ← {"topic": "/bcr_bot/scan", "msg": {...}}

4. 发布控制指令：
   → {"op": "publish", "topic": "/bcr_bot/cmd_vel", "msg": {...}}
```

### 7.4 WebSocket vs HTTP

| 特性 | HTTP | WebSocket |
|:---|:---|:---|
| **通信模式** | 请求-响应（单向） | 全双工（双向） |
| **连接方式** | 短连接（每次请求建立连接） | 长连接（保持连接） |
| **实时性** | 需轮询（polling） | 实时推送 |
| **开销** | 每次请求带完整 HTTP 头 | 帧头极小（2-14 字节） |
| **本模块用途** | code-server 页面访问 | 仿真数据实时推送 |

---

## 8. 机器人导航与控制基础

### 8.1 差分驱动运动学

本模块的仓库小车使用 **差分驱动（Differential Drive）** 模型——两个独立驱动的轮子，通过速度差实现转向：




### 8.2 Twist 消息格式

ROS2 中使用 `geometry_msgs/msg/Twist` 消息控制机器人运动：

```
Twist 消息结构：

linear (线速度)               angular (角速度)
├── x: 前进/后退速度 (m/s)     ├── x: (不用于地面机器人)
├── y: 横向速度 (m/s)          ├── y: (不用于地面机器人)
└── z: 垂直速度 (m/s)          └── z: 绕Z轴旋转速度 (rad/s)
        ↑                              ↑
    正值=前进                     正值=左转(逆时针)
    负值=后退                     负值=右转(顺时针)
```

```bash
# 前进 0.5 m/s
ros2 topic pub /bcr_bot/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# 原地旋转 1.0 rad/s
ros2 topic pub /bcr_bot/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}"

# 停止
ros2 topic pub /bcr_bot/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

### 8.3 激光雷达数据（LaserScan）

激光雷达返回的是距离测量值的环形数组：

```
LaserScan 消息解读：

       0°(正前方)
         ↑
    ┌────┴────┐
    │   🚗    │    机器人居中
    └────┬────┘
         ↓
      180°(正后方)

ranges[i] = 第 i 个角度的障碍物距离 (米)

示例：
ranges[0] = 1.5  → 正前方 1.5 米有障碍物
ranges[180] = 5.0 → 正右方 5.0 米有障碍物（较远）
ranges[90] = ∞ → 正左方没有检测到障碍物
```

### 8.4 简单避障逻辑

```python
# 简化版避障逻辑（伪代码）
def avoid_obstacle(scan_data):
    front = scan_data.ranges[0]      # 正前方距离
    left = scan_data.ranges[90]      # 左侧距离
    right = scan_data.ranges[270]    # 右侧距离
    
    if front < 0.5:                  # 前方 0.5m 内有障碍
        if left > right:
            turn_right()              # 右侧空间大，右转
        else:
            turn_left()               # 左侧空间大，左转
    else:
        go_forward()                  # 安全，继续前进
```

---

## 9. OpenClaw 智能体网关

### 9.1 什么是 OpenClaw

**OpenClaw** 是一个开源的**智能体网关框架**，它在 LLM（大语言模型）和机器人系统之间充当"翻译层"，使得用户可以通过自然语言指令控制机器人。

![OpenClaw 智能体网关](/resource/oss/download/2082034947325276161)

### 9.2 离线 vs 在线模式

| 模式 | 说明 | 本模块适用 |
|:---|:---|:---|
| **在线模式** | 调用云端 LLM API（如 DeepSeek）进行自然语言理解 | 需联网，需 API Key |
| **离线模式** | 通过本地脚本执行预定义控制逻辑 | ✅ 考场默认模式 |

> 离线考场无需填写 `DEEPSEEK_API_KEY`，自然语言控车通过本地脚本 `drive.sh` 等完成。

---

## 10. Dify LLM 应用平台

### 10.1 什么是 Dify

**Dify** 是一个开源的**LLM 应用开发平台**，提供了可视化的 Prompt 编排、RAG（检索增强生成）、Agent 智能体等功能。它让非程序员也能快速构建 AI 应用。

### 10.2 Dify 与本模块的关系

在本模块中，Dify 作为可选的 LLM 编排层，可以帮助学生构建更复杂的自然语言控车逻辑：

![Dify LLM 应用平台](/resource/oss/download/2082034947585323009)

<img src="images/dify-simple.svg" width="80" alt="Dify"/>

### 10.3 Dify 离线部署

Dify 同样支持离线部署，通过 `docker save/load` 方式导入镜像包：

```
dify/
├── images/dify-images.tar     ← 离线镜像包
├── config/                    ← 配置文件
├── load-and-start.sh          ← 一键启动脚本
└── 操作说明.md
```

---

## 11. Linux 系统部署基础

### 11.1 为什么选择 Ubuntu 22.04

| 原因 | 说明 |
|:---|:---|
| **ROS2 Humble 官方支持** | ROS2 Humble 以 Ubuntu 22.04 为 Tier-1 支持平台 |
| **LTS 长期支持** | 安全更新至 2027 年，稳定性有保障 |
| **Docker 兼容性** | Docker 官方提供 Ubuntu 22.04 的完整支持 |
| **内核版本** | 5.15 内核，对虚拟化和容器原生支持良好 |

<img src="images/ubuntu-bp.svg" width="60" alt="Ubuntu"/> <img src="images/linux-bp.svg" width="60" alt="Linux"/>

### 11.2 虚拟化技术要求

Docker 和仿真环境对硬件虚拟化有要求：

```
BIOS/UEFI 设置中需启用：

VT-x (Intel) / AMD-V (AMD)      ← CPU 虚拟化扩展
    └── 虚拟机中运行 Docker 时必需

虚拟机软件（如 VirtualBox）中：
    └── 设置 → 系统 → 处理器 → 启用 VT-x/AMD-V
```

### 11.3 U 盘启动安装流程

![Ubuntu 部署流程](/resource/oss/download/2082034947895701506)

### 11.4 常用 Linux 命令速查

```bash
# ── 系统信息 ──
lsb_release -a          # 查看 Ubuntu 版本
uname -r                # 查看内核版本
df -h                   # 磁盘使用情况
free -h                 # 内存使用情况

# ── 用户与权限 ──
sudo <command>          # 以管理员权限执行
usermod -aG docker $USER # 将用户加入 docker 组
groups $USER            # 查看用户所属组

# ── 文件操作 ──
mkdir -p /path/to/dir   # 递归创建目录
cp -r src dst           # 递归复制
rsync -a src/ dst/      # 同步目录（保留属性）
chmod +x script.sh      # 添加执行权限

# ── 进程与端口 ──
ps aux                  # 查看所有进程
lsof -i :8888           # 查看占用 8888 端口的进程
netstat -tlnp           # 查看所有监听端口
```

---

## 12. 离线部署策略

### 12.1 为什么需要离线部署

考场环境通常**无法访问外网**，这意味着所有软件和依赖必须提前准备并打包。离线部署是工程化 AI 系统的必备能力。

### 12.2 离线部署四阶段

![离线部署四阶段](/resource/oss/download/2082034948155748353)

### 12.3 镜像导出/导入完整流程

#### 有网环境（教师/管理员）：构建并导出

```bash
# 1. 构建 Docker 镜像
cd jushenai-vehicle-agent
docker compose build

# 2. 导出为 tar 文件
docker save -o ydy-eia-vehicle-agent.tar ydy-eia/vehicle-agent:latest
docker save -o ydy-eia-sim-web.tar ydy-eia/sim-web:latest

# 3. 生成分发清单
ls -lh ydy-eia-*.tar > MANIFEST.txt
```

#### 离线环境（考场/学生）：导入并启动

```bash
# 1. 导入镜像
docker load -i ydy-eia-vehicle-agent.tar
docker load -i ydy-eia-sim-web.tar

# 2. 验证导入
docker images | grep ydy-eia

# 3. 启动服务
cd docker && ./up.sh
```

### 12.4 离线部署常见问题与对策

| 问题 | 原因 | 对策 |
|:---|:---|:---|
| Docker 未安装 | 离线无法 `apt install` | 教师预装或提供离线 deb 包 |
| 镜像导入失败 | tar 文件损坏或空间不足 | 清理磁盘 + 校验 MD5 |
| 端口冲突 | 已有服务占用端口 | 修改 `.env` 文件更换端口 |
| 容器启动失败 | 虚拟化未开启 | BIOS 开启 VT-x/AMD-V |
| 仿真无数据 | Gazebo 初始化未完成 | 等待 90 秒后重试 |

### 12.5 离线包制作自动化

本模块提供 `package-exam.sh` 脚本自动化上述流程：

```bash
# 完整打包
./scripts/package-exam.sh

# 指定 ISO 文件
./scripts/package-exam.sh --iso /path/to/ubuntu-22.04-desktop-amd64.iso

# 仅更新代码，跳过镜像构建
./scripts/package-exam.sh --skip-build

# 同时打包 Dify
./scripts/package-exam.sh --dify
```

<!-- edu-oss-embedded:2082034945467199490,2082034945744023554,2082034946008264706,2082034946268311554,2082034946532552705,2082034946796793857,2082034947065229314,2082034947325276161,2082034947585323009,2082034947895701506,2082034948155748353 -->
