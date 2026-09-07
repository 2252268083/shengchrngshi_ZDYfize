# 模块3 具身智能平台部署与运维 · 实操手册

适用镜像：**`ydy-eia-jzbase`**（Ubuntu 22.04 桌面 + noVNC + VS Code + Docker）。
仿真对象：**ROS 2 turtlesim**。本环境不预装 ROS / Dify / OpenClaw，需按本手册从文件服务器下载并完成部署。

---

## 任务准备

### 1. 环境确认


| 项目     | 说明                                                  |
| -------- | ----------------------------------------------------- |
| 平台镜像 | `ydy-eia-jzbase`                                      |
| 桌面账号 | `ubuntu` / `ubuntu`                                   |
| 已预装   | MATE 桌面、VS Code、Epiphany、Docker Engine / Compose |
| 需自装   | ROS 2 Humble、Dify、OpenClaw                          |
| 运行要求 | 实验容器为特权模式（平台已配置）                      |

进入桌面后执行：

```bash
cat ~/Desktop/实训说明.txt
docker --version
docker compose version
docker info | head -8
```

若 `docker info` 失败，等待约 10 秒待内置 dockerd 启动后重试。

### 2. 文件服务器(查看实验手册前实际地址）

```bash
export FILE_SERVER=http://192.168.0.105:8088
```

若无法访问，以平台「实训资源 → 资源下载」页复制的链接为准。

服务器目录约定：

```text
$FILE_SERVER/module3/
├── scripts/
│   ├── install-ros2.sh
│   ├── install-dify.sh
│   └── install-openclaw.sh
├── ros2/
│   ├── ros.key
│   └── debs/… 或 ros2-debs.tar.gz
├── dify/
│   ├── images/dify-images.tar
│   └── config/…
└── openclaw/
    └── openclaw-image.tar
```

### 3. 下载安装包

```bash
export FILE_SERVER=http://192.168.0.105:8088
mkdir -p ~/lab-m3/{scripts,ros2,dify/images,dify/config,openclaw}
cd ~/lab-m3

curl -fL "$FILE_SERVER/module3/scripts/install-ros2.sh" -o scripts/install-ros2.sh
curl -fL "$FILE_SERVER/module3/scripts/install-dify.sh" -o scripts/install-dify.sh
curl -fL "$FILE_SERVER/module3/scripts/install-openclaw.sh" -o scripts/install-openclaw.sh
chmod +x scripts/*.sh

curl -fL "$FILE_SERVER/module3/ros2/ros.key" -o ros2/ros.key || true
# wget -c "$FILE_SERVER/module3/ros2/ros2-debs.tar.gz" -O ros2/ros2-debs.tar.gz
# tar xf ros2/ros2-debs.tar.gz -C ros2/

wget -c "$FILE_SERVER/module3/dify/images/dify-images.tar" -O dify/images/dify-images.tar
wget -c "$FILE_SERVER/module3/openclaw/openclaw-image.tar" -O openclaw/openclaw-image.tar
# 若提供 Dify 配置包：
# wget -c "$FILE_SERVER/module3/dify/dify-config.tar.gz" -O /tmp/dify-config.tar.gz
# tar xf /tmp/dify-config.tar.gz -C dify/

ls -lh scripts ros2 dify/images openclaw
```

说明：`scripts/` 与 `ros2/`、`dify/`、`openclaw/` 须保持同级，供安装脚本识别路径。

### 4. 本手册任务一览


| 任务   | 内容                               |
| ------ | ---------------------------------- |
| 任务 1 | 编写 docker-compose 并验证         |
| 任务 2 | 安装 ROS 2 Humble 并验证 turtlesim |
| 任务 3 | 创建 colcon 工作空间并发布速度指令 |
| 任务 4 | 部署 Dify 并配置 DeepSeek          |
| 任务 5 | 部署 OpenClaw 并验证大模型对话     |
| 任务 6 | OpenClaw 控制 turtlesim 端到端验证 |

---

## 任务 1　编写 docker-compose 并验证

### 目标

在已安装 Docker 的环境下，编写 `docker-compose.yml`，启动容器并确认服务可访问。

### 操作步骤

```bash
mkdir -p ~/lab-compose && cd ~/lab-compose

cat > docker-compose.yml <<'EOF'
services:
  demo-web:
    image: nginx:alpine
    container_name: lab-demo-nginx
    ports:
      - "8080:80"
    restart: unless-stopped
EOF

# 若本地无该镜像：docker pull nginx:alpine
docker compose up -d
docker compose ps
curl -sI http://127.0.0.1:8080 | head -5
```

桌面浏览器打开：`http://127.0.0.1:8080`。

### 预期结果

- `docker compose ps` 中服务状态为 running
- 浏览器或 curl 可访问 Nginx 页面

可选清理：`docker compose down`

---

## 任务 2　安装 ROS 2 Humble 并验证 turtlesim

### 目标

安装 ROS 2 Humble（含 turtlesim），确认图形仿真可启动。

### 操作步骤

```bash
cd ~/lab-m3
./scripts/install-ros2.sh
source /opt/ros/humble/setup.bash
ros2 pkg executables turtlesim | head
```

有 `ros2/debs/*.deb` 时优先离线安装；否则脚本使用清华源（需能访问镜像站）。

**终端 1** 启动仿真：

```bash
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtlesim_node
```

### 预期结果

- 命令 `ros2`、`turtlesim` 可用
- 桌面出现 turtlesim 窗口

---

## 任务 3　创建 colcon 工作空间并发布速度指令

### 目标

创建 ament_python 功能包，编写节点向海龟发布 `Twist`，用 `colcon build` 编译并运行验证。

### 操作步骤

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python cmd_vel_pub --dependencies rclpy geometry_msgs
```

编辑 `~/ros2_ws/src/cmd_vel_pub/cmd_vel_pub/publisher.py`：

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.on_timer)
        self.t0 = self.get_clock().now()
        self.get_logger().info('publishing to /turtle1/cmd_vel')

    def on_timer(self):
        elapsed = (self.get_clock().now() - self.t0).nanoseconds / 1e9
        msg = Twist()
        if elapsed < 3.0:
            msg.linear.x = 1.0
            msg.angular.z = 0.5
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

在 `setup.py` 中注册入口：

```python
entry_points={
    'console_scripts': [
        'publisher = cmd_vel_pub.publisher:main',
    ],
},
```

编译并运行（保持 turtlesim 已启动）：

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 run cmd_vel_pub publisher
```

另开终端观察：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /turtle1/cmd_vel --once
```

如需话题名 `/cmd_vel`：

```bash
ros2 run turtlesim turtlesim_node --ros-args -r /turtle1/cmd_vel:=/cmd_vel
ros2 run cmd_vel_pub publisher --ros-args -r /turtle1/cmd_vel:=/cmd_vel
```

### 预期结果

- `colcon build` 成功
- 海龟出现运动，或 `ros2 topic echo` 能收到 `Twist`

---

## 任务 4　部署 Dify 并配置 DeepSeek

### 目标

使用离线镜像包通过 Docker Compose 启动 Dify，完成管理员注册，并接入 DeepSeek 完成对话。

### 操作步骤

**1）启动 Dify**

```bash
cd ~/lab-m3
# 确认：dify/images/dify-images.tar 、 dify/config/docker-compose.yaml
./scripts/install-dify.sh
```

浏览器访问：

- 首次：`http://127.0.0.1:3000/install`
- 登录：`http://127.0.0.1:3000/signin`

查看状态（启动较慢时）：

```bash
cd ~/lab-m3/dify/config
docker compose ps
```

**2）配置 DeepSeek**

1. 进入「设置 → 模型供应商」
2. 添加 DeepSeek（或 OpenAI 兼容接口）
3. 填写教师提供的 API Key、Base URL（如 `https://api.deepseek.com/v1`）
4. 创建「聊天助手」应用，选择该模型并发送测试消息

### 预期结果

- Dify 页面可打开并完成注册/登录
- 模型配置成功，聊天有返回

---

## 任务 5　部署 OpenClaw 并验证大模型对话

### 目标

加载 OpenClaw 镜像并启动 Gateway，在 Control UI 中完成 Token 连接与 DeepSeek 对话验证。

### 操作步骤

**1）启动 Gateway**

```bash
cd ~/lab-m3
./scripts/install-openclaw.sh
# 请保存终端打印的 Gateway Token
```

浏览器打开 `http://127.0.0.1:18789/`，粘贴 Token，点击 Connect。

按界面提示配置 DeepSeek API Key。

**2）探活脚本（可选）**

```bash
mkdir -p ~/lab-m3/tests && cd ~/lab-m3/tests
cat > openclaw_smoke.py <<'PY'
#!/usr/bin/env python3
import os, sys, urllib.request, urllib.error

BASE = os.environ.get("OPENCLAW_URL", "http://127.0.0.1:18789").rstrip("/")
TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")

def main():
    try:
        with urllib.request.urlopen(BASE + "/", timeout=10) as r:
            print(f"OK: HTTP {r.status}, bytes={len(r.read())}")
    except urllib.error.URLError as e:
        print("FAIL:", e)
        return 1
    print("浏览器粘贴 Token 后 Connect。Token =", TOKEN or "(请填入安装脚本输出值)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
PY

OPENCLAW_GATEWAY_TOKEN='<安装脚本输出的Token>' python3 openclaw_smoke.py
```

在 Control UI 发送「你好」，确认模型有回复。

### 预期结果

- Gateway 页面可访问
- Token 连接成功
- 配置 Key 后对话正常

---

## 任务 6　OpenClaw 控制 turtlesim 端到端验证

### 目标

编写本地控龟脚本，先在终端验证，再通过 OpenClaw 自然语言触发，完成端到端闭环。

### 操作步骤

**1）编写控龟脚本**

```bash
mkdir -p ~/lab-m3/turtle && cd ~/lab-m3/turtle

cat > drive_turtle.py <<'PY'
#!/usr/bin/env python3
import sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

MAP = {
    "前进": (1.0, 0.0, 2.0),
    "后退": (-1.0, 0.0, 2.0),
    "左转": (0.0, 1.5, 1.5),
    "右转": (0.0, -1.5, 1.5),
    "停": (0.0, 0.0, 0.3),
}

class Driver(Node):
    def __init__(self):
        super().__init__("drive_turtle_cli")
        self.pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)

    def run(self, text: str):
        key = text.strip()
        for k, v in MAP.items():
            if k in key:
                lx, az, dur = v
                break
        else:
            print("可用: 前进/后退/左转/右转/停")
            return
        msg = Twist()
        msg.linear.x, msg.angular.z = lx, az
        end = time.time() + dur
        while time.time() < end:
            self.pub.publish(msg)
            time.sleep(0.1)
        stop = Twist()
        for _ in range(5):
            self.pub.publish(stop)
            time.sleep(0.05)

def main():
    rclpy.init()
    node = Driver()
    node.run(" ".join(sys.argv[1:]) or "前进")
    node.destroy_node()
    rclpy.shutdown()

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
```

**2）终端验证**

终端 1：

```bash
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtlesim_node
```

终端 2：

```bash
~/lab-m3/turtle/drive_turtle.sh 前进
~/lab-m3/turtle/drive_turtle.sh 停
```

**3）接入 OpenClaw**

配置 Skill（或按教师提供的 Skill 包），使自然语言映射到：

```bash
/home/ubuntu/lab-m3/turtle/drive_turtle.sh "前进"
```

在 OpenClaw 面板发送「让海龟前进」，观察仿真窗口。

### 预期结果

- 脚本可直接控制海龟
- OpenClaw 自然语言触发后海龟产生对应运动

---

## 附录 A　常用地址


| 服务       | 地址                                     |
| ---------- | ---------------------------------------- |
| 文件服务器 | http://192.168.0.105:8088/               |
| Dify       | http://127.0.0.1:3000/install 、 /signin |
| OpenClaw   | http://127.0.0.1:18789/                  |

---

## 附录 B　故障排查


| 现象                | 处理                                                        |
| ------------------- | ----------------------------------------------------------- |
| `docker info` 失败  | 等待 dockerd；确认特权实验                                  |
| 文件服务器下载失败  | 使用局域网 IP；查看资源下载页                               |
| 找不到`ros2`        | 先完成任务 2；新终端执行`source /opt/ros/humble/setup.bash` |
| Dify 启动缓慢       | `cd ~/lab-m3/dify/config && docker compose ps`，耐心等待    |
| OpenClaw 要求 Token | 使用`install-openclaw.sh` 打印的 Token                      |
| 海龟无运动          | 确认 turtlesim 已启动；话题为`/turtle1/cmd_vel`             |

---

## 附录 C　教师说明

- 本模块仅绑定镜像 **`ydy-eia-jzbase`**
- 将 `lab-images/ydy-eia-jzbase/packages/` 中 ros2、dify、openclaw、scripts 同步至文件服务器 `module3/`
- DeepSeek Key 勿放入公开下载目录