# -*- coding: utf-8 -*-
"""下载课程文档中的所有图片并修改md文件路径"""
import os
import re
import sys
import urllib.parse
import urllib.request

BASE = "http://180.114.109.253:8443"
CLIENT_ID = "e5cd7e4891bf95d1d19206ce24a7b32e"
TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2dpblR5cGUiOiJsb2dpbiIsImxvZ2luSWQiOiJzeXNfdXNlcjoyMDgyMDAwMDAwMDAwMDAwMDQ2Iiwicm5TdHIiOiI4OFB3WUZ6dFc4YWxTbVp1Q01VYkVEZUdkR2Y5c1BqcSIsImNsaWVudGlkIjoiZTVjZDdlNDg5MWJmOTVkMWQxOTIwNmNlMjRhN2IzMmUiLCJ0ZW5hbnRJZCI6IjAwMDAwMCIsInVzZXJJZCI6MjA4MjAwMDAwMDAwMDAwMDA0NiwidXNlck5hbWUiOiIxOTgzMjk4NzI4NiIsImRlcHRJZCI6MjA4MTY0MTQwMDE4ODM5OTYxOCwiZGVwdE5hbWUiOiLph5HnoJbotZvor5XnlKgiLCJkZXB0Q2F0ZWdvcnkiOiIifQ.3ww1g3o0aZCFnqbXLAbLb1oBVbXFG0Ae6l8jUmrWDvg"

ROOT = os.path.dirname(os.path.abspath(__file__))
MODULES = [
    ("module1_fer", ["basics.md", "manual.md"]),
    ("module2_har", ["basics.md", "manual.md"]),
    ("module3_embodied_ai", ["basics.md", "manual.md"]),
    ("module4_agent", ["basics.md", "manual.md"]),
]

IMG_PATTERN = re.compile(r"/resource/oss/download/(\d+)")


def download_image(img_id, out_path):
    """下载单张图片 - 使用 material/preview 接口"""
    url = f"{BASE}/prod-api/edu/student/material/preview/{img_id}?clientid={CLIENT_ID}&Authorization={urllib.parse.quote(TOKEN)}&inline=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            if not data or len(data) < 100:
                print(f"  [FAIL] {img_id}: too small ({len(data)} bytes)")
                return None
            # 检测是否是JSON错误响应
            if data[:1] == b"{":
                print(f"  [FAIL] {img_id}: {data[:100].decode('utf-8', errors='replace')}")
                return None
            # 检测图片类型
            ext = ".png"
            if data[:3] == b"\xff\xd8\xff":
                ext = ".jpg"
            elif data[:4] == b"\x89PNG":
                ext = ".png"
            elif data[:4] == b"GIF8":
                ext = ".gif"
            elif data[:4] == b"RIFF":
                ext = ".webp"
            final_path = out_path + ext
            with open(final_path, "wb") as f:
                f.write(data)
            return ext
    except Exception as e:
        print(f"  [FAIL] {img_id}: {e}")
        return None


def main():
    auth_q = f"clientid={CLIENT_ID}&Authorization={urllib.parse.quote(TOKEN)}"
    total_ok = 0
    total_fail = 0

    for mod_dir, files in MODULES:
        mod_path = os.path.join(ROOT, mod_dir)
        img_dir = os.path.join(mod_path, "images")
        os.makedirs(img_dir, exist_ok=True)
        print(f"\n=== {mod_dir} ===")

        for fname in files:
            fpath = os.path.join(mod_path, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            ids = IMG_PATTERN.findall(content)
            if not ids:
                continue
            print(f"  {fname}: {len(ids)} images")

            replacements = {}
            for img_id in ids:
                out_base = os.path.join(img_dir, img_id)
                if os.path.exists(out_base + ".png"):
                    ext = ".png"
                elif os.path.exists(out_base + ".jpg"):
                    ext = ".jpg"
                else:
                    ext = download_image(img_id, out_base)
                if ext:
                    replacements[img_id] = f"images/{img_id}{ext}"
                    total_ok += 1
                    print(f"    [OK] {img_id}{ext} ({os.path.getsize(out_base + ext)//1024}KB)")
                else:
                    total_fail += 1

            # 替换md中的路径
            for img_id, new_path in replacements.items():
                old = f"/resource/oss/download/{img_id}"
                content = content.replace(old, new_path)

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  {fname}: paths updated")

    print(f"\n[DONE] OK:{total_ok} FAIL:{total_fail}")


if __name__ == "__main__":
    main()
