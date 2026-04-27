#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tự động đẩy thư mục hiện tại lên GitHub repository.
Yêu cầu: git đã được cài đặt trên máy.
"""

import os
import subprocess
import sys

# ================== CẤU HÌNH ==================
REPO_URL = "https://github.com/az092345626/customultralitic.git"  # URL repository của bạn
TOKEN = ""  # Nhập personal access token vào đây (hoặc để trống nếu đã config)
COMMIT_MSG = "Custom YOLO with SE module"
BRANCH = "main"
# Nếu muốn đẩy từ thư mục khác, hãy đổi đường dẫn bên dưới
TARGET_DIR = os.getcwd()  # Mặc định là thư mục hiện tại
# ==============================================

def run_cmd(cmd, cwd=None):
    """Chạy lệnh shell, trả về output và raise lỗi nếu thất bại."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Lệnh thất bại: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()

def main():
    # 1. Chuyển đến thư mục đích
    os.chdir(TARGET_DIR)
    print(f"Đang làm việc tại: {os.getcwd()}")

    # 2. Kiểm tra git đã được cài chưa
    try:
        run_cmd("git --version")
    except:
        print("Git chưa được cài đặt. Vui lòng cài Git trước.")
        sys.exit(1)

    # 3. Init git nếu chưa có
    if not os.path.exists(".git"):
        print("Khởi tạo git repository...")
        run_cmd("git init")
    else:
        print("Đã tồn tại .git, bỏ qua init.")

    # 4. Tạo .gitignore nếu chưa có
    gitignore_path = ".gitignore"
    if not os.path.exists(gitignore_path):
        print("Tạo file .gitignore cơ bản...")
        with open(gitignore_path, "w") as f:
            f.write("""__pycache__/
*.pyc
.venv/
runs/
dataset/
*.zip
*.pt
*.pth
*.onnx
*.engine
*.log
.DS_Store
""")
    else:
        print("File .gitignore đã tồn tại, giữ nguyên.")

    # 5. Xử lý token và remote URL
    if TOKEN:
        # Chèn token vào URL
        repo_url_with_token = REPO_URL.replace("https://", f"https://{TOKEN}@")
    else:
        repo_url_with_token = REPO_URL

    # Kiểm tra remote origin đã tồn tại chưa
    remotes = run_cmd("git remote -v")
    if "origin" not in remotes:
        print("Thêm remote origin...")
        run_cmd(f"git remote add origin {repo_url_with_token}")
    else:
        print("Remote origin đã tồn tại, cập nhật URL...")
        run_cmd(f"git remote set-url origin {repo_url_with_token}")

    # 6. Add tất cả file
    print("Thêm file vào staging...")
    run_cmd("git add .")

    # 7. Commit
    print("Commit thay đổi...")
    run_cmd(f'git commit -m "{COMMIT_MSG}"')

    # 8. Đổi tên nhánh nếu cần
    current_branch = run_cmd("git rev-parse --abbrev-ref HEAD")
    if current_branch != BRANCH:
        print(f"Đổi tên nhánh từ {current_branch} thành {BRANCH}...")
        run_cmd(f"git branch -M {BRANCH}")

    # 9. Push lên GitHub
    print("Đang đẩy code lên GitHub...")
    run_cmd(f"git push -u origin {BRANCH}")

    print("✅ Thành công! Code đã được đẩy lên repository.")

if __name__ == "__main__":
    main()