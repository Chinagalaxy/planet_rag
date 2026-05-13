import requests
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import os
import time

load_dotenv()

api_key = os.getenv("MINERU_API_KEY")
post_url = "https://mineru.net/api/v4/file-urls/batch"
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def download_file(url, save_folder, filename):
    # 1. 确保目标文件夹存在（如果不存在则自动创建）
    os.makedirs(save_folder, exist_ok=True)
    
    # 拼接完整的保存路径
    file_path = os.path.join(save_folder, filename)
    
    print(f"正在下载文件到: {file_path}")
    
    # 2. 发起下载请求 (stream=True 是关键，表示分块下载)
    response = requests.get(url, stream=True)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 获取文件总大小（用于进度条显示）
        total_size = int(response.headers.get('content-length', 0))
        
        # 3. 写入文件
        with open(file_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        print("下载完成！")
    else:
        print(f"下载失败，状态码: {response.status_code}")

    extract_folder = os.path.join(save_folder, os.path.splitext(filename)[0])
    os.makedirs(extract_folder, exist_ok=True)

    print(f"正在解压到: {extract_folder} ...")
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print("解压完成。")
        
        os.remove(file_path)
        print(f"已删除原始压缩包: {filename}")
        
    except zipfile.BadZipFile:
        print("解压失败：文件可能不是有效的 ZIP 格式，或者下载不完整。")


def parse_by_file(file_path, save_folder):
    """通过文件上传提交文档解析任务并等待结果。"""
    folder = Path(file_path)
    pdfs = list(folder.glob('*.pdf'))
    pdf_names = [pdf.name for pdf in pdfs]
    data = {
        "files": [
            {"name": pdf_name, "data_id": f"00{idx+1}"} for idx, pdf_name in enumerate(pdf_names)
        ],
        "model_version": "vlm",
    }
    pdf_paths = [file_path / pdf_name for pdf_name in pdf_names]
    response = requests.post(post_url,headers=header,json=data)
    batch_id = None
    if response.status_code == 200:
        result = response.json()
        print('response success.')
        if result["code"] == 0:
            batch_id = result["data"]["batch_id"]
            urls = result["data"]["file_urls"]
            for i in range(0, len(urls)):
                with open(pdf_paths[i], 'rb') as f:
                    requests.put(urls[i], data=f)
        else:
            print('apply upload url failed,reason:{}'.format(result["msg"]))

    return poll_result(batch_id, pdf_names, save_folder)


def poll_result(batch_id, pdf_names, save_folder, timeout=300, interval=3):
    """轮询查询解析结果。"""
    state_labels = {
        "uploading": "文件下载中",
        "pending": "排队中",
        "running": "解析中",
        "waiting-file": "等待文件上传",
    }
    get_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    for id, pdf_name in enumerate(pdf_names):
        start = time.time()
        flag = False
        while time.time() - start < timeout:
            resp = requests.get(get_url, headers=header)
            result = resp.json()
            state = result['data']['extract_result'][id]['state']
            elapsed = int(time.time() - start)
            if state == "done":
                full_zip_url = result["data"]["extract_result"][id]["full_zip_url"]
                print(f"[{elapsed}s] {pdf_name}解析完成, Zip 下载链接: {full_zip_url}")
                base_name = os.path.splitext(pdf_name)[0]
                download_file(full_zip_url, save_folder, base_name + '.zip')
                flag = True
                break

            if state == "failed":
                print(f"[{elapsed}s] {pdf_name}解析失败: {result['data']['extract_result'][id].get('err_msg', '未知错误')}")
                flag = True
                break
            
            print(f"[{elapsed}s] {pdf_name}: {state_labels.get(state, state)}...")
            time.sleep(interval)
        if flag == False:
            print(f"[{elapsed}s] {pdf_name}轮询超时 ({timeout}s)")

# 使用示例
#content = parse_by_file("E:/ML/agent/rag/PDFs", "E:/ML/agent/rag/data")