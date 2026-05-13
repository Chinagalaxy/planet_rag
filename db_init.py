import MinerU_agent as MUA
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from vision import get_img_descriptions
import json
import re

def get_img_info(json_files: list[Path], pdf_names: list[str]) -> dict:
    '''
        从json文件中提取图片路径和图片描述
        :param json_files: json文件列表
        :param pdf_names: pdf文件名列表
        :return: 包含图片路径和图片描述的字典
    '''
    all_papers_data = {}
    for id, json_file in enumerate(json_files):
        with open(json_file, "r", encoding='utf-8') as f:
            paper_data = json.load(f)

        img_paths = []
        img_captions = []

        cnt = 1
        for item in paper_data:
            if item.get("type") == "image":
                img_path = item.get("img_path", [])
                img_caption = item.get("image_caption", [])
                for j, caption in enumerate(img_caption):
                    if not re.search(r'(fig|figure)', caption, re.IGNORECASE):
                        img_caption[j] = "" 
                caption_text = "".join(img_caption).strip() if img_caption else ""
                img_caption = caption_text
                if img_caption == "":
                    cnt += 1
                else:
                    for i in range(cnt):
                        img_captions.append(img_caption)
                    cnt = 1
                img_paths.append(img_path)

        all_papers_data[id] = {
            "paper_name": pdf_names[id],
            "img_paths": img_paths,
            "img_captions": img_captions
        }

    return all_papers_data

def chunk_markdown(
    md_files: list[Path], PDF_files: list[Path], 
    pdf_names: list[str], DATA_path: list[Path], 
    chunk_size: int = 800, chunk_overlap: int = 100
    ) -> list[Document]:
    """
    将 Markdown 文件按标题层级分割，再将每个块递归分块成指定大小的片段。
    
    参数:
        md_files     : Markdown 文件路径列表
        PDF_files    : 对应的 PDF 文件路径列表
        pdf_names    : 论文名称列表（用于元数据）
        DATA_path    : 每个论文的根目录路径列表（用于图片等资源的定位）
        chunk_size   : 文本块最大字符数
        chunk_overlap: 文本块间重叠字符数
    
    返回:
        List[Document] : 所有文本块构成的 Document 列表，每个 Document 的 metadata 包含：
            - 标题层级 (Header 1/2/3)
            - data_path / md_path / pdf_path / paper_name
            - type = "text"
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    all_docs = []
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    for id, md_file in enumerate(md_files):
        with open(md_file, "r", encoding="utf-8") as f:
            markdown_text = f.read()
        header_docs = md_splitter.split_text(markdown_text)
        for doc in header_docs:
            meta = doc.metadata.copy()
            meta["data_path"] = str(DATA_path[id].as_posix())
            meta["md_path"] = str(md_files[id].as_posix())
            meta["pdf_path"] = str(PDF_files[id].as_posix())
            meta["paper_name"] = pdf_names[id]
            meta["type"] = "text"
            chunk_docs = text_splitter.split_text(doc.page_content)
            for chunk_doc in chunk_docs:
                new_doc = Document(
                    page_content=chunk_doc,
                    metadata=meta.copy(),
                )
                all_docs.append(new_doc)
    return all_docs

def load_img_from_json(json_file: Path, DATA_path: list[Path], PDF_files: list[Path]) -> list[Document]:
    '''
    从包含图片描述的 JSON 中构建图片 Document 列表。
    参数：
        json_file     : all_papers_data_vision.json 文件路径
        DATA_path     : 每篇论文数据文件夹所在的父目录
        PDF_files    : 所有 PDF 文件所在的目录
    返回：
        List[Document] 每个元素代表一张图片及其描述
    '''
    with open(json_file, "r", encoding='utf-8') as f:
        paper_data = json.load(f)
    img_docs = []
    for paper_key, paper_info in paper_data.items():
        paper_key = int(paper_key)
        paper_name = paper_info["paper_name"]
        img_paths = paper_info.get("img_paths", [])
        img_captions = paper_info.get("img_captions", [])
        img_descriptions = paper_info.get("img_descriptions", [])

        for idx, (img_path, caption, description) in enumerate(zip(img_paths, img_captions, img_descriptions)):   
            full_img_path = DATA_path[paper_key].as_posix() + "/" + img_path
            if description != "无可识别内容":
                final_img_caption = f"图例：({caption})\n图片描述:({description})\n图片路径:({full_img_path})"
            else:
                final_img_caption = f"图例：({caption})\n图片路径:({full_img_path})"

            meta = {
                "type": "image",
                "img_id": idx,
                "img_paths": full_img_path,
                "img_caption": caption,
                "pdf_path": str(PDF_files[paper_key].as_posix()),
                "paper_name": paper_name,
            }
            new_doc = Document(
                page_content=final_img_caption,
                metadata=meta.copy(),
            )
            img_docs.append(new_doc)
    return img_docs



def init_db():

    BASE_PATH = Path("E:/ML/RAG/")
    PDF_Folder = BASE_PATH / "PDFs"
    MD_Folder = BASE_PATH / "data"
    PDF_files = list(PDF_Folder.glob("*.pdf"))
    IMG_Folder = list(MD_Folder.glob("*/images"))
    MD_files = list(MD_Folder.glob("*/*.md"))
    json_files = list(MD_Folder.glob("*/*content_list.json"))
    pdf_names = [pdf_name.stem.split(" - ")[2].strip() for pdf_name in PDF_files]
    DATA_path = [d for d in MD_Folder.glob("*") if d.is_dir()]

    MUA.parse_by_file(PDF_Folder, MD_Folder)# 解析PDF文件并保存到data文件夹中
    print("PDF文件解析完成")
    all_papers_data = get_img_info(json_files, pdf_names)
    with open("all_papers_data.json", "w+", encoding="utf-8") as f:
        json.dump(all_papers_data, f, ensure_ascii=False, indent=4)
    print("图片描述提取完成")
    text_docs = chunk_markdown(MD_files, PDF_files, pdf_names, DATA_path, chunk_size=500, chunk_overlap=20)
    print("文本信息分割完成")
    get_img_descriptions("all_papers_data.json") #使用视觉模型得到图片描述,并保存到all_papers_data_vision.json文件中
    img_docs = load_img_from_json("all_papers_data_vision.json", DATA_path, PDF_files)
    print("图片信息分割完成")
    all_docs = text_docs + img_docs
    print("所有文档分割完成")
    return all_docs
