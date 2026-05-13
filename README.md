# 🌕 Planet RAG — 多模态行星科学领域学术论文检索与问答系统

一个面向**行星科学文献**的智能问答系统，能够自动解析 PDF 论文、理解其中的图片与表格，并基于语义和关键词混合检索，用大模型生成精准回答，同时**一键打开原始文件**进行对照。确保可及时溯源，减少模型幻觉，提高回答的可信度。

## ✨ 核心功能

- **📄 本地/云端解析** 使用 MinerU 官方 API 将 PDF 转化为结构化 Markdown，保留表格、公式和图片等源文件。
- **🖼️ 多模态理解** 利用本地bge-m3模型作为文本嵌入模型， Qwen3-VL-4B-Instruct 模型为每张图片生成详细描述，确保图表、照片均可检索。
- **🔍 混合检索** 结合向量语义搜索 (Milvus索引) 与关键词搜索 (BM25)，显著提升召回率和准确性。
- **🧠 深度推理** 接入 DeepSeek 大模型，基于检索到的上下文生成有依据的回答，并自动列出参考文献。
- **📂 一键打开源文件** 支持在终端中通过系统默认程序直接打开回答中引用的论文 PDF 或原始图片，方便查证。
- **📦 完全本地嵌入** 嵌入模型 bge-m3 和视觉模型 Qwen3-VL-4B-Instruct 均运行在本地 GPU (RTX 4070 8GB)，保障数据隐私。
- **🗃️ 向量数据库** 采用 Milvus 存储文本与图片的嵌入向量，支持持久化和高效索引。

## 📐 系统架构
### - 系统实现流程：
用户提问 → 混合检索 (BM25 + Milvus索引) → 获取文本块与图片描述→ 拼接上下文 → DeepSeek 大模型生成答案 → 输出答案 + 可打开文件链接
### - 数据处理流水线：
PDF 文件 → MinerU API 解析 → 自动下载 MD + 图片 → Markdown 分块（基于markdown文件标题+递归分块）→ Qwen3-VL 图片描述（可选） → 嵌入 (bge-m3)→ 存入 Milvus 并构建混合检索器（关键字+向量）

## 💻 系统基础需求
**1、创建.env文件，填写以下内容：**
```
MINERU_API_KEY = your_mineru_api_key
DEEPSEEK_API_KEY = your_deepseek_api_key
DEEPSEEK_BASE_URL = https://api.deepseek.com
MILVUS_TOKEN = your_milvus_token （第一次创建为默认token）
```
**2、下载本地模型，代码中使用文本嵌入模型bge-m3和视觉模型Qwen3VL-4B-Instruct，可选择其他模型进行替换。将模型下载到本文件夹下的models文件夹中。**
**3、准备领域内学术论文PDF数据，保存到当前文件夹下PDFs文件夹内，格式如（可从Zotero直接导出PDF文件）：**
```
Cai et al. - 2025 - Persistent but weak magnetic field at the Moon’s midstage revealed by Chang’e-5 basalt.pdf
Zhang et al. - 2020 - Asymmetric Lunar Magnetic Perturbations Produced by Reflected Solar Wind Particles.pdf
```
**4、根据requirements.txt安装依赖，建议使用conda环境。**
## 📂 文件架构
```
planet_rag/
├── data/                 # 解析后的 Markdown 和图片
├── images/               # README.md 中使用的展示图片
├── models/               # 本地文字嵌入和视觉模型（需自行下载）
├── PDFs/                 # 原始论文 PDF
├── .env                  # API Key 等私密配置
├── db_init.py            # 数据预处理与多模态嵌入
├── MinerU_agent.py       #调用 MinerU 进行 PDF 解析，自动下载 Markdown 文件，json 结构文件和图片，需要 API Key
├── rag.py                # 交互式问答主程序
├── README.md             # 项目说明文档
├── requirements.txt      # 系统运行的基本 Python 依赖
└── vision.py             # 本地图片描述生成器，获得每张图片的图片描述用于后续多模态检索
```
## 🚀 快速开始
在准备好数据、模型等基础需求后，运行rag.py文件，即可开始使用系统。（第一次运行可能较慢，与选择的视觉模型和电脑配置有关）
```
python ./rag.py
```
## 🌟 示例
```
载入bge-m3嵌入模型...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 4802.94it/s]
bge-m3嵌入模型加载完成
=== 问答就绪，输入 'quit' 退出 ===

用户: LSE坐标系是什么？有没有示意图？

助手: LSE（Lunar Solar-wind Electric field）坐标系是一种右手坐标系，原点位于月球中心，其设计旨在便于研究太阳风对流电场相关现象。在该坐标系中，**x轴指向瞬时太阳风速度（\(V_{SW}\)）的反方向**，**y轴与太阳风电场（\(E_{SW}\)）方向相反**（即 \(E_{SW}\) 始终沿 \(-Y_{LSE}\) 方向），而z轴则由右手定则确定。由于该坐标系固定了电场方向，它特别适用于分析太 阳风对流电场驱动的粒子动力学行为 [2·文本] [6·文本]。

关于示意图，图1展示了LSE坐标系与BE坐标系的几何关系，其中明确标注了LSE的坐标轴方向（黑色、灰色、深蓝色箭头）、\(E_{SW}\)（绿色箭头）、\(B_{IMF}\)（红色箭头）以及\(V_{SW}\)（ 黑色箭头）等关键矢量 [4·图片] [8·文本]。

**引用文件**  
- E:/ML/RAG/PDFs/Zhang et al. - 2020 - Asymmetric Lunar Magnetic Perturbations Produced by Reflected Solar Wind Particles.pdf  
- E:/ML/RAG/data/Zhang et al. - 2021 - A Meandering Lunar Wake Produced by the Pickup of Reflected Solar‐Wind Ions/images/9964f71e27854845133db442b741db415cfe07652a12fbb3958619391eb4728f.jpg  
- E:/ML/RAG/PDFs/Zhang et al. - 2021 - A Meandering Lunar Wake Produced by the Pickup of Reflected Solar‐Wind Ions.pdf
是否打开所有引用文件？(y/n): Y
解析到的路径: ['E:/ML/RAG/PDFs/Zhang et al. - 2020 - Asymmetric Lunar Magnetic Perturbations Produced by Reflected Solar Wind Particles.pdf', 'E:/ML/RAG/data/Zhang et al. - 2021 - A Meandering Lunar Wake Produced by the Pickup of Reflected Solar‐Wind Ions/images/9964f71e27854845133db442b741db415cfe07652a12fbb3958619391eb4728f.jpg', 'E:/ML/RAG/PDFs/Zhang et al. - 2021 - A Meandering Lunar Wake Produced by the Pickup of Reflected Solar‐Wind Ions.pdf']
```

![LSE坐标系示意图](images/9964f71e27854845133db442b741db415cfe07652a12fbb3958619391eb4728f(1).jpg)

```
用户: 给我一张月球全球性磁场图

助手: 根据您的要求，以下是一张月球全球性磁场图：

这是一张由Ravat et al.（2020）基于月球探勘者号（Lunar Prospector）和辉夜姬（SELENE/Kaguya）的沿轨磁场梯度数据，通过全球单极模型（0.66°间距）推导出的最高分辨率月球表面总磁场 异常分布图。该图采用Hammer-Aitoff投影，以颜色梯度和对数尺度表示磁场强度（log₁₀ |B|，单位：nT），覆盖整个月球表面。 [4]

**引用文件**
- E:/ML/RAG/data/Ravat et al. - 2020 - Lunar Magnetic Field Models From Lunar Prospector and SELENEKaguya Along‐Track Magnetic Field Gradients/images/7b141f5f7e67fc81e897cbbc281957f4a40f73ad27f3f356167bce60e5a5c399.jpg
是否打开所有引用文件？(y/n): Y
解析到的路径: ['E:/ML/RAG/data/Ravat et al. - 2020 - Lunar Magnetic Field Models From Lunar Prospector and SELENEKaguya Along‐Track Magnetic Field Gradients/images/7b141f5f7e67fc81e897cbbc281957f4a40f73ad27f3f356167bce60e5a5c399.jpg']
```
![月球全球性磁场图](images/7b141f5f7e67fc81e897cbbc281957f4a40f73ad27f3f356167bce60e5a5c399(1).jpg)

## 🧐 不足与展望
- **MinerU 解析仍有缺陷**：在遇到图片紧密相连的 PDF 时，可能分割错误，导致图片描述需依赖原始图例 + 路径以保证唯一性。  
- **本地模型性能受限**：嵌入与视觉模型受本地硬件设施制约，速度与精度不如云端大型模型。  
- **数据集范围有限**：目前实验中仅包含部分月球相关学术论文，后续可以扩展到火星、小行星等其他行星领域并增大数据量。  
- **交互体验待提升**：可以考虑后续开发 Web UI，提供更友好的图形化问答界面。  
