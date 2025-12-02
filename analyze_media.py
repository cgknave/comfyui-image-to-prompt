import cv2
import numpy as np
import base64  # Python内置模块，无需在requirements.txt声明
import requests
import tqdm
from PIL import Image
import io
import streamlit as st

# ---------------------- 仅需确认这1处！----------------------
API_KEY = "ms-9f99616d-d3cf-4783-922a-1ed9599fec3a"  # 你的魔搭API密钥（已正确填写）
# -------------------------------------------------------------

# ---------------------- 界面样式（黑曜石色+圆角方框）----------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #121212;  /* 黑曜石色背景 */
            color: #e0e0e0;
        }
        .feature-card {
            background-color: #1e1e1e;
            border-radius: 20px;  /* 圆角方框 */
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #333;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .section-title {
            color: #8b5cf6;  /* 紫色强调色 */
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 15px;
            border-left: 4px solid #8b5cf6;
            padding-left: 10px;
        }
        .stButton > button {
            background-color: #8b5cf6;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 1rem;
            border: none;
        }
        .stButton > button:hover {
            background-color: #7c3aed;
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
        }
        .stTextArea > div > textarea {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-radius: 10px;
            border: 1px solid #444;
        }
        .stFileUploader > div > div {
            background-color: #2d2d2d;
            border-radius: 10px;
            border: 1px dashed #444;
        }
        .stProgress > div > div > div > div {
            background-color: #8b5cf6;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- 核心工具函数 ----------------------
def video_to_keyframes(video_file):
    """提取视频关键帧（每秒1帧，适配Streamlit云端）"""
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = []
    frame_interval = fps  # 每秒1帧
    
    with st.spinner(f"提取关键帧（共{total_frames}帧）..."):
        progress_bar = st.progress(0)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                # 转换格式+压缩尺寸（减少API消耗）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil.thumbnail((640, 360))
                keyframes.append(frame_pil)
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
    
    cap.release()
    return keyframes, fps

def image_to_base64(image):
    """图片转Base64（魔搭API要求）"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

def analyze_image(image):
    """图片细化分析（主体/材质/光影/色彩/场景）"""
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """详细分析图片，输出结构化结果：
1. 核心主体：人物/物体/动作（精准描述）
2. 纹理材质：表面质感（磨砂/光滑/颗粒感）、材质类型（布料/金属/玻璃）
3. 光影细节：光影类型（伦勃朗光/柔光/硬光）、光源方向、明暗对比
4. 色彩氛围：主色调+辅助色、色彩数值（如#FF6B6B）、色调类型（暖/冷/高饱和）
5. 场景背景：场景类型（森林/城市/室内）、背景层级（近/中/远景）
6. 构图视角：构图规则（三分法/对称）、视角（平视/俯视/特写）
输出简洁明了，分点呈现"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    }
                ]
            }
        ],
        "max_tokens": 600,
        "temperature": 0.6
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def analyze_video(video_file):
    """视频全维度分析（主体/画面/运镜/分镜头）"""
    # 步骤1：提取关键帧
    keyframes, fps = video_to_keyframes(video_file)
    if len(keyframes) == 0:
        return "视频帧提取失败，请更换视频文件"
    
    # 步骤2：关键帧转Base64
    base64_frames = [image_to_base64(frame) for frame in keyframes]
    
    # 步骤3：调用魔搭API分析
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""以下是视频的{len(base64_frames)}个关键帧（按时间顺序），全维度分析：
1. 核心主体：视频中贯穿始终的人物/物体，排除次要元素
2. 画面风格：艺术风格（电影级/动漫/纪实）、色彩基调、整体质感
3. 运镜手法：推/拉/摇/移/跟/升/降/旋转，运镜速度（快/中/慢）
4. 分镜头检测：镜头切换点（对应关键帧序号）、每个镜头的时长（秒）
5. 场景转换：硬切/淡入淡出/叠化等转换方式
输出结构化结果，分点清晰，可直接参考使用"""
                    }
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in base64_frames]
            }
        ],
        "max_tokens": 800,
        "temperature": 0.6
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------------- 页面布局（修正gap参数）----------------------
st.set_page_config(
    page_title="视频&图片解析工具（魔搭API版）",
    page_icon="📽️",
    layout="wide"
)

# 标题
st.markdown("<h1 style='text-align: center; color: #8b5cf6; margin: 20px 0;'>📽️ 视频&图片全维度解析工具</h1>", unsafe_allow_html=True)

# 左右分栏（修正gap为"medium"，Streamlit允许值）
col1, col2 = st.columns(2, gap="medium")

# 左侧：图片分析
with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📷 图片细化分析</div>', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("上传图片（JPG/PNG/WebP）", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="图片预览", use_container_width=True, clamp=True)  # 替换废弃的use_column_width
    img_analyze_btn = st.button("🚀 开始图片分析", type="primary", key="img_btn")
    st.markdown('</div>', unsafe_allow_html=True)

# 右侧：视频分析
with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎬 视频全维度分析</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("上传视频（MP4/AVI/MKV）", type=["mp4", "avi", "mkv"])
    if uploaded_video:
        st.markdown(f"📊 视频信息：{uploaded_video.name}（大小：{round(uploaded_video.size/1024/1024, 2)}MB）")
    video_analyze_btn = st.button("🎯 开始视频分析", type="primary", key="video_btn")
    st.markdown('</div>', unsafe_allow_html=True)

# 结果显示区（同样修正gap为"medium"）
st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns(2, gap="medium")

# 左下：图片分析结果
with col3:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 图片分析结果</div>', unsafe_allow_html=True)
    img_result = st.text_area("分析结果将显示在这里（可直接复制）", height=300, disabled=True, key="img_result")
    
    if img_analyze_btn and uploaded_img:
        try:
            with st.spinner("图片分析中...（约3-5秒）"):
                result = analyze_image(img)
                st.text_area("✅ 图片分析完成", value=result, height=300, key="img_result_active")
        except Exception as e:
            st.error(f"分析失败：{str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# 右下：视频分析结果
with col4:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 视频分析结果</div>', unsafe_allow_html=True)
    video_result = st.text_area("分析结果将显示在这里（可直接复制）", height=300, disabled=True, key="video_result")
    
    if video_analyze_btn and uploaded_video:
        try:
            with st.spinner("视频分析中...（约10-20秒，取决于视频长度）"):
                result = analyze_video(uploaded_video)
                st.text_area("✅ 视频分析完成", value=result, height=300, key="video_result_active")
        except Exception as e:
            st.error(f"分析失败：{str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# 底部说明
st.markdown("""
    <div style="text-align: center; margin: 30px 0; color: #888; font-size: 0.9rem;">
        ---
        🔧 工具说明：基于魔搭Qwen2.5-VL免费API，每日2000次调用额度，视频建议≤500MB
    </div>
""", unsafe_allow_html=True)