import cv2
import numpy as np
import base64
import requests
import tqdm
from PIL import Image
import io
import streamlit as st  # 轻量化Web界面，无需部署

# ---------------------- 仅需修改这1处：选择API类型并填密钥 ----------------------
API_TYPE = "modelscope"  # 二选一："modelscope"（魔搭） / "gpt4o"（GPT-4o）
API_KEY = "ms-9f99616d-d3cf-4783-922a-1ed9599fec3a"  # 替换成你的密钥（魔搭ms-开头/GPT-4o sk-开头）
# -----------------------------------------------------------------------------

# 1. 工具函数：视频转关键帧
def video_to_keyframes(video_path, frame_interval=1):
    """提取视频关键帧（每秒1帧，减少API消耗）"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = []
    
    with tqdm.tqdm(total=total_frames, desc="提取关键帧") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 每隔frame_interval秒取1帧
            if frame_idx % (fps * frame_interval) == 0:
                # 转换为RGB格式+压缩分辨率（360p）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil.thumbnail((640, 360))  # 缩小尺寸，减少Token消耗
                keyframes.append(frame_pil)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    return keyframes, fps

# 2. 工具函数：图片/帧转Base64
def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

# 3. 工具函数：调用API分析
def analyze_by_api(content_type, base64_list, api_type, api_key):
    """
    content_type: "image"（单张图片） / "video"（多张关键帧）
    base64_list: 图片/帧的Base64列表
    """
    if api_type == "modelscope":
        # 魔搭API配置（Qwen2.5-VL模型，免费）
        url = "https://api-inference.modelscope.cn/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = [{"role": "user", "content": []}]
        
        # 构建提示词
        if content_type == "image":
            prompt = "详细分析这张图片：主体、纹理材质、光影类型、色彩氛围、场景背景，输出简洁易懂的分析结果"
        else:
            prompt = f"以下是视频的{len(base64_list)}个关键帧（按时间顺序），分析：1.核心主体；2.画面风格；3.分镜头切换点；4.运镜手法，输出结构化结果"
        
        # 添加文本提示
        messages[0]["content"].append({"type": "text", "text": prompt})
        # 添加图片/关键帧
        for b64 in base64_list:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        
        payload = {"model": "Qwen/Qwen2.5-VL-72B-Instruct", "messages": messages, "max_tokens": 500}
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        return response.json()["choices"][0]["message"]["content"]
    
    elif api_type == "gpt4o":
        # GPT-4o API配置
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = [{"role": "user", "content": []}]
        
        # 构建提示词
        if content_type == "image":
            prompt = "Analyze this image in detail: subject, texture, lighting type, color atmosphere, background scene. Output in Chinese."
        else:
            prompt = f"These are {len(base64_list)} keyframes from a video (in order). Analyze: 1. Main subject; 2. Visual style; 3. Shot transitions; 4. Camera movement. Output in Chinese."
        
        # 添加文本提示
        messages[0]["content"].append({"type": "text", "text": prompt})
        # 添加图片/关键帧
        for b64 in base64_list:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        
        payload = {"model": "gpt-4o", "messages": messages, "max_tokens": 500}
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        return response.json()["choices"][0]["message"]["content"]

# 4. 轻量化Web界面（Streamlit，本地运行，无需部署）
def main():
    st.set_page_config(page_title="视频&图片解析工具", page_icon="📽️")
    st.title("📽️ 视频&图片解析工具（无ComfyUI依赖）")
    
    # 选择解析类型
    tab1, tab2 = st.tabs(["解析视频", "解析图片"])
    
    with tab1:
        st.subheader("上传视频（MP4/AVI/MKV）")
        video_file = st.file_uploader("支持格式：MP4/AVI/MKV（建议≤500MB）", type=["mp4", "avi", "mkv"])
        
        if video_file:
            # 保存上传的视频到临时文件
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            if st.button("🚀 开始分析视频"):
                with st.spinner("提取关键帧中..."):
                    keyframes, fps = video_to_keyframes(temp_video_path)
                    st.success(f"成功提取{len(keyframes)}个关键帧（每秒1帧）")
                
                with st.spinner("调用API分析...（约10-20秒）"):
                    # 转换关键帧为Base64
                    base64_frames = [image_to_base64(frame) for frame in keyframes]
                    # 调用API
                    result = analyze_by_api("video", base64_frames, API_TYPE, API_KEY)
                    # 显示结果
                    st.subheader("视频分析结果")
                    st.write(result)
    
    with tab2:
        st.subheader("上传图片（JPG/PNG/WebP）")
        img_file = st.file_uploader("支持格式：JPG/PNG/WebP", type=["jpg", "jpeg", "png", "webp"])
        
        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="图片预览", width=300)
            
            if st.button("🚀 开始分析图片"):
                with st.spinner("调用API分析...（约3-5秒）"):
                    # 转换图片为Base64
                    base64_img = [image_to_base64(img)]
                    # 调用API
                    result = analyze_by_api("image", base64_img, API_TYPE, API_KEY)
                    # 显示结果
                    st.subheader("图片分析结果")
                    st.write(result)

if __name__ == "__main__":
    main()