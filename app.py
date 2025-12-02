import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2
import numpy as np
from tqdm import tqdm
import time

# ---------------------- 仅需修改这1处！----------------------
API_KEY = "ms-9f99616d-d3cf-4783-922a-1ed9599fec3a"  # 替换成你的魔搭Token（以ms-开头）
# -------------------------------------------------------------

# 魔搭API固定配置（无需修改）
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
API_URL = "https://api-inference.modelscope.cn/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ---------------------- 自定义CSS（界面优化核心）----------------------
st.markdown("""
    <style>
        /* 黑曜石色背景 */
        .stApp {
            background-color: #121212;
            color: #e0e0e0;
        }
        /* 圆角功能方框 */
        .feature-card {
            background-color: #1e1e1e;
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #333;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        /* 标题样式 */
        .section-title {
            color: #8b5cf6;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 15px;
            border-left: 4px solid #8b5cf6;
            padding-left: 10px;
        }
        /* 按钮样式 */
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
        /* 文本框样式 */
        .stTextArea > div > textarea {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-radius: 10px;
            border: 1px solid #444;
        }
        /* 上传组件样式 */
        .stFileUploader > div > div {
            background-color: #2d2d2d;
            border-radius: 10px;
            border: 1px dashed #444;
        }
        /* 进度条样式 */
        .stProgress > div > div > div > div {
            background-color: #8b5cf6;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- 核心工具函数 ----------------------
def image_to_base64(image):
    """图片转Base64（魔搭API要求）"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

def call_moda_api(prompt, image_base64=None):
    """调用魔搭API生成结果"""
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": []}],
        "max_tokens": 500,
        "temperature": 0.6,
        "top_p": 0.9
    }
    
    # 添加文本提示
    payload["messages"][0]["content"].append({"type": "text", "text": prompt})
    
    # 添加上传图片（如有）
    if image_base64:
        image_url = f"data:image/jpeg;base64,{image_base64}"
        payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"API调用失败：{str(e)}"

def extract_video_frames(video_file, frame_interval=30):
    """提取视频关键帧（每隔frame_interval帧取1帧）"""
    frames = []
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    with st.spinner(f"提取视频关键帧（共{total_frames}帧，每隔{frame_interval}帧取1帧）..."):
        progress_bar = st.progress(0)
        for frame_idx in tqdm(range(0, total_frames, frame_interval), desc="提取帧"):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            if ret:
                # 转换为PIL图像（RGB格式）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            progress_bar.progress(min((frame_idx / total_frames), 1.0))
    
    video.release()
    st.success(f"成功提取{len(frames)}个关键帧！")
    return frames, fps

def analyze_video_frames(frames):
    """分析视频关键帧，提取画面信息"""
    frame_analyses = []
    progress_bar = st.progress(0)
    
    with st.spinner("分析关键帧画面内容..."):
        for i, frame in enumerate(frames):
            frame_base64 = image_to_base64(frame)
            # 帧分析提示词（细化版）
            frame_prompt = """
            详细分析这一帧画面：
            1. 主体：人物/物体/动作（精准描述）
            2. 画面风格：艺术风格、色彩基调、色调数值（如#FF6B6B）
            3. 光影：光影类型（伦勃朗光/柔光/硬光）、光源方向、明暗对比
            4. 构图：构图规则、主体占比、背景层级
            5. 细节：纹理材质、装饰元素、环境细节
            输出简洁明了，每条用分号分隔
            """
            analysis = call_moda_api(frame_prompt, frame_base64)
            frame_analyses.append({"frame_idx": i, "analysis": analysis})
            progress_bar.progress((i + 1) / len(frames))
    
    return frame_analyses

def detect_video_cuts(frames, threshold=30):
    """检测视频分镜头（基于帧间差异）"""
    cuts = [0]  # 初始镜头从第0帧开始
    gray_frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY) for frame in frames]
    prev_frame = gray_frames[0]
    
    with st.spinner("检测视频分镜头..."):
        for i in range(1, len(gray_frames)):
            # 计算帧间差异（绝对差值均值）
            diff = cv2.absdiff(prev_frame, gray_frames[i])
            mean_diff = np.mean(diff)
            
            if mean_diff > threshold:
                cuts.append(i)
                prev_frame = gray_frames[i]
    
    # 确保最后一帧是结尾
    if cuts[-1] != len(frames) - 1:
        cuts.append(len(frames) - 1)
    
    # 生成镜头信息
    shots = []
    for i in range(len(cuts) - 1):
        start_idx = cuts[i]
        end_idx = cuts[i + 1]
        shots.append({
            "shot_id": i + 1,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "duration_frames": end_idx - start_idx + 1
        })
    
    st.success(f"检测到{len(shots)}个分镜头！")
    return shots

def analyze_camera_movement(frames, fps):
    """分析运镜手法（基于帧间特征点匹配）"""
    if len(frames) < 2:
        return "运镜分析：视频过短，无法识别运镜手法"
    
    # 提取特征点（SIFT算法）
    sift = cv2.SIFT_create()
    prev_frame = np.array(frames[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)
    
    movements = []
    frame_interval = max(1, len(frames) // 5)  # 抽样5个片段分析
    
    with st.spinner("分析运镜手法..."):
        for i in range(1, len(frames), frame_interval):
            curr_frame = np.array(frames[i])
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)
            
            if prev_des is not None and curr_des is not None:
                # 特征点匹配
                matcher = cv2.FlannBasedMatcher()
                matches = matcher.knnMatch(prev_des, curr_des, k=2)
                
                # 筛选优质匹配
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) > 10:
                    # 计算位移向量
                    prev_points = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    curr_points = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # 计算单应性矩阵，提取平移/旋转/缩放
                    H, _ = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # 提取平移量
                        tx = H[0, 2]
                        ty = H[1, 2]
                        # 提取缩放因子
                        scale_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
                        scale_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
                        
                        # 判断运镜类型
                        if abs(tx) > 20 or abs(ty) > 20:
                            if abs(tx) > abs(ty):
                                movements.append("横向平移（左/右移）")
                            else:
                                movements.append("纵向平移（上/下移）")
                        if scale_x > 1.1 or scale_y > 1.1:
                            movements.append("推镜（放大）")
                        elif scale_x < 0.9 or scale_y < 0.9:
                            movements.append("拉镜（缩小）")
    
    # 去重并生成结果
    unique_movements = list(set(movements))
    if unique_movements:
        return f"运镜手法：{', '.join(unique_movements)}；运镜节奏：中等（基于{fps}FPS分析）"
    else:
        return f"运镜手法：固定镜头（无明显位移/缩放）；运镜节奏：平稳（基于{fps}FPS分析）"

# ---------------------- 页面布局 ----------------------
st.set_page_config(
    page_title="ComfyUI 高级提示词反推工具（图片+视频）",
    page_icon="🖼️",
    layout="wide"
)

# 页面标题
st.markdown("<h1 style='text-align: center; color: #8b5cf6; margin: 20px 0;'>🖼️ 高级提示词反推工具（图片+视频）</h1>", unsafe_allow_html=True)

# 主布局：左右分栏（图片功能+视频功能）
col1, col2 = st.columns(2, gap="20px")

# ---------------------- 左侧：图片分析功能 ----------------------
with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📷 图片超细化分析</div>', unsafe_allow_html=True)
    
    # 图片上传
    uploaded_img = st.file_uploader(
        "上传图片（JPG/PNG/WebP）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )
    
    # 图片预览
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="图片预览", use_column_width=True, clamp=True)
    
    # 生成按钮
    img_generate_btn = st.button("🚀 生成细化提示词", type="primary", key="img_btn")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- 右侧：视频分析功能 ----------------------
with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎬 视频全维度分析</div>', unsafe_allow_html=True)
    
    # 视频上传
    uploaded_video = st.file_uploader(
        "上传视频（MP4/AVI/MKV）",
        type=["mp4", "avi", "mkv"],
        accept_multiple_files=False
    )
    
    # 视频信息显示
    if uploaded_video:
        st.markdown(f"📊 视频信息：{uploaded_video.name}（大小：{round(uploaded_video.size/1024/1024, 2)}MB）")
    
    # 生成按钮
    video_generate_btn = st.button("🎯 开始视频分析", type="primary", key="video_btn")
    st.markdown('</div>', unsafe_allow_html=True)

# 结果显示区：上下分栏（图片结果+视频结果）
st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns(2, gap="20px")

# ---------------------- 左下：图片分析结果 ----------------------
with col3:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 图片细化提示词</div>', unsafe_allow_html=True)
    
    img_result = st.text_area(
        "适配ComfyUI的超细化提示词（可直接复制）",
        height=300,
        disabled=True,
        key="img_result"
    )
    
    # 图片分析逻辑触发
    if img_generate_btn and uploaded_img:
        with st.spinner("图片超细化分析中...（约5-10秒）"):
            img_base64 = image_to_base64(img)
            # 超细化提示词模板
            img_prompt = """
            作为专业AI绘画提示词工程师，超细化分析这张图片，输出适配ComfyUI的提示词，包含以下所有维度，用逗号分隔，关键词精准：
            1. 主体：人物/物体/动作（精准描述，如"1girl, solo, standing, smiling"）
            2. 风格：艺术风格（如oil painting/anime/3D render）、细节风格（如Ghibli style/photorealistic）
            3. 色彩：主色调+辅助色、色彩数值（如#FF6B6B/#3498DB）、色彩氛围（暖色调/冷色调/高饱和）
            4. 光影：光影类型（伦勃朗光/柔光/硬光/逆光）、光源方向（左上方/正上方）、明暗对比（高/中/低）
            5. 材质纹理：物体表面材质（布料/金属/玻璃/木质）、纹理细节（磨砂/光滑/颗粒感/蕾丝花纹）
            6. 构图：构图规则（三分法/对称构图/引导线构图）、视角（平视/俯视/特写/全景）、主体占比（70%/50%）
            7. 背景：背景场景（森林/城市/室内）、背景层级（近景/中景/远景）、背景细节（光斑/雾气/灰尘）
            8. 质量参数：分辨率（8k/4k）、细节等级（ultra detailed）、质感（sharp focus/soft focus）
            9. 额外元素：装饰元素（铆钉/珍珠/花纹）、环境特效（rain/snow/glow）
            要求：关键词无重复，逻辑清晰，可直接复制到ComfyUI的Text Prompt节点使用
            """
            result = call_moda_api(img_prompt, img_base64)
            # 更新结果框（启用并显示）
            st.text_area(
                "✅ 图片提示词生成成功",
                value=result,
                height=300,
                key="img_result_active"
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- 右下：视频分析结果 ----------------------
with col4:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 视频全维度分析结果</div>', unsafe_allow_html=True)
    
    video_result = st.text_area(
        "视频主体、画面、运镜、分镜头分析（可直接用于视频生成提示词）",
        height=300,
        disabled=True,
        key="video_result"
    )
    
    # 视频分析逻辑触发
    if video_generate_btn and uploaded_video:
        # 保存上传的视频到临时文件
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        # 步骤1：提取关键帧
        frames, fps = extract_video_frames("temp_video.mp4", frame_interval=30)
        
        if len(frames) == 0:
            st.error("视频帧提取失败，请更换视频文件！")
        else:
            # 步骤2：分析关键帧内容
            frame_analyses = analyze_video_frames(frames)
            
            # 步骤3：检测分镜头
            shots = detect_video_cuts(frames)
            
            # 步骤4：分析运镜手法
            camera_movement = analyze_camera_movement(frames, fps)
            
            # 步骤5：汇总主体（基于帧分析一致性）
            with st.spinner("汇总视频主体信息..."):
                subject_prompt = f"""
                以下是视频{len(frames)}个关键帧的分析结果，请提取视频的核心主体（人物/物体），要求：
                1. 主体明确（如"一个穿着红色连衣裙的女孩"）
                2. 排除临时出现的次要元素
                3. 描述简洁精准
                帧分析结果：{[item['analysis'] for item in frame_analyses]}
                """
                main_subject = call_moda_api(subject_prompt)
            
            # 步骤6：汇总画面风格
            with st.spinner("汇总画面风格..."):
                style_prompt = f"""
                以下是视频关键帧的分析结果，请汇总视频的整体画面风格，包括：
                1. 艺术风格（如电影级/动漫/纪实）
                2. 色彩基调（如冷色调/暖色调/高饱和）
                3. 整体质感（如细腻/粗糙/复古）
                帧分析结果：{[item['analysis'] for item in frame_analyses]}
                """
                overall_style = call_moda_api(style_prompt)
            
            # 生成最终结果
            final_result = f"""
            🎯 视频核心主体：{main_subject}
            🎨 整体画面风格：{overall_style}
            🎥 运镜手法分析：{camera_movement}
            🎬 分镜头详情：
            共检测到{len(shots)}个分镜头：
            {chr(10).join([f"  镜头{shot['shot_id']}：第{shot['start_frame']}-{shot['end_frame']}帧（时长{shot['duration_frames']}帧，约{round(shot['duration_frames']/fps, 2)}秒）" for shot in shots])}
            📝 视频生成提示词（适配ComfyUI Video节点）：
            {main_subject}, {overall_style}, {camera_movement.split('：')[1]}, {len(shots)} shots, video resolution 1080p, 30fps, ultra detailed frames, smooth camera movement, professional cinematography
            """
            
            # 更新结果框
            st.text_area(
                "✅ 视频分析完成",
                value=final_result,
                height=300,
                key="video_result_active"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 底部说明
st.markdown("""
    <div style="text-align: center; margin: 30px 0; color: #888; font-size: 0.9rem;">
        ---
        🔧 工具说明：图片分析支持超细化关键词生成，视频分析支持最大10分钟视频（建议≤500MB）
        每日免费额度2000次，适配ComfyUI文生图/文生视频节点
    </div>
""", unsafe_allow_html=True)