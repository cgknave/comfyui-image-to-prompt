import streamlit as st
import requests
from PIL import Image
import io
import base64  # 新增：用于图片Base64编码（魔搭API要求）

# ---------------------- 仅需修改这1处！----------------------
API_KEY = "ms-9f99616d-d3cf-4783-922a-1ed9599fec3a"  # 替换成你步骤1获取的魔搭Token（以ms-开头）
# -------------------------------------------------------------

# 魔搭API固定配置（无需修改）
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"  # 免费可用的多模态模型
API_URL = "https://api-inference.modelscope.cn/v1/chat/completions"  # 魔搭API固定地址

# 请求头（无需修改）
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 页面配置（无需修改）
st.set_page_config(
    page_title="ComfyUI 魔搭免费提示词反推",
    page_icon="🖼️",
    layout="wide"
)

# 页面标题和说明（无需修改）
st.title("🖼️ ComfyUI 魔搭免费提示词反推工具")
st.markdown("""
    上传图片 → 云端生成适配ComfyUI的文生图提示词（免费、稳定、中文友好）
    - 每日免费调用2000次，满足个人创作需求
    - 生成的提示词可直接复制到ComfyUI的「Text Prompt」节点
""")

# 分栏布局（左侧上传图片，右侧显示结果）
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 上传图片")
    # 图片上传组件（支持JPG/PNG，最大5MB）
    uploaded_file = st.file_uploader(
        "支持格式：JPG/PNG/WebP（建议图片清晰、主体明确）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )
    
    # 预览上传的图片
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传图片预览", use_column_width=True)

with col2:
    st.subheader("📝 生成的ComfyUI提示词")
    # 提示词显示框（默认禁用，生成后启用）
    result_box = st.text_area(
        "点击下方按钮生成，生成后可直接复制",
        height=250,
        disabled=True
    )

# 生成提示词按钮（核心逻辑，无需修改）
if uploaded_file and st.button("🚀 开始生成提示词", type="primary"):
    with st.spinner("魔搭云端处理中...（约3-10秒）"):
        try:
            # 1. 图片转Base64编码（魔搭API要求的格式，必须按这个来）
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")  # 统一转换为JPEG格式
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{img_base64}"  # 魔搭支持的图片URL格式

            # 2. 构造符合魔搭API的请求参数（固定格式，无需修改）
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "详细描述这张图片：主体（人物/物体/动作）、场景背景、艺术风格（如油画/3D渲染/二次元）、色彩氛围、光影效果、构图视角、纹理质感，输出适配ComfyUI的文生图提示词，关键词用逗号分隔，清晰易懂，可直接复制使用"
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                "max_tokens": 250,  # 最大提示词长度（足够详细）
                "temperature": 0.7,  # 生成随机性（越低越精准）
                "top_p": 0.9
            }

            # 3. 调用魔搭API
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60  # 超时时间（避免网络慢导致失败）
            )
            response.raise_for_status()  # 检查请求是否成功

            # 4. 提取并显示提示词
            result = response.json()
            comfyui_prompt = result["choices"][0]["message"]["content"].strip()
            
            # 更新结果框（启用并显示提示词）
            result_box = st.text_area(
                "✅ 提示词生成成功（可直接复制到ComfyUI）",
                value=comfyui_prompt,
                height=250
            )

        except Exception as e:
            # 错误提示（帮助排查问题）
            st.error(f"生成失败！原因：{str(e)}")
            st.markdown("""
                排查建议：
                1. 检查API_KEY是否正确（必须是魔搭的ms-开头Token）；
                2. 图片是否超过5MB（建议压缩后重试）；
                3. 网络是否正常（魔搭是国内API，无需科学上网）。
            """)

# 底部使用说明（无需修改）
st.markdown("""
    ---
    ### 使用技巧：
    1. 图片越清晰、主体越突出，提示词生成越精准；
    2. 若提示词过长，可手动删减重复关键词；
    3. 搭配ComfyUI的「Negative Prompt」（如low quality, blurry）优化生成效果；
    4. 每日免费额度2000次，足够个人使用。
""")