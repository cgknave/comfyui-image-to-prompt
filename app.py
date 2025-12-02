import streamlit as st
import requests
from PIL import Image
import io

# ---------------------- 仅需修改这1处！----------------------
API_KEY ="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"  # 替换成第一步的hf_xxxxxx
# -------------------------------------------------------------

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip2-opt-2.7b"
headers = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(page_title="ComfyUI云端提示词反推", page_icon="🖼️")
st.title("🖼️ ComfyUI 云端提示词反推工具")
st.markdown("上传图片，云端生成提示词，直接复制到ComfyUI！")

# 上传图片
uploaded_file = st.file_uploader("选择图片（JPG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 预览图片
    image = Image.open(uploaded_file)
    st.image(image, caption="上传预览", width=300)
    
    # 生成提示词按钮
    if st.button("🚀 生成提示词"):
        with st.spinner("云端处理中..."):
            # 转换图片为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # 调用云端API
            data = {
                "inputs": "详细描述图片：主体（人物/物体/动作）、场景背景、艺术风格、色彩氛围、光影效果、构图视角、纹理质感，用于AI绘画，适配ComfyUI"
            }
            files = {"parameters": ("image.jpg", img_byte_arr, "image/jpeg")}
            
            try:
                response = requests.post(API_URL, headers=headers, data=data, files=files, timeout=30)
                response.raise_for_status()
                prompt = response.json()["generated_text"]
                
                # 显示提示词
                st.subheader("生成的提示词（可直接复制）")
                st.text_area("", value=prompt, height=200)
                st.success("提示词生成成功！")
            except Exception as e:
                st.error(f"生成失败：{str(e)}")