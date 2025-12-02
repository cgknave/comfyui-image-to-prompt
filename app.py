import streamlit as st
import requests
from PIL import Image
import io

# 1. 替换为你的魔搭API密钥
API_KEY = "ms-9f99616d-d3cf-4783-922a-1ed9599fec3a"  # 直接使用生成的完整Token（无需去掉前缀）
# 2. 选择模型ID
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
# 3. 魔搭API固定地址
API_URL = f"https://api-inference.modelscope.cn/v1/chat/completions"

# 请求头配置
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

st.set_page_config(page_title="ComfyUI云端提示词反推（魔搭版）")
st.title("🖼️ ComfyUI 魔搭免费API提示词反推")

# 上传图片
uploaded_file = st.file_uploader("选择图片（JPG/PNG）", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传预览", width=300)

    # 生成提示词按钮
    if st.button("🚀 生成ComfyUI提示词"):
        with st.spinner("魔搭云端处理中..."):
            # 转换图片为Base64编码（魔搭API支持的格式）
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_base64 = img_byte_arr.getvalue().hex()  # 转换为十六进制编码

            # 构造请求（Qwen2.5-VL的提示词模板）
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "详细描述这张图片的主体、风格、色彩、纹理、场景，输出适配ComfyUI的文生图提示词，关键词清晰、分点但用逗号连接"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"hex://{img_base64}"  # 魔搭支持的图片编码格式
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 200
            }

            try:
                # 调用魔搭API
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                # 提取生成的提示词
                prompt = response.json()["choices"][0]["message"]["content"]
                st.subheader("生成的ComfyUI提示词")
                st.text_area("", value=prompt, height=200)
                st.success("提示词生成成功！")
            except Exception as e:
                st.error(f"生成失败：{str(e)}")