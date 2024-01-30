export MODEL_FILE=mistral-7b-instruct-v0.1.Q4_K_M.gguf
if [ ! -f "$MODEL_FILE" ]; then \
    wget -O "$MODEL_FILE" https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/$MODEL_FILE?download=true; \
fi
streamlit run app.py