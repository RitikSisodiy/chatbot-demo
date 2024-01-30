# Use the official Streamlit base image
FROM python:3.9-slim
# Install build tools and dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake wget
# Set the working directory in the container
ENV MODEL_FILE mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Check if the model file already exists before downloading
RUN if [ ! -f "$MODEL_FILE" ]; then \
    wget -O "$MODEL_FILE" https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/$MODEL_FILE?download=true; \
fi
WORKDIR /app

# Copy the local code to the container
COPY . /app


RUN pip install -r requirements.txt  # Add any additional requirements if needed

# Expose the port where Streamlit will run
EXPOSE 8501

# Command to run the application
CMD ["sh", "entry_point.sh"]
