# Use the official Streamlit base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . /app

# Download the model file
RUN wget -O mistral-7b-instruct-v0.1.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf?download=true


RUN pip install -r requirements.txt  # Add any additional requirements if needed

# Expose the port where Streamlit will run
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]  # Replace "app.py" with the actual name of your Python script
