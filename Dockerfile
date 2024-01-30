# Use the official Streamlit base image
FROM python:3.9-slim
# Install build tools and dependencies
WORKDIR /app

# Copy the local code to the container
COPY . /app
RUN apt-get update && \
    apt-get install -y build-essential cmake wget



RUN pip install -r requirements.txt  # Add any additional requirements if needed

# Expose the port where Streamlit will run
EXPOSE 8501

# Command to run the application
CMD ["sh", "entry_point.sh"]
