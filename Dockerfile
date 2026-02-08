# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Hugging Face Spaces default port
EXPOSE 7860

# Command to run the Streamlit app on port 7860
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=7860", "--server.address=0.0.0.0"]