# 1. Use Python 3.12 as the base
FROM python:3.12

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . .

# 5. Expose the port Chainlit runs on (7860 is standard for Hugging Face)
EXPOSE 7860

# 6. Run the app
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]