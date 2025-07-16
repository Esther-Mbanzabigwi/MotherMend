# 🔧 Base image
FROM python:3.10-slim

# 📁 Set working directory
WORKDIR /app

# 🧱 Copy everything into the container
COPY . /app

# 🐍 Install dependencies (use wheels where possible)
RUN pip install --upgrade pip \
 && pip install --prefer-binary -r requirements.txt

# 🔓 Expose port
EXPOSE 10000

# 🚀 Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
