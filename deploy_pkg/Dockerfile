FROM python:3.11-slim

WORKDIR /app

# pip mirror for faster downloads in China
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install only Python deps
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy project files
COPY . .

# Flask serves both API and static index.html
EXPOSE 5001

CMD ["python", "backend/server.py"]
