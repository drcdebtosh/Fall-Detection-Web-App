# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Environment ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- Install Linux Packages ----------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Install Python Dependencies ----------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Copy Project ----------
COPY . .

# ---------- Required Folders ----------
RUN mkdir -p uploads outputs

# ---------- Flask Port ----------
EXPOSE 5050

# ---------- Run ----------
CMD ["python", "app.py"]