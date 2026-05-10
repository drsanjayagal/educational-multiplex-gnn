FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
COPY . .
RUN pip install -e .
ENTRYPOINT ["python"]
CMD ["--help"]
