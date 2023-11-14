FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /opt/ml/code

COPY src/training/ .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train.py"]
