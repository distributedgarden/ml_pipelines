FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /opt/ml/code

RUN mkdir -p /opt/ml/input/data/training
RUN mkdir -p /opt/ml/model

COPY src/training/train.py ./
COPY src/training/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]

