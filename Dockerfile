FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /usr/src/app

# copy the current directory contents into the container at /usr/src/app
COPY src/ .

COPY src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "training/train.py"]

