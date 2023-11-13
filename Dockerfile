FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

WORKDIR /usr/src/app

# copy the current directory contents into the container at /usr/src/app
COPY src/ .

COPY src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train.py"]

