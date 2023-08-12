# FROM docker.repos.balad.ir/python:3.10
FROM python:3.10


WORKDIR /code

RUN  pip install -i https://download.pytorch.org/whl/cpu torchvision torch
RUN pip install -U --no-cache-dir gdown --pre


COPY ./requirements.txt /code/requirements.txt

RUN pip install -r requirements.txt

COPY ./app /code/app 
CMD ["uvicorn", "app.fast_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
