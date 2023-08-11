# FROM docker.repos.balad.ir/python:3.10
FROM python:3.10


WORKDIR /code

RUN  pip install -i https://download.pytorch.org/whl/cpu torchvision torch
RUN pip install -U --no-cache-dir gdown --pre

# COPY ./load_model.py /code/
# RUN python /code/load_model.py

COPY ./requirements.txt /code/requirements.txt

# RUN  pip install  --trusted-host https://pypi.mirrors.ustc.edu.cn/ -i https://pypi.mirrors.ustc.edu.cn/simple/ -r /code/requirements.txt
RUN pip install -r requirements.txt

COPY ./fast_api_service.py /code/fast_api_service.py

CMD ["uvicorn", "fast_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
