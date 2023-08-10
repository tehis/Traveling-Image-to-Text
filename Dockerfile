# FROM docker.repos.balad.ir/python:3.10
FROM python:3.10


WORKDIR /code

# RUN pip install --trusted-host https://mirrors.aliyun.com -i https://mirrors.aliyun.com/pypi/simple/ pysocks
# RUN  pip install torchvision torch --proxy socks5://127.0.0.1:1080  --index-url https://download.pytorch.org/whl/cpu

# COPY ./load_model.py /code/
# RUN python /code/load_model.py


COPY ./requirements.txt /code/requirements.txt

# RUN  pip install  --trusted-host https://pypi.mirrors.ustc.edu.cn/ -i https://pypi.mirrors.ustc.edu.cn/simple/ -r /code/requirements.txt
RUN pip install -r requirements.txt

COPY ./fast_api_service.py /code/fast_api_service.py

CMD ["uvicorn", "fast_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
# COPY ./run_mlflow_and_fastapi.sh /code/app/run_mlflow_and_fastapi.sh
# RUN chmod +x /code/app/run_mlflow_and_fastapi.sh


# CMD /code/app/run_mlflow_and_fastapi.sh
