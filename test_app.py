import requests
import time


def test_latency(image_path: str):
    start_time = time.time()
    with open(image_path, 'rb') as file:
        files = {'image': file}
        response = requests.post('https://traveling-guide.darkube.app:8000/predict', files=files)

    assert response.status_code == 200
    print('response: ', response.text)
    end_time = time.time()
    latency = end_time - start_time
    print(f'Latency: {latency} seconds')


if __name__ == '__main__':
    # for i in range(5):
    test_latency(f'./images/eiffel.jpg')



# from mlflow import log_metric, log_param, log_artifact, set_tracking_uri

# if __name__ == "__main__":
#     # remote_server_uri = 'localhost:5000' # this value has been replaced
#     # set_tracking_uri(remote_server_uri)
#     # Log a parameter (key-value pair)
#     log_param("param1", 5)

#     # Log a metric; metrics can be updated throughout the run
#     log_metric("foo", 1)
#     log_metric("foo", 2)
#     log_metric("foo", 3)

    # Log an artifact (output file)
    # with open("output.txt", "w") as f:
    #     f.write("Hello world!")
    # log_artifact("output.txt")

