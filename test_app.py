import time

import requests


def test_latency(image_path: str):
    start_time = time.time()
    with open(image_path, "rb") as file:
        files = {"image": file}
        response = requests.post(
            "https://shadow-traveling-guide.darkube.app/predict", files=files
        )

    assert response.status_code == 200
    print("response: ", response.text)
    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")



# if __name__ == "__main__":
#     test_latency("./images/eiffel.jpg")


def send_to_shadow_server(image):
    response = requests.post(
        "https://shadow-traveling-guide.darkube.app/predict", files={"image": image}
    )
    return response

with open("./images/eiffel.jpg", "rb") as image:
    send_to_shadow_server(image)
