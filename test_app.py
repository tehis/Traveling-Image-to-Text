import time

import requests


def test_latency(image_path: str):
    start_time = time.time()
    with open(image_path, "rb") as file:
        files = {"image": file}

        response = requests.post(
            "https://traveling-guide.darkube.app/predict", files=files
        )

    assert response.status_code == 200
    # print("response: ", response.text)
    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")


# if __name__ == "__main__":
#     test_latency("./images/eiffel.jpg")


from multiprocessing.dummy import Pool

import requests

pool = Pool(5)  # Creates a pool with ten threads; more threads = more concurrency.
# "pool" is a module attribute; you can be sure there will only
# be one of them in your application
# as modules are cached after initialization.

if __name__ == "__main__":
    futures = []
    for x in range(10):
        futures.append(pool.apply_async(test_latency, ["./images/eiffel.jpg"]))
    # futures is now a list of 10 futures.
    print("Finished")
    for future in futures:
        print(future.get())  # For each future, wait until the request is
