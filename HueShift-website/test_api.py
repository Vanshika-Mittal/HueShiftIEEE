import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

KAGGLE_ENDPOINT = os.getenv("KAGGLE_ENDPOINT")


def test_api():
    # health endpoint
    print("Testing health endpoint...")
    health_response = requests.get(f"{KAGGLE_ENDPOINT}/health")
    print(f"Health Status: {health_response.status_code}")
    print(f"Health Response: {health_response.text}\n")

    # process endpoint
    print("Testing process endpoint...")
    image_path = "results/teaser.jpg"
    try:
        with open(image_path, "rb") as f:
            files = {"image": ("teaser.jpg", f, "image/jpeg")}
            process_response = requests.post(
                f"{KAGGLE_ENDPOINT}/api/process", files=files
            )
            print(f"Process Status: {process_response.status_code}")
            print(f"Process Headers: {process_response.headers}")
            print(f"Process Response: {process_response.text}")

            if process_response.status_code == 200:
                result = process_response.json()
                if "id" in result:
                    print(f"\nTesting results endpoint for ID: {result['id']}")
                    result_response = requests.get(
                        f"{KAGGLE_ENDPOINT}/api/results/{result['id']}"
                    )
                    print(f"Result Status: {result_response.status_code}")
                    print(f"Result Headers: {result_response.headers}")
                    if result_response.status_code == 200:
                        print("Successfully retrieved result image")
                    else:
                        print(f"Failed to get result: {result_response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_api()
