import requests
import os

url = "http://localhost:5000/predict"
images_directory = os.path.join(os.getcwd(), "dataset\\test\\images")
payload = {
    "images_directory": images_directory
}

response = requests.post(url, data=payload)

if response.status_code == 200:
    predictions = response.json()
    
else:
    print("Error: ", response.status_code)
