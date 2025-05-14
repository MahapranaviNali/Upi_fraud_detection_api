import requests

payload = {
    "sender_upi": "user1@upi",
    "receiver_upi": "merchant@upi",
    "amount": 250.0
}

res = requests.post("http://127.0.0.1:8000/predict", json=payload)

print(res.json())
