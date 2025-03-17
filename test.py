import requests

url = "http://127.0.0.1:5000/predict"

news = {"text": "Big News: KKR's speed sensation ruled out of IPL 2025, defending champions announce 27-year-old star as his replacement"}

response = requests.post(url, json=news)

print(response.json())
