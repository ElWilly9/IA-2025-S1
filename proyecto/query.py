import requests

url = "http://localhost:8000/consulta"
data = {"pregunta": "¿Cuál es la presión correcta de las llantas de la Bajaj CT100 KS?"}

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
