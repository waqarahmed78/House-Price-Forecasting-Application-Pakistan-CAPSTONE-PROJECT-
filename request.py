import requests


url = 'http://10.8.24.116:9000/predict_api'
r = requests.post(url,json={'Bathrooms':2, 'Bedrooms':3, 'Rooms':4, 'Area':200 })

print(r.json())
