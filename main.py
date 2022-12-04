from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pandas as pd
import pickle

app = FastAPI()

columns_for_train = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

@app.on_event("startup")
def on_start():
    print('Work is on!')
    global model
    try:
        
        with open("model_for_api.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(e)
        exit(1)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

example = Item(name="Opel",
                year=2014,
                selling_price = 100000,
                torque="100 Nm @ 1750rpm",
                km_driven=100,
                fuel="Fuel",
                seller_type="Individual",
                transmission="Manual",
                owner="First Owner",
                mileage=100,
                engine=1000,
                max_power=1343,
                seats=2)

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    obj = pd.DataFrame([x.dict() for x in [item]])
    obj = obj[columns_for_train]
    return float(model.predict(obj))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    res = []
    for el in items:
        res.append(predict_item(el))
    return res

print('===========')
on_start()
print(predict_item(example))
print(predict_items([example, example]))
print('===========')

@app.get('/')
async def root():
    return {"message": "hello world!"}

