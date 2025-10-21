from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return{"message" : "Hello, Fast API"}

@app.get("/item/{item_id}")
def read_item(item_id):
    return{"item_id":item_id}
    
@app.get("/items/")
def read_items(skip=0, limit=10): # default 값을 지정
    return {'skip': skip, 'limit': limit}