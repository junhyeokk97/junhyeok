from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return{"message" : "Hello, World"}

@app.get("/hello/")
def read_root():
    return{"message" : "Hello, Hello, Hello"}

@app.get("/hi/")
def read_root():
    return{"message" : "Hi, Hi, Hi"}