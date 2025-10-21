from fastapi import FastAPI



app = FastAPI()

@app.get("/")  # 데코레이터: 함수 위에 작성되어 함수의 동작을 감싸고 확장함
def read_root():
    return {"message": "Hello, FastAPI"}

@app.get("/hi/{item_id}")  # 경로 파라미터 item_id를 받는 엔드포인트
def read_items(skip=0, limit=10):  # 타입 힌트도 추가
    return {'skip': skip, "limit": limit}
  
  

