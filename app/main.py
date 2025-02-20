import uvicorn
from fastapi import FastAPI

from routers import face_detect
try:
    app = FastAPI()
except Exception as e:
    print(e)
    print("Không khởi tạo được ứng dụng")
    exit()
app.include_router(face_detect.router)

if __name__ == '__main__':


    uvicorn.run(app=app, host="127.0.0.1", port= 8000)