
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from vision_server.yolo_script import YoloParser

class AnnotateRequest(BaseModel):
    image_b64: str
    box_threshold: float = 0.05
    iou_threshold: float = 0.1

class AnnotateResponse(BaseModel):
    annotated_b64: str
    button_coordinates: dict


parser_instance = None
count = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global parser_instance
    print("Loading YOLO model into memory")
    parser_instance = YoloParser(model_path="/data1/visionAgent/OmniParser/weights/icon_detect/model.pt")
    yield
    print("Shutting down server")

app = FastAPI(title="YOLO UI Parser API", lifespan=lifespan)

@app.post("/annotate", response_model=AnnotateResponse)
async def api_annotate(request: AnnotateRequest):
    if not parser_instance:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")
    
    try:
        b64_img, coords = await parser_instance.annotate_image(
            image_input=request.image_b64,
            box_threshold=request.box_threshold,
            iou_threshold=request.iou_threshold
        )
        return AnnotateResponse(
            annotated_b64=b64_img,
            button_coordinates=coords
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": parser_instance.device if parser_instance else "loading"}

if __name__ == "__main__":
    uvicorn.run("yolo_server:app", host="127.0.0.1", port=8020, reload=True)
