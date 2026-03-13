import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from rembg import remove

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        input_bytes = await file.read()
        output_bytes = remove(input_bytes)

        return StreamingResponse(
            BytesIO(output_bytes),
            media_type="image/png"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
