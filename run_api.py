import uvicorn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.app import app

if __name__ == "__main__":
    print("Starting AG News Classifier API...")
    print("Access the web interface at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )