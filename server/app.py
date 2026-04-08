from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Server is running!"}

# Example endpoint for health check
@app.get("/health")
def health_check():
    return {"status": "ok"}
