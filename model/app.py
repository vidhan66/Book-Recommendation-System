from fastapi import FastAPI
import uvicorn
from recommend import api_recommend_cbf, api_recommend_cf, api_recommend_hybrid

app = FastAPI(
    title="Book Recommendation API",
    description="Endpoints for CBF, CF, and Hybrid book recommendations",
    version="1.0"
)

# Mount the endpoints defined in recommend.py
app.add_api_route("/recommend/cbf", api_recommend_cbf, methods=["GET"])
app.add_api_route("/recommend/cf", api_recommend_cf, methods=["GET"])
app.add_api_route("/recommend/hybrid", api_recommend_hybrid, methods=["GET"])

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
