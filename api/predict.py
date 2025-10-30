from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ml_core import load_and_prepare_data, train_model, predict_price

app = FastAPI()

# Model init on cold start (Vercel runs hot/cold)
X_train, _, y_train, _, scaler, feature_names = load_and_prepare_data()
model = train_model(X_train, y_train, model_name='Random Forest')

@app.post("/api/predict")
async def predict(request: Request):
    body = await request.json()
    features = body.get('features')
    if not (features and scaler and model):
        return JSONResponse({'error': 'Model or features missing.'}, status_code=400)
    pred = predict_price(model, scaler, features)
    return {'prediction': pred}