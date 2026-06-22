# California House Price Prediction API

A production-ready FastAPI application powered by a Random Forest Machine Learning model that provides instant house price estimates for California properties.

## 📖 Overview

This project delivers a high-performance API service for California house price prediction. It solves the real-world problem of inconsistent and slow property valuation by providing data-driven, instant price estimates.

**Key Benefits:**
- Real estate agents can provide instant, accurate price estimates
- Customers receive consistent, data-backed valuations
- Scalable system that serves thousands of requests
- Built using industry-standard tools and best practices

## 🎯 Problem Statement

Real estate companies face significant challenges with property price estimation:
- **Time-consuming process**: Agents rely on manual calculations and experience
- **Inconsistent results**: Different agents provide varying estimates for the same property
- **Scalability issues**: Manual processes don't scale with growing customer base
- **Slow decision-making**: Customers wait hours or days for price estimates

Traditional approaches fail to meet the demands of modern real estate operations.

## 🚀 Solution

This project implements an end-to-end Machine Learning solution:
- **Data Analysis**: Explores and understands the California housing dataset
- **Model Training**: Trains a Random Forest Regressor on historical data
- **Model Deployment**: Exposes predictions via a FastAPI REST API
- **Scalable Architecture**: Ready for cloud deployment and high traffic

**Business Value**:
- 100x faster price estimation (instant vs hours/days)
- 100% consistent results across all agents
- Unlimited scalability for growing businesses
- Improved customer trust and satisfaction

## ✨ Features

- Instant house price predictions via REST API
- Input validation with Pydantic schemas
- Health check endpoint for monitoring
- Model performance metrics exposed
- Production-ready FastAPI application
- Scikit-learn Random Forest model
- Model serialization with Joblib

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast (high-performance) web framework
- **Pydantic** - Data validation and settings management

### Machine Learning
- **Scikit-learn** - Machine learning library
- **Random Forest Regressor** - Prediction model
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Deployment
- **Render / Railway / AWS / Azure** - Cloud deployment options
- **Uvicorn** - ASGI server

### Tools & Libraries
- **Joblib** - Model serialization
- **Git** - Version control

## 📂 Project Structure

```
d:\FAST-API-Ecommerces\
├── explore.py          # Data exploration and analysis
├── train.py            # Model training pipeline
├── main.py             # FastAPI application
├── house_model.joblib  # Trained ML model (generated)
├── house_feature_names.joblib  # Feature names (generated)
├── house_model_metadata.joblib # Model metadata (generated)
└── README.md           # Project documentation
```

**Key Files:**
- `explore.py`: Loads and analyzes the California housing dataset
- `train.py`: Trains the Random Forest model and saves artifacts
- `main.py`: FastAPI application with prediction endpoints

## ⚙️ Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd FAST-API-Ecommerces
   ```

2. **Create a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Train the model** (generates required .joblib files)
   ```powershell
   python train.py
   ```

## ▶️ Usage

1. **Start the FastAPI server**
   ```powershell
   uvicorn main:app --reload
   ```

2. **Access the API documentation**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

| Method | Endpoint | Description | Request Type |
|--------|----------|-------------|--------------|
| `GET` | `/` | Home endpoint with API status | None |
| `GET` | `/health` | Health check with model metadata | None |
| `POST` | `/predict` | Predict house price | JSON |

## 📊 Example Request

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

## 📊 Example Response

```json
{
  "predicted_price": 452600.0
}
```

## 🧪 Testing

1. **Test via Swagger UI**: Navigate to `http://localhost:8000/docs` and use the interactive interface
2. **Test with curl**:
   ```powershell
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
     "MedInc": 8.3252,
     "HouseAge": 41.0,
     "AveRooms": 6.9841,
     "AveBedrms": 1.0238,
     "Population": 322.0,
     "AveOccup": 2.5556,
     "Latitude": 37.88,
     "Longitude": -122.23
   }'
   ```

## 🔮 Future Improvements

- **Model Enhancements**: Hyperparameter tuning, feature engineering, try different algorithms (XGBoost, LightGBM)
- **Data Pipeline**: Add data versioning with DVC, automate retraining
- **Monitoring**: Add Prometheus metrics, model drift detection
- **Database**: Integrate PostgreSQL to store prediction history
- **Authentication**: Add OAuth2/JWT for secure API access
- **Rate Limiting**: Protect API from abuse
- **Dockerization**: Containerize application for consistent deployment
- **CI/CD**: Automate testing and deployment with GitHub Actions
- **Frontend**: Build a simple web interface for agents and customers

## 👨‍💻 Author

Project Developer

## ⭐ Acknowledgements

- Scikit-learn team for the California Housing dataset
- FastAPI community for excellent documentation
