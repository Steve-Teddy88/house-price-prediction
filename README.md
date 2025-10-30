# House Price Prediction Machine Learning Project

## 🏠 Overview

This is a comprehensive machine learning project that demonstrates a complete workflow for predicting house prices using the California Housing dataset. The project showcases best practices in data science, including data preprocessing, model training, evaluation, and visualization.

## 🚀 Features

- **Automatic Data Loading**: Uses the California Housing dataset from scikit-learn
- **Data Preprocessing**: Handles missing values, feature scaling, and data splitting
- **Multiple Models**: Compares Linear Regression, Random Forest, and Gradient Boosting
- **Comprehensive Evaluation**: Uses MAE, RMSE, R², and cross-validation metrics
- **Rich Visualizations**: Creates correlation heatmaps, feature importance plots, and performance comparisons
- **Professional Documentation**: Well-documented functions with clear explanations

## 📊 What You'll Learn

1. **Data Science Workflow**: Complete ML pipeline from data loading to model deployment
2. **Feature Engineering**: Handling missing values and feature scaling
3. **Model Comparison**: Evaluating multiple algorithms with different metrics
4. **Visualization**: Creating informative plots for data analysis
5. **Best Practices**: Professional code structure with proper documentation

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Project**:
   ```bash
   python model_trainer.py
   ```

## 📈 Project Structure

The project is organized into modular files:

- `model_trainer.py`: Main script to orchestrate the ML workflow
- `ml_core.py`: Core machine learning functions
- `visualization.py`: Functions for creating visualizations
- `api/predict.py`: FastAPI for model deployment

## 🎯 Key Components

### Data Preprocessing
- Missing value handling
- Feature scaling with StandardScaler
- Train-test split (80-20)

### Models Used
1. **Linear Regression**: Baseline model for comparison
2. **Random Forest**: Ensemble method with feature importance
3. **Gradient Boosting**: Advanced ensemble method

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Standard deviation of prediction errors
- **R² Score**: Proportion of variance explained by the model
- **Cross-validation**: Model stability assessment

### Visualizations
- Feature correlation heatmap
- House price distribution
- Feature importance analysis
- Model performance comparison
- Actual vs Predicted plots
- Residual analysis

## 📊 Expected Output

When you run the project, you'll see:

1. **Dataset Information**: Sample count, features, and statistics
2. **Preprocessing Status**: Missing value checks and scaling confirmation
3. **Model Training Progress**: Real-time training updates
4. **Performance Metrics**: Detailed evaluation results for each model
5. **Visualizations**: Interactive plots and saved image file
6. **Summary Report**: Best model identification and recommendations

## 🎓 Learning Objectives

This project teaches you:

- **Data Science Fundamentals**: Complete ML workflow
- **Python Best Practices**: Clean, documented, modular code
- **Model Evaluation**: Understanding different performance metrics
- **Visualization Skills**: Creating informative data plots
- **Real-world Application**: Practical house price prediction

## 🔧 Customization

You can easily modify the project:

- **Add New Models**: Include additional algorithms in the `train_model()` function in `ml_core.py`
- **Change Dataset**: Replace the California Housing dataset with your own data in `ml_core.py`
- **Modify Visualizations**: Customize plots in the `visualize_results()` function in `visualization.py`
- **Add Features**: Implement feature engineering techniques in `ml_core.py`

## 📝 Requirements

- Python 3.7+
- pandas>=1.5.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scikit-learn>=1.1.0
- fastapi>=0.115.0
- uvicorn>=0.29.0
- joblib>=1.4.2
- pydantic>=2.8.2

## 🎉 Getting Started

1. Clone or download this project
2. Install the requirements: `pip install -r requirements.txt`
3. Run the project: `python model_trainer.py`
4. Explore the generated visualizations and analysis

## 📞 Support

This project is designed to be educational and self-contained. All code includes detailed comments and docstrings to help you understand each step of the machine learning process.

---

**Happy Learning! 🚀** 