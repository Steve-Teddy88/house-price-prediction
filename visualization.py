
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def visualize_results(X, y, models, X_test, y_test, feature_names):
    """
    Create comprehensive visualizations for the analysis.
    
    Args:
        X (pd.DataFrame): Original feature matrix
        y (pd.Series): Target variable
        models (dict): Trained models
        X_test (np.array): Test features
        y_test (pd.Series): Test target
        feature_names (list): Names of features
    """
    print("\n📈 Creating Visualizations...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Correlation heatmap
    plt.subplot(3, 3, 1)
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 2. Target distribution
    plt.subplot(3, 3, 2)
    plt.hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('House Price (in $100k)')
    plt.ylabel('Frequency')
    plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Feature importance (using Random Forest)
    plt.subplot(3, 3, 3)
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    
    # 4. Model performance comparison
    plt.subplot(3, 3, 4)
    model_names = list(models.keys())
    r2_scores = []
    
    for name in model_names:
        y_pred = models[name].predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
    
    bars = plt.bar(model_names, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Actual vs Predicted plots for each model
    for i, (name, model) in enumerate(models.items()):
        plt.subplot(3, 3, 5 + i)
        y_pred = model.predict(X_test)
        
        plt.scatter(y_test, y_pred, alpha=0.6, color='#FF6B6B')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{name}: Actual vs Predicted', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. Residual plots
    plt.subplot(3, 3, 8)
    best_model_name = max(models.keys(), key=lambda x: r2_score(y_test, models[x].predict(X_test)))
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    residuals = y_test - y_pred_best
    
    plt.scatter(y_pred_best, residuals, alpha=0.6, color='#4ECDC4')
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot ({best_model_name})', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 7. Error distribution
    plt.subplot(3, 3, 9)
    plt.hist(residuals, bins=30, alpha=0.7, color='#45B7D1', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('house_price_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'house_price_prediction_analysis.png'")
