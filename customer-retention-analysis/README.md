# Customer Segmentation & Retention Analysis

**Author:** Harpinder Singh Chhabra  
**Email:** aekas142@gmail.com  
**GitHub:** [@HarpinderC](https://github.com/HarpinderC)

---

## Status: 🚧 Model Comparison Complete

## Completed
- ✅ Full data preprocessing + feature engineering (30 features)
- ✅ Probabilistic CLV modeling (0.699 correlation)
- ✅ 6-model benchmark comparison
- ✅ ROC curves and model comparison visualizations
- ✅ Honest assessment: sample size limits accuracy

## Model Comparison
| Model | AUC | Notes |
|-------|-----|-------|
| Random Forest | 0.64 | Best performer |
| Gradient Boosting | ~0.62 | - |
| XGBoost | ~0.61 | - |
| LightGBM | ~0.60 | - |
| CatBoost | ~0.61 | - |
| Logistic Regression | 0.58 | Baseline |

**Note:** Small dataset (815 customers) limits supervised model performance. Model used as diagnostic tool.

## Next Steps
- Enhancement experiments (SMOTE, feature interactions)
- SHAP explainability analysis
- Business strategy development
