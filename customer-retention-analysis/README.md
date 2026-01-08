# Customer Segmentation & Retention Analysis

**Author:** Harpinder Singh Chhabra  
**Email:** aekas142@gmail.com  
**GitHub:** [@HarpinderC](https://github.com/HarpinderC)

---

## Status: 🚧 Model Optimization Complete

## Completed
- ✅ Full data preprocessing + feature engineering (30 features)
- ✅ Probabilistic CLV modeling (0.699 correlation)
- ✅ 6-model benchmark comparison (Random Forest best: AUC 0.64)
- ✅ Enhancement experiment:
  - SMOTE oversampling (decreased performance - documented transparently)
  - Hyperparameter tuning on Random Forest
  - Feature interaction engineering
- ✅ Best model: Random Forest with class weighting (saved)

## Honest Assessment
The churn classifier achieves 0.64 AUC - below production threshold. This is due to small sample size (815 customers). The model is repositioned as a **diagnostic tool** for behavioral pattern identification rather than production-ready predictor.

## Next Steps
- SHAP explainability analysis
- Business segmentation strategy
- Deployment dashboard
