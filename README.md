**README.md**

```
# Predictive Maintenance Project

Короткий опис файлів і папок:

├── **111**  
│   ├─ **kaggle/**  
│   │   ├─ `train_FD00?.txt` — вхідні часові ряди для тренування RUL-моделей  
│   │   ├─ `test_FD00?.txt`  — тестові дані без RUL  
│   │   └─ `RUL_FD00?.txt`   — справжні залишкові ресурси для тесту  
│   └─ **kaggle/**  
│       ├─ **Healthy/** — `.mat`-файли вібрацій нормальних машин  
│       └─ **Faulty/**  — `.mat`-файли вібрацій пошкоджених машин  

├  
│   ├─ `data_prep.py`   — зчитування й об’єднання NASA-файлів + завантаження RUL  
│   ├─ `features.py`    — інженіринг ознак (видалення пласких/корельованих, фільтрація)  
│   ├─ `model_rul.py`   — побудова й калібрування пайплайна для прогнозу RUL  
│   └─ `model_vib.py`   — підготовка RMS/частотних фіч і класифікація вібрацій  


├── **outputs/**  
│   ├─ `train_all_processed.csv`   — зведена таблиця NASA з RUL  
│   ├─ `df_feats.csv`              — CSV з RMS/частотними фічами + мітками  
│   ├─ `rf_rul_pipeline.pkl`       — збережений RandomForestRegressor пайплайн  
│   ├─ `ir_rul_calib.pkl`          — калібратор IsotonicRegression для RUL  

