# Used-Car Price Prediction

Predicts the price of a used car in Israel from its listing, using an ElasticNet regression model.

**Context:** final project, machine learning course · **Team:** Nadav Mashiach, Amit Stein

## Data
1,500 Hebrew-language used-car listings (`dataset (3).csv`, 20 columns: manufacturer, model, year, ownership hand, gear, engine, area, city, colour, km, price and more). 1,485 rows remain after cleaning.

## Approach
1. **Cleaning:** parsed numeric fields, imputed missing numbers with the median and missing categories with "Unknown", and removed year outliers with the IQR rule.
2. **Feature engineering:** car age, an age × ownership-hand interaction, listing season and a combined model–manufacturer field.
3. **Model:** ElasticNet with standard scaling and one-hot encoding in a `ColumnTransformer`, tuned by `GridSearchCV` over alpha and l1_ratio with 10-fold cross-validation.
4. **Interpretation:** permutation importance and model coefficients.

## Results
- Test RMSE: **₪11,057**.
- Strongest price drivers: car age and the number of previous owners (age × hand interaction, year, hand).

## Files
- `final_project.ipynb`: the full pipeline with explanations
- `dataset (3).csv`: the listings data

**Tools:** Python · pandas · NumPy · scikit-learn

A deployed version with a Flask web app is in [Final-project-in-the-course-Advanced-Data-Mining-and-Analysis-in-Python-](https://github.com/steinamit/Final-project-in-the-course-Advanced-Data-Mining-and-Analysis-in-Python-).
