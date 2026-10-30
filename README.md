# Used-Car Price Prediction

> Estimating the price of a used car in Israel from its listing with a tuned, interpretable ElasticNet model.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

## Highlights

- End-to-end pipeline on 1,500 real Hebrew-language car listings.
- Engineered features: car age, age × previous-owners interaction, listing season, model–manufacturer.
- ElasticNet tuned by grid search with 10-fold cross-validation.
- Test RMSE of ₪11,057, with permutation importance to explain what drives price.

## Approach

1. **Cleaning:** numeric parsing, median imputation, "Unknown" for missing categories, IQR outlier removal on year (1,500 → 1,485 rows).
2. **Features:** car age, age × hand interaction, season of listing, combined model–manufacturer field.
3. **Modeling:** `ColumnTransformer` (scaling + one-hot encoding) feeding ElasticNet, tuned with `GridSearchCV` over alpha and l1_ratio, 10-fold CV, 80/20 split.
4. **Interpretation:** permutation importance and coefficients.

## Results

| Metric | Value |
|---|---|
| Test RMSE | ₪11,057 |
| Top drivers (permutation importance) | Age × hand interaction, year, car age, number of previous owners |

## Repository structure

| Path | Description |
|---|---|
| `final_project.ipynb` | Full pipeline with explanations |
| `dataset (3).csv` | 1,500 used-car listings, 20 columns |

## Context

Final project, machine learning course. Team project by Nadav Mashiach and Amit Stein.

## Credits

A deployed version of this model, served as a Flask web app, lives in [Final-project-in-the-course-Advanced-Data-Mining-and-Analysis-in-Python-](https://github.com/steinamit/Final-project-in-the-course-Advanced-Data-Mining-and-Analysis-in-Python-).
