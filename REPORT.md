# Restaurant ETA Prediction System

## 1. Problem Definition

The goal of this project was to build a machine learning system that predicts the delivery/fulfilment time of restaurant orders.

Because the target variable, `Time_taken(min)`, is a continuous numerical value measured in minutes, I treated the task as a supervised regression problem.

I defined the prediction point as the moment a customer places an order. It is this assumption that affected feature selection for me in this project. My thinking was, information that would only become available later in the delivery process should not be available to the model at prediction time.

It is for this reason that I excluded `Time_Order_picked`. Even though it exists in the dataset, the pickup time would not yet be known when an ETA is first generated. Using that feature would introduce temporal leakage and could produce an unrealistical evaluation.

---

## 2. Dataset and Data Preparation

The project used the Food Delivery Time dataset on Kaggle provided for the assignment.

The training dataset contained 45,593 observations. The target column, `Time_taken(min)`, was initially stored as strings like `(min) 24`, so before modelling, it was parsed into a numerical target.

Upon further inspection I also saw that several numerical columns were stored as strings and missing values were sometimes represented by the literal string `"NaN"`. These values were also converted into appropriate numerical values or actual missing values before modelling.

### Feature Engineering

I created additional features from the raw data:

- `distance_km`: this is the straight-line distance between the restaurant and delivery location using the Haversine formula.
- `order_hour`: this is the numerical representation of the order time.
- `day_of_week`: this is extracted from the order date.

Raw identifiers such as `ID` and `Delivery_person_ID` were excluded. The raw coordinate, date and time columns were also removed after their useful information had been transformed into engineered features.

### Coordinate Quality

Exploratory analysis revealed a data problem in the geographical coordinates.

The initial Haversine calculation produced a maximum delivery distance of approximately 19,693 km, which was clearly inconsistent with a food-delivery problem.

Investigation on this showed two issues:

1. Some restaurant coordinates contained negative signs while the corresponding delivery coordinates were geographically close when their absolute values were considered.
2. 3,640 restaurant locations (7.98%) were recorded as `(0, 0)`.

The restaurant coordinate sign inconsistencies were then corrected before calculating distance. And `(0,0)` locations were treated as missing rather than as real geographical positions.

After correction, the distance distribution became much more plausible:

- Mean distance: 9.72 km
- Median distance: 9.19 km
- Maximum distance: 20.97 km

The 3,640 invalid locations resulted in missing distance values, which were later handled by the preprocessing pipeline.

---

## 3. Preprocessing Pipeline

The data was split into training and test sets using an 80/20 split with `random_state=42`.

This produced:

- Training set: 36,474 orders
- Test set: 9,119 orders

The split was performed before fitting any learned preprocessing operations.

I used scikit-learn's `Pipeline` and `ColumnTransformer` to keep preprocessing and model training together and reduce the risk of train-test contamination.

### Numerical Features

The numerical features were:

- Delivery person age
- Delivery person rating
- Vehicle condition
- Number of multiple deliveries
- Distance
- Order hour

Missing numerical values were filled using median imputation, followed by `StandardScaler`.

### Categorical Features

The categorical features were:

- Weather conditions
- Road traffic density
- Type of order
- Type of vehicle
- Festival
- City
- Day of week

Missing categorical values were filled using the most frequent category, followed by `OneHotEncoder(handle_unknown="ignore")`.

Keeping these learned preprocessing operations inside the Pipeline ensures that statistics such as medians, scaling parameters and category mappings are learned from training data rather than from the complete dataset.

---

## 4. Model Selection

Before training the main regression models, I established a simple mean-prediction baseline.

The baseline produced:

| Model    |   MAE |  RMSE |     R² |
| -------- | ----: | ----: | -----: |
| Baseline | 7.579 | 9.364 | ~0.000 |

I then compared three regression algorithms using the same feature set and preprocessing approach:

| Model             | MAE (min) | RMSE (min) |        R² |
| ----------------- | --------: | ---------: | --------: |
| Random Forest     | **3.300** |  **4.169** | **0.802** |
| Gradient Boosting |     3.663 |      4.600 |     0.759 |
| Ridge Regression  |     4.803 |      6.045 |     0.583 |
| Baseline          |     7.579 |      9.364 |    ~0.000 |

Ridge Regression provided a regularised linear benchmark but performed worse than the tree-based models, suggesting that the ETA problem contains nonlinear relationships and interactions that are not captured as effectively by the linear model.

Of the three, Random Forest produced the strongest initial results, so I selected it for hyperparameter tuning.

---

## 5. Hyperparameter Tuning

I used `RandomizedSearchCV` with 3-fold cross-validation on the training set.

I optimised Mean Absolute Error (MAE) because ETA is measured in minutes, making MAE directly interpretable as the average magnitude of prediction error.

The search explored:

- Number of trees
- Maximum tree depth
- Minimum samples required to split a node
- Minimum samples per leaf
- Number of features considered at each split

The selected configuration was:

```text
n_estimators      = 200
max_depth         = 20
min_samples_split = 10
min_samples_leaf  = 1
max_features      = 0.7
```

The best cross-validation MAE was 3.224 minutes

The test set was not used to select these hyperparameters.

---

## 6. Final Evaluation

After tuning, the selected Random Forest was evaluated on the held-out test set.

| Metric |        Result |
| ------ | ------------: |
| MAE    | 3.240 minutes |
| RMSE   | 4.072 minutes |
| R²     |         0.811 |

The MAE means that the predicted ETA differed from the actual delivery time by approximately 3.24 minutes on average.

The RMSE of 4.07 minutes is higher than the MAE because RMSE penalises larger prediction errors more strongly.

The R² of 0.811 indicates that the model explained approximately 81.1% of the variance in delivery time on the held-out test set relative to predicting the target mean.

Compared with the baseline MAE of 7.579 minutes, the tuned model reduced MAE by approximately 57% on this split.

The cross-validation MAE of 3.224 minutes was also close to the held-out test MAE of 3.240 minutes, indicating similar performance across the training cross-validation folds and the final test sample.

---

## 7. Error Analysis

The final model's errors were not distributed uniformly across all orders.

### Traffic

Among orders with known traffic information:

| Traffic | Mean Absolute Error |
| ------- | ------------------: |
| Low     |            2.76 min |
| Medium  |            3.11 min |
| High    |            3.39 min |
| Jam     |            3.68 min |
| Missing |            6.35 min |

Prediction error was higher under heavier traffic conditions in this test set. Orders with missing traffic information performed substantially worse.

### Weather

Errors across known weather categories ranged from approximately 2.94 to 3.37 minutes.

Missing weather information was associated with a considerably higher MAE of approximately 6.35 minutes.

The differences between the known weather categories were relatively small, so I would not conclude from this experiment alone that a particular weather condition systematically causes poorer model performance.

### Distance

| Distance | Mean Absolute Error |
| -------- | ------------------: |
| 0–5 km   |            2.94 min |
| 5–10 km  |            3.12 min |
| 10–15 km |            3.44 min |
| 15+ km   |            3.36 min |

The model was generally more accurate for shorter deliveries, although prediction error did not increase uniformly across every distance group.

### Missing Information

Missing input data was one of the clearest weaknesses identified during error analysis.

| Input Data                   | Mean Absolute Error |
| ---------------------------- | ------------------: |
| Complete rows                |            3.10 min |
| At least one missing feature |            3.92 min |

Error generally increased as more input features were missing. For example, observations with five or six missing features showed substantially higher average errors, although these groups contained relatively few observations.

This demonstrates that imputation allows the pipeline to continue making predictions when information is unavailable, but it cannot fully replace the predictive information contained in the missing values.

### City

Semi-urban orders showed a higher observed MAE of approximately 4.00 minutes, but there were only 27 semi-urban observations in the test set. This sample is too small to treat the result as strong evidence that the model systematically performs worse for semi-urban orders.

---

## 8. Feature Importance

The fitted Random Forest's highest individual transformed feature importances included:

| Feature                 | Importance |
| ----------------------- | ---------: |
| Delivery person rating  |      0.208 |
| Low traffic indicator   |      0.115 |
| Multiple deliveries     |      0.113 |
| Distance                |      0.108 |
| Delivery person age     |      0.096 |
| Vehicle condition       |      0.076 |
| Sunny weather indicator |      0.070 |
| Fog indicator           |      0.040 |
| Cloudy indicator        |      0.040 |
| Order hour              |      0.036 |

Delivery-person rating was the most influential individual transformed feature. Distance, multiple deliveries, rider characteristics, vehicle condition, traffic and weather also contributed to the model's predictions.

Because categorical variables are one-hot encoded, their importance is distributed across multiple transformed columns. These feature importances therefore describe what the fitted model relied on for prediction and should not be interpreted as evidence of causality.

---

## 9. Limitations and Future Work

The current experiment uses a random train-test split. A production ETA system should additionally be evaluated using a temporal split, where the model is trained on older orders and evaluated on newer orders. This would more closely represent the real deployment scenario.

The error analysis also showed that missing operational information reduces prediction quality. Improving upstream data completeness, particularly for traffic, weather, rider and location information, could therefore improve ETA reliability.

The current distance feature uses Haversine distance, which represents straight-line geographical distance rather than the actual road route. A production system could use route distance and estimated travel duration from a routing service.

---

## 10. Production Considerations

This project is a prototype rather than a production-ready ETA system. Before deployment, additional engineering concerns like the following would need to be addressed.

### Data and Feature Availability

The production system would need to guarantee that every feature used by the model is available and correctly defined at prediction time. Missing operational information was associated with higher prediction error in the experiment, so data-quality monitoring would be important.

The preprocessing pipeline can handle missing values and unseen categorical values through imputation and `OneHotEncoder(handle_unknown="ignore")`, but this prevents inference failures rather than guaranteeing reliable predictions when important information is absent.

### Model and Data Drift

Traffic patterns, restaurant operations, rider behaviour and delivery regions can change over time. A model trained on historical data may therefore become less representative of current deliveries.

Production monitoring should track feature distributions and prediction errors over time. Where ground-truth delivery times become available after orders are completed, metrics such as MAE and RMSE could be monitored on recent orders and compared with historical performance.

A retraining strategy should be based on observed degradation or meaningful changes in the underlying data rather than assuming that a fixed model remains reliable indefinitely.

### New Cities and Categories

The model may encounter cities, vehicle types or other categorical values that were not represented during training. The current encoder is configured to tolerate unknown categories, preventing the pipeline from failing, but predictions for substantially different operating environments may still be unreliable.

Expansion into a new city should therefore include validation using representative data from that environment.

### Distance and Routing

The current `distance_km` feature uses Haversine distance. This measures straight-line geographical distance and does not account for road networks, route restrictions or actual travel time.

A production ETA system could replace or supplement this feature with route distance and estimated travel duration from a routing system.

### Prediction Uncertainty

A single ETA such as "31 minutes" can imply more certainty than the model actually has. A production system should investigate prediction intervals or other uncertainty estimates so that the product can communicate realistic ETA ranges where appropriate.

### Validation Strategy

The current experiment uses a random train-test split. Before deployment, I would additionally perform temporal validation by training on older orders and evaluating on newer orders. This would more closely simulate how the model would encounter future orders in production.

## 11. Conclusion

This project developed a leakage-aware scikit-learn pipeline for restaurant ETA prediction.

A mean baseline, Ridge Regression, Random Forest and Gradient Boosting were evaluated. Random Forest produced the strongest initial performance and was subsequently fine-tuned using cross-validation.

The final tuned Random Forest achieved:

- MAE: 3.240 minutes
- RMSE: 4.072 minutes
- R²: 0.811

Beyond the aggregate metrics, error analysis showed that predictions were less reliable when operational information was missing and somewhat less accurate under heavier traffic and for longer-distance orders.
