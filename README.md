E-commerce Survival Analysis

This repository contains a machine learning project focused on Survival Analysis within an e-commerce context. The primary objective is to model and predict the "time-to-event," specifically, how long a product remains on the market before being purchased.

Project Overview

In e-commerce, understanding the factors that influence the speed of a sale is critical for inventory management and pricing strategies. This notebook implements a survival modeling pipeline to estimate the probability of a product being sold over time, accounting for right-censored data (products that haven't been sold yet).

Tech Stack

Language: Python 3.12

Key Libraries:

XGBSE: XGBoost Survival Embeddings for advanced gradient-boosted survival models.

Scikit-Survival: For Kaplan-Meier estimation and data structuring.

Feature-engine: For Weight of Evidence (WoE) encoding and categorical treatment.

Lifelines: For statistical survival analysis visualization.

Scikit-Learn: For data preprocessing and scaling.

Data & Methodology

1. Data Preprocessing

Categorical Encoding: High-cardinality features like product_id and brand are treated using RareLabelEncoder and WoEEncoder to capture their predictive power without overfitting.

Constant Feature Removal: Low-variance features that provide little information are automatically dropped.

Scaling: Numeric features are normalized using MinMaxScaler to ensure stable model convergence.

2. Exploratory Data Analysis

The project utilizes the Kaplan-Meier Estimator to visualize the baseline survival function of the inventory, providing a clear picture of the probability that a product remains unsold at various time intervals.

3. Modeling Approach

The core of the analysis uses XGBSE (XGBoost Survival Embeddings). Unlike standard regression, this approach handles:

Censoring: Distinguishes between products that were sold (event occurred) and those still listed (censored).

Non-linearities: Captures complex relationships between price, color, brand, and sales speed.

Results
The final models produce expected survival times and probability curves for each product. The notebook includes a workflow to extract the "Top 17" products with the highest expected probability of imminent sale, which are exported for business review or Kaggle submission.
