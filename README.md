# Using Classification Models to Improve Telemarketing Strategies

## Business Context
A Portuguese banking institution was conducting a telemarketing campaign to solicit new business for its Term Deposit products. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required.  

## Research Objectives
The institution compiled over 40,000 records encompassing a wide array of attributes, including demographic information, details of previous customer interactions, and macroeconomic indicators. This exploratory data analysis and research aims to leverage the information to improve the bank's telemarketing strategies for promoting Term Deposit products.

## Data Source
The data is obtained from the work of Moro, S., Rita, P., and Cortez, P. (2012). Bank Marketing. UCI Machine Learning Repository. [https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306). "The data is related to direct marketing campaigns of a Portuguese banking institution." 

### Data Dictionary
#### Bank client data:
- `age` (numeric)
- `job`: type of job (categorical: "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown")
- `marital`: marital status (categorical: "divorced", "married", "single", "unknown"; note: "divorced" means divorced or widowed)
- `education` (categorical: "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown")
- `default`: has credit in default? (categorical: "no", "yes", "unknown")
- `housing`: has housing loan? (categorical: "no", "yes", "unknown")
- `loan`: has personal loan? (categorical: "no", "yes", "unknown")

#### Related to the last contact of the current campaign:
- `contact`: contact communication type (categorical: "cellular", "telephone")
- `month`: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
- `day_of_week`: last contact day of the week (categorical: "mon", "tue", "wed", "thu", "fri")
- `duration`: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#### Other attributes:
- `campaign`: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- `pdays`: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- `previous`: number of contacts performed before this campaign and for this client (numeric)
- `poutcome`: outcome of the previous marketing campaign (categorical: "failure", "nonexistent", "success")

#### Social and economic context attributes
- `emp.var.rate`: employment variation rate - quarterly indicator (numeric)
- `cons.price.idx`: consumer price index - monthly indicator (numeric)
- `cons.conf.idx`: consumer confidence index - monthly indicator (numeric)
- `euribor3m`: EURIBOR 3 month rate - daily indicator (numeric)
- `nr.employed`: number of employees - quarterly indicator (numeric)

#### Output variable (desired target): `y` ("yes", "no")

## Exploratory Data Analysis
### Data Quality and Missing Value Treatments:
The dataset contains 41,188 records with 20 attributes and one outcome variable - whether a customer subscribes to a term deposit product. The data quality is generally good with some issues. Here are the highlights:
- There are no duplicates.
- Missing values are observed in several attributes. 26% of the records have at least one missing value in the record. In most cases, missing values are indicated by the term "unknown" as the value in the database.
- One data integrity issue was discovered and these 4,110 (10% of total) records are excluded from the dataset to ensure the reliability of the analysis.

#### Missing values treatments
In most cases, missing values are indicated by the term "unknown" as the value in the database. Records with missing values are kept. The term 'unknown' is preserved as a categorical value to build the models so that models can process these incomplete records and predict their likely outcomes.
<p align='center'>
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/row_missing_val_density.png">
</p>

## Methodology
Four predictive classification models are constructed, fine-tuned, and evaluated, each with distinct approaches: one approximates by considering the nearest data points, another makes rule-based decisions, a third separates classes ('yes'/'no') with a straight line (or segments), and the last one uses complex boundaries to distinguish classes.

Since the main objective is to use the records of the customers who have been contacted in the past or during the current campaigns to predict the likely outcome for customers who have not been contacted, the dataset is split into two – one for training the models and the other for predictions. The training dataset contains 1,060 records with 677 customers subscribing to the Term Deposit product, and 383 do not.

## Model Selection and Performance
Three of the models have the best and similar accuracy scores of 71% to 72%. They differ slightly in making false positive or false negative classifications on a testing dataset with 515 records. The similarity in accuracy scores among the top three models suggests that the problem is complex, and no single model vastly outperforms the others. The Support Vector Machine model that uses complex boundaries to delineate classes has the best ROC curve and a slight edge over the others with an AUC = 0.76. An AUC of 0.5 suggests no discrimination (i.e., the model has no capacity to distinguish between the positive and negative classes), whereas an AUC of 1.0 suggests perfect classification.
<div align="center">
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/roc_auc.png">
</div>
(Please see Supplemental Information on why the KNN model with the best AUC=1 is not considered as the best model.)

## Factors Affecting the Telemarketing Campaign Outcome
The two close runner-up models provide insights on what factors are most likely to affect the outcome of a call to a customer.

The **Decision Tree Classifier** that uses successive binary rules to make decisions indicates the 3-month Euro Interbank Offered Rate (EURIBOR), Day of Week, Month, Consumer Price Index (CPI), and the number of calls, in the order of importance, are the leading factors affecting the outcome. The **Logistic Regression Classifier** that uses straight line segments to make classifications identifies the EURIBOR, Day of Week (where Thursday is the most favorable and Monday is the most unfavorable), the CPI, previous campaign outcome, and number of calls are the key factors affecting the outcome.
<div align="center">
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/decision_tree_cum_importance.png">
</div>
<p></p>
<div align="center">
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/feat_importance_logistic_reg.png">
</div>

Although the two models share almost the same set of leading factors, the Logistic Regression Model is more confident in making negative outcome predictions than the Decision Tree Classifier. The best Support Vector Machine Classifier model classifies both the positive and negative classes with higher probabilities than the two runner-up models.
<div align="center">
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/tree_pred_proba.png">
</div>
Decision Tree Classifier model is more confident in predicting positive outcomes than the Logistic Regression Classifier model shown below.
<p></p>
<div align="center">
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/logreg_pred_proba.png">
</div>
<p></p>
The best model, Support Vector Machine Classifier, has the best balance of predicting both positive and negative outcomes in high confidence as shown below.
<p align="center">
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/svc_pred_proba.png">
</p>

## Actionable Recommendations
**Prioritize High-Probability Customers:** Focus initial efforts on the subset of customers identified by the models as having the highest likelihood of subscribing to the Term Deposit. This approach ensures the efficient allocation of limited resources. The model excepts a prbability-weighted forcast of 4,298 out of 35,551 (12.1%) of customers who has not been contacted to subscribe the the Term Deposit product if the telemarketing campaign continue its course.
<p align="center">
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/svc_predictions.png">
</p>

**Leverage Negative Outcomes for Improving the Models:** Actively incorporating the results of negative outcomes back into the models can provide valuable learning opportunities. This feedback loop can help refine the models, making them more robust and accurate over time.

**Data Collection:** Efforts should be made to collect more high quality data, especially from interactions resulting in negative outcomes. This additional data can help address the current imbalance between positive and negative outcomes in the dataset, potentially leading to better model performance. It would be beneficial to investigate the root causes of the data integrity issue uncovered in this analysis to prevent similar problems in future data collection efforts. 
<p align='center'>
<img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/missing_value_pct_col.png">
</p>

**Continuous Model Evaluation:** Regularly re-evaluate the models with updates to the data to ensure they remain relevant and effective over time. This practice can help identify when a model may need adjustments or replacement due to changing market conditions or customer behaviors.

**Experimentation and Adaptation:** Consider running small-scale experiments with different approaches based on the models' insights. For example, testing different contact strategies based on the day of the week or scaling up the campaign when the interest rate and CPI are more favorable for customers could yield valuable insights and further optimize the campaign's effectiveness.

## Supplemental Information

### Data Integrity Issue:
- If a customer has never been contacted by telemarketing in the past, `pdays` should be `999` and `poutcome` should be `'nonexistent'`, and vice versa. However, 4,110 violations of this integrity rule are observed. These records are excluded from the analysis.

### Feature Engineering and Selection
- The dataset contains 10 numeric features, 10 categorical features, and one binary target variable. Missing values in all categorical features are filled with the term `'unknown'`. All categorical features are one-hot encoded. However, prior to one-hot encoding, LightGBM, Microsoft’s Gradient Boosting Machine implementation, is used as a feature selection tool to filter out the least important categorical features.
<p align="center"><img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/gbm_cumulative_gain_by_features.png" width=600></p>

### Variable Correlations
- Strong correlations among a set of features are identified, and two of the features are manually dropped to break the strong correlations for better interpretability of the models.
<p align="center"><img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/heatmap_correlations.png" width=600></p>
 
### Performance Metrics
- Although the K-nearest Neighbors Classifier, with an accuracy rate of 99% and AUC = 1, it is not considered the best model because both the postive and negative predictions have low probabilities, lower than the baseline probability, as shown below.
<p align="center"><img src="https://github.com/chihming-chen/using-classification-models-to-improve-telemarketing-strategies/blob/main/images/knn_pred_proba.png" width=600></p>
