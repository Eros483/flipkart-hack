<div align="center">

# GridLock Hackathon 2.0 Submission
Modelling spatial temporal data from Bangalore Traffic.

</div>

---

### **Overview**
The guide is structured in reverse, i.e the latest approach mentioned first, the initial steps mentioned last.

The structure of the zip is as follows:

```
gridlock
├── train.csv
├── test.csv
├── submission.py
├── dataset_exploration.ipynb
├── approach.pdf
├── README.md
├── requirements.txt
├── DISCARDED_optuna_finetuning_work.ipynb
└── DISCARDED_lightgbm_engineering_work.ipynb
```

To run the code to generate the final submission:
1. (Optional environment creation) Use `conda` or your preferred environment manager.
```
conda create -n flipkart python=3.12
conda activate flipkart
```

2. Setup and run (can use `uv` if desired)
```
pip install -r requirements.txt
python submission.py
```
---

**Tools Used**
- Standard data manipulation (Pandas and Numpy)

- Machine Learning frameworks (LightGBM, XGBoost, CatBoost)

- Deep Learning frameworks: PyTorch

- Scikit learn ecosystem (yeo-johnson transform, Kmeans clustering, Kfold)

- Spatial Engineering (Pygeohash and geohash2)

- Optuna for hyperparameter studies.

- Matplotlib, seaborn for visualisation.

---

**Further Observations**

- It seems that the competition dataset was leaked, however to maintain fairness in accordance with the rules, no external data was used.

- Finetuned a transformer (Qwen 2.5 8B) on the dataset, extreme overfitting.

---

**Revamped Approach for feature engineering and model ensemble**

- Resolved day/night distribution shift by moving to a weighted split for training where the training dataset's day 48 features are weighed lightly, and day 49 features are given more weight.

- Resolved target leakage for `geo_mean` and `geo_ts_mean` features to control OOF.

- Introduced even more lag features (intervals of +, - 15, 30, 60 and 120 minutes)

- Interaction features for specific local temporal dynamics.

- Expanded geographical prefix features for broader spatial patterns.

- Validation set switched to temporal validation from the last 3 timestamps of day 49, to serve as a BETTER proxy for test dataset scoring.

**The result: 91.05**

---

**Applying Deep Learning**

Combined the previous lightGBM benchmark model, with a deep net to identify features seperately, and blend/ensemble the models together.

- Approach 1: Use a LSTM -> **Score dropped to 86**

- Approach 2: Use a Temporal Convolutional Network -> **Score dropped to 88**

The conclusion was that tree based models remained superior, and a different approach was required.

---

**Picking an Ensemble Set Up**

- Used XGBoost and CatBoost alongisde LightGBM to explore our excellent features in different ways.

- Built a tri model ensemble, optimized blend weights using optuna.

**Score dropped to 87.**

---

**Transition to Spatio-Temporal**

To move forward from just time series modelling, we shifted to linking both, and finding relationships
between time and space together.

- Macro-Resolution (geo4): Extracted the first 4 characters of the geohash to create 39km x 19km bounding boxes, capturing city-wide district patterns.
- Traffic Archetypes (K-Means): Pivoted Day 48's 24-hour curves and clustered the locations into 5 archetypes (e.g., Business Parks vs. Residential Sleepers).

- The Geohash Ring: Attempted to calculate the spillover demand of the 8 surrounding grids
(though LightGBM's native lat/lon awareness mostly absorbed this).

**The Result: 89.4861**

---

**T-24 hours LAG Trap**

K fold encoding was too smooth, and since all time stamps in day 49 are present in day 49, thought providing look ups would help.

The Experiment: Replaced statistical averages with strict dictionaries mapped directly from Day 48.

- The Result: A catastrophic drop to 81.2.

- Day 48 and Day 49 did not share the same baseline traffic behavior (e.g., transitioning from a quiet Sunday to a massive Monday sale event).

- Further attempting to map and find relationship across the exact same data points that only varied in the day itself made no difference, there was no relationship to be found.

---

**Baseline and Preprocessing**

Post EDA, we observed that, a per road type modellling stategy was needed, i.e seperate lightGBM models for highway, street and residential.

- We started with a robust pipeline featuring a per-road-type modeling strategy (training separate LightGBM models for Highway, Street, and Residential).

- Spatial Decoding: Translated geohash strings into continuous lat and lon coordinates to give the trees a physical map.

- Cyclical Time: Encoded time_minutes using sine/cosine transformations so the model understood that 23:59 is adjacent to 00:01.

- Target Transformation: Applied a Yeo-Johnson power transform to the demand target to reduce extreme skewness and stabilize the RMSE loss function.

- OOF Encoding: Used 5-Fold Out-of-Fold target encoding on specific interactions like (geo5, hour) to capture neighborhood baselines without target leakage.

**Baseline Score: ~89.20**