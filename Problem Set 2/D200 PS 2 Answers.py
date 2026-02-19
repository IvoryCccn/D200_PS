# =====================================================
#                     Problem Set 2
# =====================================================

"""
crsid: nc681
name: Ning Chen
group: personal
"""

# %%
# --------------------- Problem 1 ---------------------
import numpy as np
import pandas as pd

from choice_learn.datasets import load_modecanada
from choice_learn.data import ChoiceDataset
from choice_learn.models import ConditionalLogit

# (1a)
transport_df = load_modecanada(as_frame=True)
print(f"Dataset shape: {transport_df.shape}")
print(transport_df.head(8))

case1 = transport_df[transport_df["case"] == 1].copy()
print(case1)

n_alts = case1.shape[0]
chosen_alts = case1.loc[case1["choice"] == 1, "alt"].iloc[0]

print(f"\n(1a):")
print(f"Number of available alternatives in case 1: {n_alts}")
print(f"Chosen alternative in case 1: {chosen_alts}")

# (1b)
canada_dataset = ChoiceDataset.from_single_long_df(
    df=transport_df,
    items_id_column="alt",
    choices_id_column="case",
    choices_column="choice",
    shared_features_columns=["income"],
    items_features_columns=["cost", "freq", "ovt", "ivt"],
    choice_format="one_zero",
)

print(f"\n(1b):")
print(canada_dataset.summary())


# (1c)
# U_ij = beta_intercept_j
#     + beta_cost * cost_j
#     + beta_freq * freq_j
#     + beta_ovt  * ovt_j
#     + beta_ivt_j * ivt_j
#     + beta_income_j * income_i

# shared coefficients: cost, freq, ovt
# alternative-specific: intercept, ivt, income

print(f"\n(1c):")
print("""
The disutility of in-vehicle time differs across transport modes. 
Driving requires attention and effort, making time more burdensome, 
while time on trains or planes is more comfortable and can often 
be used productively. Therefore, one minute of travel time does not 
affect utility equally across modes, and mode-specific coefficients 
allow the model to capture these differences.
""")

# %%
# (1d)
model = ConditionalLogit(optimizer="lbfgs")

# a. shared coefficients: same across all alternatives
model.add_shared_coefficient(feature_name="cost", items_indexes=[0, 1, 2, 3])
model.add_shared_coefficient(feature_name="freq", items_indexes=[0, 1, 2, 3])
model.add_shared_coefficient(feature_name="ovt",  items_indexes=[0, 1, 2, 3])

# b. alternative-specific coefficients (Normalize one alternative to zero, reference = car (index 3))
model.add_coefficients(feature_name="ivt",    items_indexes=[0, 1, 2])  # air, bus, train
model.add_coefficients(feature_name="income", items_indexes=[0, 1, 2])  # air, bus, train
model.add_coefficients(feature_name="intercept", items_indexes=[0, 1, 2])  # air, bus, train

# e. fit and report
model.fit(canada_dataset, get_report=True)

print(f"\n(1d):")
print(model.report)

"""
    Coefficient Name  Coefficient Estimation  Std. Err    z_value        P(.>z)
0          beta_cost               -0.024272  0.002348 -10.337507  0.000000e+00
1          beta_freq                0.089421  0.003780  23.655085  0.000000e+00
2           beta_ovt               -0.034442  0.002114 -16.294903  0.000000e+00
3         beta_ivt_0                0.025394  0.002820   9.006035  0.000000e+00
4         beta_ivt_1               -0.008410  0.001606  -5.236915  1.632832e-07
5         beta_ivt_2               -0.007901  0.000691 -11.437542  0.000000e+00
6      beta_income_0                0.038369  0.003220  11.917126  0.000000e+00
7      beta_income_1               -0.025134  0.005197  -4.835948  1.325129e-06
8      beta_income_2                0.013653  0.002894   4.717676  2.385544e-06
9   beta_intercept_0               -1.834629  0.216308  -8.481568  0.000000e+00
10  beta_intercept_1               -2.196729  0.286975  -7.654767  1.931788e-14
11  beta_intercept_2               -0.236236  0.207094  -1.140716  2.539881e-01
"""

# (1e)
print(f"\n(1e):")

print("""
1. Sign of beta_cost = -0.0243 is negative as expected, because 
higher cost reduces the utility of a mode, which is consistent with 
rational economic behaviour.
""")

print("""
2. Interceptes: Air (-1.835), Bus (-2.197), Train (-0.236)
Train has the highest baseline utility among the non-car alternatives; 
Car (reference = 0) has the overall highest baseline utility, 
reflecting its dominant modal share in the dataset."
""")

print("""
3. Income coefficients:
Air (+0.038) and Train (+0.014), higher income increases preference (premium mode), 
but slight positive effect for train; Bus (-0.025), higher income reduces 
preference for bus (budget mode). This makes intuitive sense: wealthier 
travellers favour faster/more comfortable modes and avoid bus.
""")

# (1f)
chosen_car = transport_df[transport_df["alt"] == "car"]["choice"].sum()
total_cases = transport_df["case"].nunique()
market_share_car = chosen_car / total_cases

beta_cost = -0.024272
mean_car_cost = 63.7637
eta = beta_cost * mean_car_cost * (1 - market_share_car)

print(f"\n(1f):")
print(f"Observed car market share: {market_share_car:.4f}")
print(f"Own-price elasticity for car: {eta:.4f}")
print("""
A 1% increase in car cost reduces the probability of choosing car 
by approximately 0.76%. The elasticity is less than 1 in absolute value, 
indicating car demand is price-inelastic — consistent with car being the 
dominant mode (51% market share) with no perfect substitute for many 
travellers on this route.
""")



# %%
# --------------------- Problem 2 ---------------------
from choice_learn.datasets import load_expedia
try:
    expedia_dataset = load_expedia(as_frame=False, preprocessing="rumnet")
except FileNotFoundError as e:
    print(e)  # exact saving path

# (2a)
import pandas as pd
import os
from choice_learn.datasets import load_expedia
from choice_learn.data import ChoiceDataset

data_path = r"d:\Miniconda\envs\python3_11_D200\Lib\site-packages\choice_learn\datasets\data\expedia.csv"

small_path = r"d:\Miniconda\envs\python3_11_D200\Lib\site-packages\choice_learn\datasets\data\expedia_backup.csv"

if not os.path.exists(small_path):
    df_small = pd.read_csv(data_path, nrows=5000)
    os.rename(data_path, small_path)
    df_small.to_csv(data_path, index=False)
    print(f"Saved {len(df_small)} rows as new expedia.csv")

expedia_dataset = load_expedia(as_frame=False, preprocessing="rumnet")

print(f"\n(2a)")
print(expedia_dataset.summary())

print("""
The dataset has 207 choices, 39 items, 10 continuous item features plus 1 
categorical, 13 shared features. Choice set sizes vary (up to 39 hotels per search)
""")

n_total = len(expedia_dataset.choices)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

train_dataset = expedia_dataset[:n_train]
test_dataset = expedia_dataset[n_train:]

# (2b)
from choice_learn.models import ConditionalLogit
import tensorflow as tf
cl_model = ConditionalLogit(optimizer="lbfgs")

# Shared coefficients
cl_model.add_shared_coefficient(feature_name="log_price", items_indexes=list(range(39)))
cl_model.add_shared_coefficient(feature_name="prop_starrating", items_indexes=list(range(39)))
cl_model.add_shared_coefficient(feature_name="prop_review_score", items_indexes=list(range(39)))
cl_model.add_shared_coefficient(feature_name="prop_brand_bool", items_indexes=list(range(39)))
cl_model.add_shared_coefficient(feature_name="prop_location_score1", items_indexes=list(range(39)))
cl_model.add_shared_coefficient(feature_name="prop_location_score2", items_indexes=list(range(39)))

# Fit model
cl_model.fit(train_dataset, get_report=True)
print(cl_model.report)

"""
            Coefficient Name  Coefficient Estimation  Std. Err   z_value        P(.>z)
0             beta_log_price               -0.316484  0.039713 -7.969323  1.554312e-15
1       beta_prop_starrating                0.200356  0.148323  1.350810  1.767562e-01
2     beta_prop_review_score                0.239599  0.177237  1.351856  1.764214e-01
3       beta_prop_brand_bool               -0.260854  0.268910 -0.970044  3.320248e-01
4  beta_prop_location_score1                0.279772  0.146862  1.904995  5.678073e-02
5  beta_prop_location_score2                0.787995  0.323057  2.439186  1.472040e-02
"""

# Cross-entropy loss
probs = cl_model.predict_probas(test_dataset)
true_choices = test_dataset.choices

y_true = np.zeros((n_test, 39))
for i, c in enumerate(true_choices):
    y_true[i, c] = 1

loss_fn = tf.keras.losses.CategoricalCrossentropy()
ce_loss = loss_fn(y_true, probs).numpy()
print(f"\n(2b)")
print(f"ConditionalLogit Test Cross-Entropy Loss: {ce_loss:.4f}")

# (2c)
print(f"\n(2c)")
print("""
Interpretation: 
(a) β_log_price = −0.317 (p<0.001): Negative and highly significant. 
Higher price strongly reduces booking probability, which is economically sensible. 

(b) β_location_score2 = +0.788 (p=0.015): Most impactful positive feature. 
Location desirability strongly increases utility. 

(c) β_location_score1 = +0.280 (p=0.057): Marginally significant positive effect. 

(d) β_starrating, β_review, β_brand: Positive signs but not statistically significant at 5%.
""")

# (2d) RUMnet model
from choice_learn.models import RUMnet

rumnet_model = RUMnet(
    num_products_features=11,
    num_customer_features=13,
    width_eps_x=20,
    width_eps_z=20,
    depth_eps_x=2,
    depth_eps_z=2,
    heterogeneity_x=5,
    heterogeneity_z=5,
    tol=1e-6,
    width_u=20,
    depth_u=2,
    optimizer="adam",
    epochs=200,
    batch_size=32,
    lr=0.001,
)

rumnet_model.instantiate()
rumnet_model.fit(train_dataset)

# Cross-entropy loss
probs_rum = rumnet_model.predict_probas(test_dataset)
ce_loss_rum = loss_fn(y_true, probs_rum).numpy()
print(f"\n(2d)")
print(f"RUMnet Test Cross-Entropy Loss: {ce_loss_rum:.4f}")
print(f"ConditionalLogit Test Cross-Entropy Loss: {ce_loss:.4f}")

# (2e) Comparison: Conditional Logit vs RUMnet
print(f"\n(2e)")
print("""
Key Tradeoffs:

1. Predictive Performance:
   - Conditional Logit: 2.4705 (Better)
   - RUMnet: 3.0711
   - Reason: Only ~165 training choices; RUMnet has too many parameters for 
     this sample size, causing poor generalization.

2. Interpretability:
   - Conditional Logit: HIGH. Coefficients have direct economic meaning (e.g. 
     beta_price = -0.316 means higher price reduces booking probability).
   - RUMnet: LOW. Neural network weights are not interpretable.

3. Flexibility:
   - Conditional Logit: LOW. Assumes utility is linear in features.
   - RUMnet: HIGH. Can capture nonlinear and heterogeneous preferences across customers.

4. Data Requirements:
   - Conditional Logit: Works well with small samples.
   - RUMnet: Needs large datasets (100k+ choices) to outperform simpler models. 
     With full Expedia data, RUMnet would likely win.

5. Speed:
   - Conditional Logit: Fast, convex optimization (L-BFGS).
   - RUMnet: Slow, non-convex gradient descent.

Conclusion:
   For small datasets and economic interpretation -> Conditional Logit.
   For large datasets with complex preference patterns -> RUMnet.
""")