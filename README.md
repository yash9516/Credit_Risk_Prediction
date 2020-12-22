## Context

LendingClub is an American peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform. The company claims that $15.98 billion in loans had been originated through its platform up to December 31, 2015.

## Inspiration

This is great dataset to measure the credit risk of P2P lending loans from the Year 2015 to 2020.

<hr/>

# P2P Credit Risk Lending Prediction

<hr/>

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model that can predict wether or nor a borrower will pay back their loan? This way in the future when **we get a new potential customer we can assess whether or not they are likely to pay back the loan.**

## Table of Contents

* [1. Import Dataset](#1)
* [2. Limit the Feature Space](#2)
 * [2.1 Drop features missing more than 30% data](#2.1)
 * [2.2 Only keep loan features known to potential investors](#2.2)
* [3. Data Analysis and Pre-processing](#3)
 * [3.1 Overlook on Data](#3.1)
 * [3.2  Pre-processing on each Variable](#3.2)
* [4. More Pre-processing](#4)
 * [4.1 Convert loan status to 0/1 charge-off indicator](#4.1)
 * [4.2 Create dummy variables](#4.2)
 * [4.3 Train/test split](#4.3)
* [5. Linear Dependence of Charge-off on the Predictors](#5)
 * [5.1 Pearson correlation](#5.1)
* [6. Model Training and Testing](#6)
 * [6.1 Logistic regression with SGD training](#6.1)
 * [6.2 Random forest classifier](#6.2)
 * [6.3 Tune hyperparameters on the chosen model more finely](#6.3)
 * [6.4 Test set evaluation](#6.4)
* [7. Conclusion](#7)

Link to the dataset: https://www.kaggle.com/ethon0426/lending-club-20072020q1


<a id="1"></a>
# 1. Setup and Import
<hr>


```python
path ='data/Loan_status_2007-2020Q3.csv'
```




```python
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import scipy as sp

# visualization
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
mpl.style.use('ggplot')
sns.set(style='whitegrid')

# scaling and train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)
pd.set_option("display.precision", 4) # Show numbers only with 2 decimal places

# machine learning algorithm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


# evaluation on test data
from sklearn.metrics import classification_report,confusion_matrix
import shap

```


```python
# Create dataframe from dataset file
loans = pd.read_csv(path, low_memory = False, index_col=0, )
loans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>url</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>total_rec_prncp</th>
      <th>total_rec_int</th>
      <th>total_rec_late_fee</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_d</th>
      <th>last_pymnt_amnt</th>
      <th>next_pymnt_d</th>
      <th>last_credit_pull_d</th>
      <th>last_fico_range_high</th>
      <th>last_fico_range_low</th>
      <th>collections_12_mths_ex_med</th>
      <th>mths_since_last_major_derog</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>annual_inc_joint</th>
      <th>dti_joint</th>
      <th>verification_status_joint</th>
      <th>acc_now_delinq</th>
      <th>tot_coll_amt</th>
      <th>tot_cur_bal</th>
      <th>open_acc_6m</th>
      <th>open_act_il</th>
      <th>open_il_12m</th>
      <th>open_il_24m</th>
      <th>mths_since_rcnt_il</th>
      <th>total_bal_il</th>
      <th>il_util</th>
      <th>open_rv_12m</th>
      <th>open_rv_24m</th>
      <th>max_bal_bc</th>
      <th>all_util</th>
      <th>total_rev_hi_lim</th>
      <th>inq_fi</th>
      <th>total_cu_tl</th>
      <th>inq_last_12m</th>
      <th>acc_open_past_24mths</th>
      <th>avg_cur_bal</th>
      <th>bc_open_to_buy</th>
      <th>bc_util</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>mo_sin_old_il_acct</th>
      <th>mo_sin_old_rev_tl_op</th>
      <th>mo_sin_rcnt_rev_tl_op</th>
      <th>mo_sin_rcnt_tl</th>
      <th>mort_acc</th>
      <th>mths_since_recent_bc</th>
      <th>mths_since_recent_bc_dlq</th>
      <th>mths_since_recent_inq</th>
      <th>mths_since_recent_revol_delinq</th>
      <th>num_accts_ever_120_pd</th>
      <th>num_actv_bc_tl</th>
      <th>num_actv_rev_tl</th>
      <th>num_bc_sats</th>
      <th>num_bc_tl</th>
      <th>num_il_tl</th>
      <th>num_op_rev_tl</th>
      <th>num_rev_accts</th>
      <th>num_rev_tl_bal_gt_0</th>
      <th>num_sats</th>
      <th>num_tl_120dpd_2m</th>
      <th>num_tl_30dpd</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>num_tl_op_past_12m</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>revol_bal_joint</th>
      <th>sec_app_fico_range_low</th>
      <th>sec_app_fico_range_high</th>
      <th>sec_app_earliest_cr_line</th>
      <th>sec_app_inq_last_6mths</th>
      <th>sec_app_mort_acc</th>
      <th>sec_app_open_acc</th>
      <th>sec_app_revol_util</th>
      <th>sec_app_open_act_il</th>
      <th>sec_app_num_rev_accts</th>
      <th>sec_app_chargeoff_within_12_mths</th>
      <th>sec_app_collections_12_mths_ex_med</th>
      <th>hardship_flag</th>
      <th>hardship_type</th>
      <th>hardship_reason</th>
      <th>hardship_status</th>
      <th>deferral_term</th>
      <th>hardship_amount</th>
      <th>hardship_start_date</th>
      <th>hardship_end_date</th>
      <th>payment_plan_start_date</th>
      <th>hardship_length</th>
      <th>hardship_dpd</th>
      <th>hardship_loan_status</th>
      <th>orig_projected_additional_accrued_interest</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1077501</td>
      <td>credit_card</td>
      <td>Computer</td>
      <td>860xx</td>
      <td>AZ</td>
      <td>27.65</td>
      <td>0.0</td>
      <td>Jan-1985</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>13648.0</td>
      <td>83.7%</td>
      <td>9.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5863.1552</td>
      <td>5833.84</td>
      <td>5000.00</td>
      <td>863.16</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jan-2015</td>
      <td>171.62</td>
      <td>NaN</td>
      <td>May-2020</td>
      <td>704.0</td>
      <td>700.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1077430</td>
      <td>car</td>
      <td>bike</td>
      <td>309xx</td>
      <td>GA</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>Apr-1999</td>
      <td>740.0</td>
      <td>744.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1687.0</td>
      <td>9.4%</td>
      <td>4.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1014.5300</td>
      <td>1014.53</td>
      <td>456.46</td>
      <td>435.17</td>
      <td>0.00</td>
      <td>122.9</td>
      <td>1.11</td>
      <td>Apr-2013</td>
      <td>119.66</td>
      <td>NaN</td>
      <td>Oct-2016</td>
      <td>499.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1077175</td>
      <td>small_business</td>
      <td>real estate business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>8.72</td>
      <td>0.0</td>
      <td>Nov-2001</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2956.0</td>
      <td>98.5%</td>
      <td>10.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3005.6668</td>
      <td>3005.67</td>
      <td>2400.00</td>
      <td>605.67</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jun-2014</td>
      <td>649.91</td>
      <td>NaN</td>
      <td>Jun-2017</td>
      <td>739.0</td>
      <td>735.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>AIR RESOURCES BOARD</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1076863</td>
      <td>other</td>
      <td>personel</td>
      <td>917xx</td>
      <td>CA</td>
      <td>20.00</td>
      <td>0.0</td>
      <td>Feb-1996</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>5598.0</td>
      <td>21%</td>
      <td>37.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12231.8900</td>
      <td>12231.89</td>
      <td>10000.00</td>
      <td>2214.92</td>
      <td>16.97</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jan-2015</td>
      <td>357.48</td>
      <td>NaN</td>
      <td>Apr-2016</td>
      <td>604.0</td>
      <td>600.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>University Medical Group</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1075358</td>
      <td>other</td>
      <td>Personal</td>
      <td>972xx</td>
      <td>OR</td>
      <td>17.94</td>
      <td>0.0</td>
      <td>Jan-1996</td>
      <td>695.0</td>
      <td>699.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>27783.0</td>
      <td>53.9%</td>
      <td>38.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4066.9082</td>
      <td>4066.91</td>
      <td>3000.00</td>
      <td>1066.91</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jan-2017</td>
      <td>67.30</td>
      <td>NaN</td>
      <td>Apr-2018</td>
      <td>684.0</td>
      <td>680.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
loans.shape
```




    (2925493, 141)



<a id="2"></a>
# 2. Limit the Feature Space

The full dataset has 150 features for each loan. We'll select features in two steps:

1. Drop features with more than 30% of their data missing.
2. Of the remaining features, choose only those that would be available to an investor before deciding to fund the loan.

<a id="2.1"></a>
## 2.1 Drop features missing more than 30% data


```python
missing_fractions = loans.isnull().mean().sort_values(ascending=False)
missing_fractions = missing_fractions.to_frame(name = 'percent_missing').reset_index()
missing_fractions.columns = ['col_name' , 'percent_missing']
missing_fractions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_name</th>
      <th>percent_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hardship_loan_status</td>
      <td>0.9510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hardship_reason</td>
      <td>0.9509</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hardship_status</td>
      <td>0.9509</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hardship_dpd</td>
      <td>0.9509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hardship_type</td>
      <td>0.9509</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(6,3), dpi=90)
missing_fractions.plot.hist(bins=20)
plt.title('Histogram of Feature Incompleteness')
plt.xlabel('Fraction of data missing')
plt.ylabel('Feature count')
```



```python

```

From the above histogram, we see there's a large gap between features missing "some" data (&lt;20%) and those missing "lots" of data (&gt;40%). Because it's generally very difficult to accurately impute data with more than 30% missing values, we drop such columns. First store all variables missing more than 30% data in an alphabetical list:


```python
loans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>url</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>total_rec_prncp</th>
      <th>total_rec_int</th>
      <th>total_rec_late_fee</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_d</th>
      <th>last_pymnt_amnt</th>
      <th>next_pymnt_d</th>
      <th>last_credit_pull_d</th>
      <th>last_fico_range_high</th>
      <th>last_fico_range_low</th>
      <th>collections_12_mths_ex_med</th>
      <th>mths_since_last_major_derog</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>annual_inc_joint</th>
      <th>dti_joint</th>
      <th>verification_status_joint</th>
      <th>acc_now_delinq</th>
      <th>tot_coll_amt</th>
      <th>tot_cur_bal</th>
      <th>open_acc_6m</th>
      <th>open_act_il</th>
      <th>open_il_12m</th>
      <th>open_il_24m</th>
      <th>mths_since_rcnt_il</th>
      <th>total_bal_il</th>
      <th>il_util</th>
      <th>open_rv_12m</th>
      <th>open_rv_24m</th>
      <th>max_bal_bc</th>
      <th>all_util</th>
      <th>total_rev_hi_lim</th>
      <th>inq_fi</th>
      <th>total_cu_tl</th>
      <th>inq_last_12m</th>
      <th>acc_open_past_24mths</th>
      <th>avg_cur_bal</th>
      <th>bc_open_to_buy</th>
      <th>bc_util</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>mo_sin_old_il_acct</th>
      <th>mo_sin_old_rev_tl_op</th>
      <th>mo_sin_rcnt_rev_tl_op</th>
      <th>mo_sin_rcnt_tl</th>
      <th>mort_acc</th>
      <th>mths_since_recent_bc</th>
      <th>mths_since_recent_bc_dlq</th>
      <th>mths_since_recent_inq</th>
      <th>mths_since_recent_revol_delinq</th>
      <th>num_accts_ever_120_pd</th>
      <th>num_actv_bc_tl</th>
      <th>num_actv_rev_tl</th>
      <th>num_bc_sats</th>
      <th>num_bc_tl</th>
      <th>num_il_tl</th>
      <th>num_op_rev_tl</th>
      <th>num_rev_accts</th>
      <th>num_rev_tl_bal_gt_0</th>
      <th>num_sats</th>
      <th>num_tl_120dpd_2m</th>
      <th>num_tl_30dpd</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>num_tl_op_past_12m</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>revol_bal_joint</th>
      <th>sec_app_fico_range_low</th>
      <th>sec_app_fico_range_high</th>
      <th>sec_app_earliest_cr_line</th>
      <th>sec_app_inq_last_6mths</th>
      <th>sec_app_mort_acc</th>
      <th>sec_app_open_acc</th>
      <th>sec_app_revol_util</th>
      <th>sec_app_open_act_il</th>
      <th>sec_app_num_rev_accts</th>
      <th>sec_app_chargeoff_within_12_mths</th>
      <th>sec_app_collections_12_mths_ex_med</th>
      <th>hardship_flag</th>
      <th>hardship_type</th>
      <th>hardship_reason</th>
      <th>hardship_status</th>
      <th>deferral_term</th>
      <th>hardship_amount</th>
      <th>hardship_start_date</th>
      <th>hardship_end_date</th>
      <th>payment_plan_start_date</th>
      <th>hardship_length</th>
      <th>hardship_dpd</th>
      <th>hardship_loan_status</th>
      <th>orig_projected_additional_accrued_interest</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1077501</td>
      <td>credit_card</td>
      <td>Computer</td>
      <td>860xx</td>
      <td>AZ</td>
      <td>27.65</td>
      <td>0.0</td>
      <td>Jan-1985</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>13648.0</td>
      <td>83.7%</td>
      <td>9.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5863.1552</td>
      <td>5833.84</td>
      <td>5000.00</td>
      <td>863.16</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jan-2015</td>
      <td>171.62</td>
      <td>NaN</td>
      <td>May-2020</td>
      <td>704.0</td>
      <td>700.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1077430</td>
      <td>car</td>
      <td>bike</td>
      <td>309xx</td>
      <td>GA</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>Apr-1999</td>
      <td>740.0</td>
      <td>744.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1687.0</td>
      <td>9.4%</td>
      <td>4.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1014.5300</td>
      <td>1014.53</td>
      <td>456.46</td>
      <td>435.17</td>
      <td>0.00</td>
      <td>122.9</td>
      <td>1.11</td>
      <td>Apr-2013</td>
      <td>119.66</td>
      <td>NaN</td>
      <td>Oct-2016</td>
      <td>499.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1077175</td>
      <td>small_business</td>
      <td>real estate business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>8.72</td>
      <td>0.0</td>
      <td>Nov-2001</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2956.0</td>
      <td>98.5%</td>
      <td>10.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3005.6668</td>
      <td>3005.67</td>
      <td>2400.00</td>
      <td>605.67</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jun-2014</td>
      <td>649.91</td>
      <td>NaN</td>
      <td>Jun-2017</td>
      <td>739.0</td>
      <td>735.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>AIR RESOURCES BOARD</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1076863</td>
      <td>other</td>
      <td>personel</td>
      <td>917xx</td>
      <td>CA</td>
      <td>20.00</td>
      <td>0.0</td>
      <td>Feb-1996</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>5598.0</td>
      <td>21%</td>
      <td>37.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12231.8900</td>
      <td>12231.89</td>
      <td>10000.00</td>
      <td>2214.92</td>
      <td>16.97</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jan-2015</td>
      <td>357.48</td>
      <td>NaN</td>
      <td>Apr-2016</td>
      <td>604.0</td>
      <td>600.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>University Medical Group</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=1075358</td>
      <td>other</td>
      <td>Personal</td>
      <td>972xx</td>
      <td>OR</td>
      <td>17.94</td>
      <td>0.0</td>
      <td>Jan-1996</td>
      <td>695.0</td>
      <td>699.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>27783.0</td>
      <td>53.9%</td>
      <td>38.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4066.9082</td>
      <td>4066.91</td>
      <td>3000.00</td>
      <td>1066.91</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>Jan-2017</td>
      <td>67.30</td>
      <td>NaN</td>
      <td>Apr-2018</td>
      <td>684.0</td>
      <td>680.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
drop_list = sorted(list(missing_fractions[missing_fractions.percent_missing > 0.3].col_name))
print(drop_list)
print("\n\n Drop Features: ", len(drop_list))
```

    ['annual_inc_joint', 'deferral_term', 'dti_joint', 'hardship_amount', 'hardship_dpd', 'hardship_end_date', 'hardship_last_payment_amount', 'hardship_length', 'hardship_loan_status', 'hardship_payoff_balance_amount', 'hardship_reason', 'hardship_start_date', 'hardship_status', 'hardship_type', 'il_util', 'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record', 'mths_since_rcnt_il', 'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq', 'next_pymnt_d', 'orig_projected_additional_accrued_interest', 'payment_plan_start_date', 'revol_bal_joint', 'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 'sec_app_earliest_cr_line', 'sec_app_fico_range_high', 'sec_app_fico_range_low', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_num_rev_accts', 'sec_app_open_acc', 'sec_app_open_act_il', 'sec_app_revol_util', 'verification_status_joint']
    
    
     Drop Features:  37
    


```python
# Drop these features
loans.drop(labels=drop_list, axis=1, inplace=True)
```


```python
loans.shape
```




    (2925493, 104)



<a id="2.2"></a>
## 2.2 Only keep loan features known to potential investors


```python
print(sorted(loans.columns))
```

    ['acc_now_delinq', 'acc_open_past_24mths', 'addr_state', 'all_util', 'annual_inc', 'application_type', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag', 'delinq_2yrs', 'delinq_amnt', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'funded_amnt', 'funded_amnt_inv', 'grade', 'hardship_flag', 'home_ownership', 'id', 'initial_list_status', 'inq_fi', 'inq_last_12m', 'inq_last_6mths', 'installment', 'int_rate', 'issue_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d', 'loan_amnt', 'loan_status', 'max_bal_bc', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'open_acc', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m', 'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'policy_code', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'pymnt_plan', 'recoveries', 'revol_bal', 'revol_util', 'sub_grade', 'tax_liens', 'term', 'title', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_acc', 'total_bal_ex_mort', 'total_bal_il', 'total_bc_limit', 'total_cu_tl', 'total_il_high_credit_limit', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim', 'url', 'verification_status', 'zip_code']
    

For each of these features, we check the description in the Data Dictionary and only keep the features that would have been available to investors considering an investment in the loan. These include features in the loan application, and any features added by LendingClub when the loan listing was accepted, such as the loan grade and interest rate.

I'm using my best available knowledge to determine which loan features are known to potential investors. I am not an investor on LendingClub, so my knowledge of the LendingClub investment process is not exact. When in doubt, I err on the side of dropping the feature.


```python
keep_list = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status', 'zip_code']
print("Keep features: ", len(keep_list))
```

    Keep features:  31
    


```python
drop_list = [col for col in loans.columns if col not in keep_list]
print(drop_list)

print("\n\nNumber of features needed to be dropped: ",len(drop_list))
```

    ['funded_amnt', 'funded_amnt_inv', 'pymnt_plan', 'url', 'delinq_2yrs', 'inq_last_6mths', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'total_bal_il', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'hardship_flag', 'debt_settlement_flag']
    
    
    Number of features needed to be dropped:  73
    


```python
loans.drop(labels=drop_list, axis=1, inplace=True)
loans.shape
```




    (2925493, 31)



<a id="3"></a>
# 3. Data Analysis and Pre-processing

* Overlook on Data
* Pre-processing on each Variable

<a id="3.1"></a>
## 3.1 Overlook on Data


```python
print("Records: ", loans.shape[0], "\nFeatures: ", loans.shape[1])
print("\nInformation of Dataset\n-------")
loans.info()
```

    Records:  2925493 
    Features:  31
    
    Information of Dataset
    -------
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2925493 entries, 0 to 105450
    Data columns (total 31 columns):
     #   Column                Dtype  
    ---  ------                -----  
     0   id                    object 
     1   loan_amnt             float64
     2   term                  object 
     3   int_rate              object 
     4   installment           float64
     5   grade                 object 
     6   sub_grade             object 
     7   emp_title             object 
     8   emp_length            object 
     9   home_ownership        object 
     10  annual_inc            float64
     11  verification_status   object 
     12  issue_d               object 
     13  loan_status           object 
     14  purpose               object 
     15  title                 object 
     16  zip_code              object 
     17  addr_state            object 
     18  dti                   float64
     19  earliest_cr_line      object 
     20  fico_range_low        float64
     21  fico_range_high       float64
     22  open_acc              float64
     23  pub_rec               float64
     24  revol_bal             float64
     25  revol_util            object 
     26  total_acc             float64
     27  initial_list_status   object 
     28  application_type      object 
     29  mort_acc              float64
     30  pub_rec_bankruptcies  float64
    dtypes: float64(12), object(19)
    memory usage: 714.2+ MB
    


```python
loans.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>earliest_cr_line</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>5000.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Computer</td>
      <td>860xx</td>
      <td>AZ</td>
      <td>27.65</td>
      <td>Jan-1985</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>13648.0</td>
      <td>83.7%</td>
      <td>9.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>car</td>
      <td>bike</td>
      <td>309xx</td>
      <td>GA</td>
      <td>1.00</td>
      <td>Apr-1999</td>
      <td>740.0</td>
      <td>744.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1687.0</td>
      <td>9.4%</td>
      <td>4.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>small_business</td>
      <td>real estate business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>8.72</td>
      <td>Nov-2001</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2956.0</td>
      <td>98.5%</td>
      <td>10.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>AIR RESOURCES BOARD</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>49200.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>other</td>
      <td>personel</td>
      <td>917xx</td>
      <td>CA</td>
      <td>20.00</td>
      <td>Feb-1996</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>5598.0</td>
      <td>21%</td>
      <td>37.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>University Medical Group</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>80000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>other</td>
      <td>Personal</td>
      <td>972xx</td>
      <td>OR</td>
      <td>17.94</td>
      <td>Jan-1996</td>
      <td>695.0</td>
      <td>699.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>27783.0</td>
      <td>53.9%</td>
      <td>38.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1075269</td>
      <td>5000.0</td>
      <td>36 months</td>
      <td>7.90%</td>
      <td>156.46</td>
      <td>A</td>
      <td>A4</td>
      <td>Veolia Transportaton</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>36000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>wedding</td>
      <td>My wedding loan I promise to pay back</td>
      <td>852xx</td>
      <td>AZ</td>
      <td>11.20</td>
      <td>Nov-2004</td>
      <td>730.0</td>
      <td>734.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>7963.0</td>
      <td>28.3%</td>
      <td>12.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1069639</td>
      <td>7000.0</td>
      <td>60 months</td>
      <td>15.96%</td>
      <td>170.08</td>
      <td>C</td>
      <td>C5</td>
      <td>Southern Star Photography</td>
      <td>8 years</td>
      <td>RENT</td>
      <td>47004.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Loan</td>
      <td>280xx</td>
      <td>NC</td>
      <td>23.51</td>
      <td>Jul-2005</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>17726.0</td>
      <td>85.6%</td>
      <td>11.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1072053</td>
      <td>3000.0</td>
      <td>36 months</td>
      <td>18.64%</td>
      <td>109.43</td>
      <td>E</td>
      <td>E1</td>
      <td>MKC Accounting</td>
      <td>9 years</td>
      <td>RENT</td>
      <td>48000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>car</td>
      <td>Car Downpayment</td>
      <td>900xx</td>
      <td>CA</td>
      <td>5.35</td>
      <td>Jan-2007</td>
      <td>660.0</td>
      <td>664.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>8221.0</td>
      <td>87.5%</td>
      <td>4.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1071795</td>
      <td>5600.0</td>
      <td>60 months</td>
      <td>21.28%</td>
      <td>152.39</td>
      <td>F</td>
      <td>F2</td>
      <td>NaN</td>
      <td>4 years</td>
      <td>OWN</td>
      <td>40000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>small_business</td>
      <td>Expand Business &amp; Buy Debt Portfolio</td>
      <td>958xx</td>
      <td>CA</td>
      <td>5.55</td>
      <td>Apr-2004</td>
      <td>675.0</td>
      <td>679.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>5210.0</td>
      <td>32.6%</td>
      <td>13.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1071570</td>
      <td>5375.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>121.45</td>
      <td>B</td>
      <td>B5</td>
      <td>Starbucks</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>15000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>other</td>
      <td>Building my credit history.</td>
      <td>774xx</td>
      <td>TX</td>
      <td>18.08</td>
      <td>Sep-2004</td>
      <td>725.0</td>
      <td>729.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>9279.0</td>
      <td>36.5%</td>
      <td>3.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1070078</td>
      <td>6500.0</td>
      <td>60 months</td>
      <td>14.65%</td>
      <td>153.45</td>
      <td>C</td>
      <td>C3</td>
      <td>Southwest Rural metro</td>
      <td>5 years</td>
      <td>OWN</td>
      <td>72000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>High intrest Consolidation</td>
      <td>853xx</td>
      <td>AZ</td>
      <td>16.12</td>
      <td>Jan-1998</td>
      <td>695.0</td>
      <td>699.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>4032.0</td>
      <td>20.6%</td>
      <td>23.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1069908</td>
      <td>12000.0</td>
      <td>36 months</td>
      <td>12.69%</td>
      <td>402.54</td>
      <td>B</td>
      <td>B5</td>
      <td>UCLA</td>
      <td>10+ years</td>
      <td>OWN</td>
      <td>75000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Consolidation</td>
      <td>913xx</td>
      <td>CA</td>
      <td>10.78</td>
      <td>Oct-1989</td>
      <td>675.0</td>
      <td>679.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>23336.0</td>
      <td>67.1%</td>
      <td>34.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1064687</td>
      <td>9000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>305.38</td>
      <td>C</td>
      <td>C1</td>
      <td>Va. Dept of Conservation/Recreation</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>debt_consolidation</td>
      <td>freedom</td>
      <td>245xx</td>
      <td>VA</td>
      <td>10.08</td>
      <td>Apr-2004</td>
      <td>710.0</td>
      <td>714.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>10452.0</td>
      <td>91.7%</td>
      <td>9.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1069866</td>
      <td>3000.0</td>
      <td>36 months</td>
      <td>9.91%</td>
      <td>96.68</td>
      <td>B</td>
      <td>B1</td>
      <td>Target</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>15000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>citicard fund</td>
      <td>606xx</td>
      <td>IL</td>
      <td>12.56</td>
      <td>Jul-2003</td>
      <td>705.0</td>
      <td>709.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>7323.0</td>
      <td>43.1%</td>
      <td>11.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1069057</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>325.74</td>
      <td>B</td>
      <td>B2</td>
      <td>SFMTA</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>100000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>other</td>
      <td>Other Loan</td>
      <td>951xx</td>
      <td>CA</td>
      <td>7.06</td>
      <td>May-1991</td>
      <td>720.0</td>
      <td>724.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>11997.0</td>
      <td>55.5%</td>
      <td>29.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1069759</td>
      <td>1000.0</td>
      <td>36 months</td>
      <td>16.29%</td>
      <td>35.31</td>
      <td>D</td>
      <td>D1</td>
      <td>Internal revenue Service</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>28000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation Loan</td>
      <td>641xx</td>
      <td>MO</td>
      <td>20.31</td>
      <td>Sep-2007</td>
      <td>665.0</td>
      <td>669.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>6524.0</td>
      <td>81.5%</td>
      <td>23.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1065775</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>15.27%</td>
      <td>347.98</td>
      <td>C</td>
      <td>C4</td>
      <td>Chin's Restaurant</td>
      <td>4 years</td>
      <td>RENT</td>
      <td>42000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>home_improvement</td>
      <td>Home</td>
      <td>921xx</td>
      <td>CA</td>
      <td>18.60</td>
      <td>Oct-1998</td>
      <td>670.0</td>
      <td>674.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>24043.0</td>
      <td>70.2%</td>
      <td>28.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1069971</td>
      <td>3600.0</td>
      <td>36 months</td>
      <td>6.03%</td>
      <td>109.57</td>
      <td>A</td>
      <td>A1</td>
      <td>Duracell</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>110000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>major_purchase</td>
      <td>Holiday</td>
      <td>067xx</td>
      <td>CT</td>
      <td>10.52</td>
      <td>Aug-1993</td>
      <td>760.0</td>
      <td>764.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>22836.0</td>
      <td>16%</td>
      <td>42.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1062474</td>
      <td>6000.0</td>
      <td>36 months</td>
      <td>11.71%</td>
      <td>198.46</td>
      <td>B</td>
      <td>B3</td>
      <td>Connection Inspection</td>
      <td>1 year</td>
      <td>MORTGAGE</td>
      <td>84000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>medical</td>
      <td>Medical</td>
      <td>890xx</td>
      <td>UT</td>
      <td>18.44</td>
      <td>Oct-2003</td>
      <td>685.0</td>
      <td>689.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.73%</td>
      <td>14.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1069742</td>
      <td>9200.0</td>
      <td>36 months</td>
      <td>6.03%</td>
      <td>280.01</td>
      <td>A</td>
      <td>A1</td>
      <td>Network Interpreting Service</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>77385.19</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>lowerratemeanseasiertogetoutofdebt!</td>
      <td>921xx</td>
      <td>CA</td>
      <td>9.86</td>
      <td>Jan-2001</td>
      <td>755.0</td>
      <td>759.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>7314.0</td>
      <td>23.1%</td>
      <td>28.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1069740</td>
      <td>20250.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>484.63</td>
      <td>C</td>
      <td>C4</td>
      <td>Archdiocese of Galveston Houston</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>43370.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Freedom From Credit Cards</td>
      <td>770xx</td>
      <td>TX</td>
      <td>26.53</td>
      <td>Nov-1997</td>
      <td>725.0</td>
      <td>729.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>17813.0</td>
      <td>85.6%</td>
      <td>22.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1039153</td>
      <td>21000.0</td>
      <td>36 months</td>
      <td>12.42%</td>
      <td>701.73</td>
      <td>B</td>
      <td>B4</td>
      <td>Osram Sylvania</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>105000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>debt_consolidation</td>
      <td>Debt Cleanup</td>
      <td>335xx</td>
      <td>FL</td>
      <td>13.22</td>
      <td>Feb-1983</td>
      <td>705.0</td>
      <td>709.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>32135.0</td>
      <td>90.3%</td>
      <td>38.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1069710</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.71%</td>
      <td>330.76</td>
      <td>B</td>
      <td>B3</td>
      <td>Value Air</td>
      <td>10+ years</td>
      <td>OWN</td>
      <td>50000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit Card Loan</td>
      <td>799xx</td>
      <td>TX</td>
      <td>11.18</td>
      <td>Jul-1985</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>10056.0</td>
      <td>82.4%</td>
      <td>21.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1069700</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.71%</td>
      <td>330.76</td>
      <td>B</td>
      <td>B3</td>
      <td>Wells Fargo Bank</td>
      <td>5 years</td>
      <td>RENT</td>
      <td>50000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation</td>
      <td>917xx</td>
      <td>CA</td>
      <td>16.01</td>
      <td>Apr-2003</td>
      <td>710.0</td>
      <td>714.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>17800.0</td>
      <td>91.8%</td>
      <td>17.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1069559</td>
      <td>6000.0</td>
      <td>36 months</td>
      <td>11.71%</td>
      <td>198.46</td>
      <td>B</td>
      <td>B3</td>
      <td>bmg-educational</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>76000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>major_purchase</td>
      <td>cash</td>
      <td>900xx</td>
      <td>CA</td>
      <td>2.40</td>
      <td>Jun-2001</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>5963.0</td>
      <td>29.7%</td>
      <td>7.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1069697</td>
      <td>15000.0</td>
      <td>36 months</td>
      <td>9.91%</td>
      <td>483.38</td>
      <td>B</td>
      <td>B1</td>
      <td>Winfield Pathology Consultants</td>
      <td>2 years</td>
      <td>MORTGAGE</td>
      <td>92000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>No more credit card debt!</td>
      <td>605xx</td>
      <td>IL</td>
      <td>29.44</td>
      <td>Feb-2002</td>
      <td>730.0</td>
      <td>734.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>13707.0</td>
      <td>93.9%</td>
      <td>31.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1069800</td>
      <td>15000.0</td>
      <td>36 months</td>
      <td>14.27%</td>
      <td>514.64</td>
      <td>C</td>
      <td>C2</td>
      <td>nyc transit</td>
      <td>9 years</td>
      <td>RENT</td>
      <td>60000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>debt_consolidation</td>
      <td>consolidation</td>
      <td>103xx</td>
      <td>NY</td>
      <td>15.22</td>
      <td>Oct-2003</td>
      <td>695.0</td>
      <td>699.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>5872.0</td>
      <td>57.6%</td>
      <td>11.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1069657</td>
      <td>5000.0</td>
      <td>60 months</td>
      <td>16.77%</td>
      <td>123.65</td>
      <td>D</td>
      <td>D2</td>
      <td>Frito Lay</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>50004.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>other</td>
      <td>Other</td>
      <td>150xx</td>
      <td>PA</td>
      <td>13.97</td>
      <td>Oct-2003</td>
      <td>675.0</td>
      <td>679.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>4345.0</td>
      <td>59.5%</td>
      <td>22.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1069799</td>
      <td>4000.0</td>
      <td>36 months</td>
      <td>11.71%</td>
      <td>132.31</td>
      <td>B</td>
      <td>B3</td>
      <td>Shands Hospital at the University of Fl</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>106000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation</td>
      <td>326xx</td>
      <td>FL</td>
      <td>5.63</td>
      <td>Aug-1984</td>
      <td>680.0</td>
      <td>684.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>6110.0</td>
      <td>37.7%</td>
      <td>44.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1047704</td>
      <td>8500.0</td>
      <td>36 months</td>
      <td>11.71%</td>
      <td>281.15</td>
      <td>B</td>
      <td>B3</td>
      <td>Oakridge homes</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>25000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit card repayment loan</td>
      <td>564xx</td>
      <td>MN</td>
      <td>12.19</td>
      <td>Nov-2006</td>
      <td>695.0</td>
      <td>699.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>6441.0</td>
      <td>59.1%</td>
      <td>12.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1032111</td>
      <td>4375.0</td>
      <td>36 months</td>
      <td>7.51%</td>
      <td>136.11</td>
      <td>A</td>
      <td>A3</td>
      <td>NaN</td>
      <td>7 years</td>
      <td>MORTGAGE</td>
      <td>17108.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation</td>
      <td>141xx</td>
      <td>NY</td>
      <td>20.34</td>
      <td>Dec-1987</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>11210.0</td>
      <td>86.9%</td>
      <td>12.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1069539</td>
      <td>31825.0</td>
      <td>36 months</td>
      <td>7.90%</td>
      <td>995.82</td>
      <td>A</td>
      <td>A4</td>
      <td>Audubon Mutual Housing Corporation</td>
      <td>5 years</td>
      <td>MORTGAGE</td>
      <td>75000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation Loan</td>
      <td>080xx</td>
      <td>NJ</td>
      <td>14.03</td>
      <td>Nov-1981</td>
      <td>760.0</td>
      <td>764.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>23151.0</td>
      <td>27.4%</td>
      <td>26.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1065420</td>
      <td>10000.0</td>
      <td>60 months</td>
      <td>15.96%</td>
      <td>242.97</td>
      <td>C</td>
      <td>C5</td>
      <td>US Legal Support</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>29120.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Brown Eyes Loan</td>
      <td>330xx</td>
      <td>FL</td>
      <td>22.83</td>
      <td>Feb-1997</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>16158.0</td>
      <td>63.6%</td>
      <td>31.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1069591</td>
      <td>5000.0</td>
      <td>36 months</td>
      <td>8.90%</td>
      <td>158.77</td>
      <td>A</td>
      <td>A5</td>
      <td>Good Samaritan Society</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>24044.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Consolidation</td>
      <td>974xx</td>
      <td>OR</td>
      <td>11.93</td>
      <td>Apr-2005</td>
      <td>700.0</td>
      <td>704.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>2224.0</td>
      <td>29.3%</td>
      <td>16.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1069530</td>
      <td>7000.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>245.97</td>
      <td>C</td>
      <td>C5</td>
      <td>GREG BARRETT DRYWALL</td>
      <td>7 years</td>
      <td>RENT</td>
      <td>34000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit Card Loan</td>
      <td>934xx</td>
      <td>CA</td>
      <td>6.35</td>
      <td>Oct-2007</td>
      <td>675.0</td>
      <td>679.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>6113.0</td>
      <td>60.5%</td>
      <td>6.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1069522</td>
      <td>12400.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>403.91</td>
      <td>B</td>
      <td>B2</td>
      <td>Sharp Lawn Inc.</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>41000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>December 2011 Credit Card Loan</td>
      <td>405xx</td>
      <td>KY</td>
      <td>11.80</td>
      <td>Jul-2005</td>
      <td>720.0</td>
      <td>724.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>11095.0</td>
      <td>57.2%</td>
      <td>12.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1069361</td>
      <td>10800.0</td>
      <td>36 months</td>
      <td>9.91%</td>
      <td>348.03</td>
      <td>B</td>
      <td>B1</td>
      <td>Department of Justice</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>55596.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>moving</td>
      <td>Moving to better job; help a veteran!</td>
      <td>946xx</td>
      <td>CA</td>
      <td>5.50</td>
      <td>Dec-2000</td>
      <td>705.0</td>
      <td>709.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>2535.0</td>
      <td>13.3%</td>
      <td>18.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1069357</td>
      <td>15000.0</td>
      <td>36 months</td>
      <td>7.90%</td>
      <td>469.36</td>
      <td>A</td>
      <td>A4</td>
      <td>Gateway Hospice</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>45000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation Loan</td>
      <td>445xx</td>
      <td>OH</td>
      <td>8.48</td>
      <td>Nov-2004</td>
      <td>730.0</td>
      <td>734.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>6704.0</td>
      <td>50.4%</td>
      <td>27.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1069356</td>
      <td>6000.0</td>
      <td>36 months</td>
      <td>12.42%</td>
      <td>200.50</td>
      <td>B</td>
      <td>B4</td>
      <td>Cox Communications</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>36852.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Credit Card Consolidation</td>
      <td>850xx</td>
      <td>AZ</td>
      <td>10.62</td>
      <td>Apr-2007</td>
      <td>695.0</td>
      <td>699.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>7310.0</td>
      <td>66.5%</td>
      <td>12.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1069346</td>
      <td>12500.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>282.44</td>
      <td>B</td>
      <td>B5</td>
      <td>United States Infrastructure Corporation</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>27000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Credit card debt</td>
      <td>604xx</td>
      <td>IL</td>
      <td>16.04</td>
      <td>Dec-2001</td>
      <td>735.0</td>
      <td>739.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>10143.0</td>
      <td>50%</td>
      <td>24.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1067573</td>
      <td>9600.0</td>
      <td>36 months</td>
      <td>7.51%</td>
      <td>298.67</td>
      <td>A</td>
      <td>A3</td>
      <td>Nucor Building Systems</td>
      <td>4 years</td>
      <td>RENT</td>
      <td>68004.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>other</td>
      <td>Other Loan</td>
      <td>292xx</td>
      <td>SC</td>
      <td>6.63</td>
      <td>Jan-2003</td>
      <td>760.0</td>
      <td>764.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>2896.0</td>
      <td>18.1%</td>
      <td>11.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1069506</td>
      <td>12000.0</td>
      <td>36 months</td>
      <td>7.90%</td>
      <td>375.49</td>
      <td>A</td>
      <td>A4</td>
      <td>John Wiley Jr.</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>62300.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation</td>
      <td>088xx</td>
      <td>NJ</td>
      <td>16.70</td>
      <td>Mar-1994</td>
      <td>725.0</td>
      <td>729.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>8601.0</td>
      <td>21%</td>
      <td>25.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1069314</td>
      <td>3000.0</td>
      <td>36 months</td>
      <td>18.25%</td>
      <td>108.84</td>
      <td>D</td>
      <td>D5</td>
      <td>NaN</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>65000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>other</td>
      <td>Year End Cleanup</td>
      <td>180xx</td>
      <td>PA</td>
      <td>17.39</td>
      <td>Sep-1998</td>
      <td>670.0</td>
      <td>674.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>43936.0</td>
      <td>98.1%</td>
      <td>22.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1060578</td>
      <td>4400.0</td>
      <td>36 months</td>
      <td>16.77%</td>
      <td>156.37</td>
      <td>D</td>
      <td>D2</td>
      <td>citizens bank</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>55000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>debt</td>
      <td>029xx</td>
      <td>RI</td>
      <td>20.01</td>
      <td>Jun-2004</td>
      <td>690.0</td>
      <td>694.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>25237.0</td>
      <td>99%</td>
      <td>11.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1069469</td>
      <td>6000.0</td>
      <td>36 months</td>
      <td>6.03%</td>
      <td>182.62</td>
      <td>A</td>
      <td>A1</td>
      <td>Stewart Enterprises, Inc.</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>45600.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt-Consolidation</td>
      <td>700xx</td>
      <td>LA</td>
      <td>5.34</td>
      <td>Nov-1995</td>
      <td>790.0</td>
      <td>794.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>3378.0</td>
      <td>32.5%</td>
      <td>28.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1051117</td>
      <td>14000.0</td>
      <td>36 months</td>
      <td>9.91%</td>
      <td>451.15</td>
      <td>B</td>
      <td>B1</td>
      <td>golfers warehouse</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>80000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>small_business</td>
      <td>Small Business Loan</td>
      <td>010xx</td>
      <td>MA</td>
      <td>10.00</td>
      <td>Jul-1999</td>
      <td>725.0</td>
      <td>729.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>2219.0</td>
      <td>13.4%</td>
      <td>40.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1069465</td>
      <td>5000.0</td>
      <td>36 months</td>
      <td>8.90%</td>
      <td>158.77</td>
      <td>A</td>
      <td>A5</td>
      <td>STERIS Corporation</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>100000.00</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>debt_consolidation</td>
      <td>Debt Consolidation for a better life</td>
      <td>441xx</td>
      <td>OH</td>
      <td>16.33</td>
      <td>Jun-1995</td>
      <td>705.0</td>
      <td>709.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>74351.0</td>
      <td>62.1%</td>
      <td>35.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1069283</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>325.74</td>
      <td>B</td>
      <td>B2</td>
      <td>Costco Wholesale Corp.</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>27000.00</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>other</td>
      <td>Other Loan</td>
      <td>104xx</td>
      <td>NY</td>
      <td>11.47</td>
      <td>Nov-2006</td>
      <td>700.0</td>
      <td>704.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>7814.0</td>
      <td>52.3%</td>
      <td>10.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1069287</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>6.03%</td>
      <td>304.36</td>
      <td>A</td>
      <td>A1</td>
      <td>Helicoil</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>60000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit Card Loan</td>
      <td>067xx</td>
      <td>CT</td>
      <td>12.74</td>
      <td>Sep-1992</td>
      <td>790.0</td>
      <td>794.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>14019.0</td>
      <td>19.5%</td>
      <td>18.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1069453</td>
      <td>11000.0</td>
      <td>36 months</td>
      <td>6.62%</td>
      <td>337.75</td>
      <td>A</td>
      <td>A2</td>
      <td>cognizant technology solutions</td>
      <td>5 years</td>
      <td>RENT</td>
      <td>70000.00</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt Consolitation</td>
      <td>061xx</td>
      <td>CT</td>
      <td>10.85</td>
      <td>Jan-2002</td>
      <td>750.0</td>
      <td>754.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>11638.0</td>
      <td>36.4%</td>
      <td>7.0</td>
      <td>f</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
loans.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>total_acc</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.9255e+06</td>
      <td>2.9255e+06</td>
      <td>2.9255e+06</td>
      <td>2.9224e+06</td>
      <td>2.9255e+06</td>
      <td>2.9255e+06</td>
      <td>2.9255e+06</td>
      <td>2.9255e+06</td>
      <td>2.9255e+06</td>
      <td>2.9255e+06</td>
      <td>2.8755e+06</td>
      <td>2.9241e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.5359e+04</td>
      <td>4.5239e+02</td>
      <td>7.9937e+04</td>
      <td>1.9296e+01</td>
      <td>7.0051e+02</td>
      <td>7.0451e+02</td>
      <td>1.1676e+01</td>
      <td>1.7628e-01</td>
      <td>1.6965e+04</td>
      <td>2.4018e+01</td>
      <td>1.5123e+00</td>
      <td>1.2268e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.4784e+03</td>
      <td>2.7295e+02</td>
      <td>1.1175e+05</td>
      <td>1.5720e+01</td>
      <td>3.3766e+01</td>
      <td>3.3767e+01</td>
      <td>5.7333e+00</td>
      <td>5.2438e-01</td>
      <td>2.2996e+04</td>
      <td>1.2076e+01</td>
      <td>1.8698e+00</td>
      <td>3.5314e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.0000e+02</td>
      <td>4.9300e+00</td>
      <td>0.0000e+00</td>
      <td>-1.0000e+00</td>
      <td>6.1000e+02</td>
      <td>6.1400e+02</td>
      <td>0.0000e+00</td>
      <td>0.0000e+00</td>
      <td>0.0000e+00</td>
      <td>1.0000e+00</td>
      <td>0.0000e+00</td>
      <td>0.0000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.0000e+03</td>
      <td>2.5350e+02</td>
      <td>4.7000e+04</td>
      <td>1.2080e+01</td>
      <td>6.7500e+02</td>
      <td>6.7900e+02</td>
      <td>8.0000e+00</td>
      <td>0.0000e+00</td>
      <td>5.9930e+03</td>
      <td>1.5000e+01</td>
      <td>0.0000e+00</td>
      <td>0.0000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.3000e+04</td>
      <td>3.8230e+02</td>
      <td>6.6000e+04</td>
      <td>1.8100e+01</td>
      <td>6.9500e+02</td>
      <td>6.9900e+02</td>
      <td>1.1000e+01</td>
      <td>0.0000e+00</td>
      <td>1.1493e+04</td>
      <td>2.2000e+01</td>
      <td>1.0000e+00</td>
      <td>0.0000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.0000e+04</td>
      <td>6.0392e+02</td>
      <td>9.5000e+04</td>
      <td>2.4880e+01</td>
      <td>7.2000e+02</td>
      <td>7.2400e+02</td>
      <td>1.5000e+01</td>
      <td>0.0000e+00</td>
      <td>2.0644e+04</td>
      <td>3.1000e+01</td>
      <td>2.0000e+00</td>
      <td>0.0000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.0000e+04</td>
      <td>1.7198e+03</td>
      <td>1.1000e+08</td>
      <td>9.9900e+02</td>
      <td>8.4500e+02</td>
      <td>8.5000e+02</td>
      <td>1.0400e+02</td>
      <td>8.6000e+01</td>
      <td>2.9048e+06</td>
      <td>1.7600e+02</td>
      <td>9.4000e+01</td>
      <td>1.2000e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Target Variable
loans['loan_status'].value_counts(dropna=False)
```




    Fully Paid                                             1497783
    Current                                                1031016
    Charged Off                                             362548
    Late (31-120 days)                                       16154
    In Grace Period                                          10028
    Late (16-30 days)                                         2719
    Issued                                                    2062
    Does not meet the credit policy. Status:Fully Paid        1988
    Does not meet the credit policy. Status:Charged Off        761
    Default                                                    433
    NaN                                                          1
    Name: loan_status, dtype: int64



We're going to try to learn differences in the features between completed loans that have been fully paid or charged off. We won't consider loans that are current, don't meet the credit policy, defaulted, or have a missing status. So we only keep the loans with status "Fully Paid" or "Charged Off."


```python
loans = loans.loc[loans['loan_status'].isin(['Fully Paid', 'Charged Off'])]
```


```python
# Categories in percentage
loans['loan_status'].value_counts(normalize=True, dropna=False)*100
```




    Fully Paid     80.5116
    Charged Off    19.4884
    Name: loan_status, dtype: float64



* This is an imbalance problem, because we have a lot more entries of people that fully paid their loans then people that did not pay back.
* We can expect to probably do very well in terms of accuracy but our precision and recall are going to be the true metrics that we will have to evaluate our model based off of.

* In the loan amount distribution we can see spikes in even ten thousend dollar, so this is indicating that there are certain amounts that are basically standard loans.

<a id="3.2"></a>
## 3.2 Pre-processing on each Variable

We'll inspect each feature individually, and do the following:

1. Drop the feature if it is not useful for predicting the loan status.
2. View summary statistics and visualize the data, plotting against the loan status.
3. Modify the feature to make it useful for modeling, if necessary.

We define a function for plotting a variable and comparing with the loan status:


```python
def plot_var(col_name, full_name, continuous):
    """
    Visualize a variable with and without faceting on the loan status.
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    # Plot without loan status
    if continuous:
        sns.distplot(loans.loc[loans[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(loans[col_name], order=sorted(loans[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(full_name)
    ax1.set_ylabel('Count')
    ax1.set_title(full_name)

    # Plot with loan status
    if continuous:
        sns.boxplot(x=col_name, y='loan_status', data=loans, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(full_name + ' by Loan Status')
    else:
        charge_off_rates = loans.groupby(col_name)['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
        sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color='#5975A4', saturation=1, ax=ax2)
        ax2.set_ylabel('Fraction of Loans Charged-off')
        ax2.set_title('Charge-off Rate by ' + full_name)
    ax2.set_xlabel(full_name)
    
    plt.tight_layout()
```

Print the remaining features for future reference:


```python
print(list(loans.columns))
```

    ['id', 'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'purpose', 'title', 'zip_code', 'addr_state', 'dti', 'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'application_type', 'mort_acc', 'pub_rec_bankruptcies']
    

### 3.2.1 id

Data Dictionary: "A unique [LendingClub] assigned ID for the loan listing."

Are all the IDs unique?


```python
loans['id'].describe()
```




    count       1860331
    unique      1860331
    top       113846917
    freq              1
    Name: id, dtype: object



Yes, they are all unique. The ID is not useful for modeling, either as a categorical variable (there are too many distinct values) or as a numerical variable (the IDs vary wildly in magnitude, likely without any significance), so we drop this variable.


```python
loans.drop('id', axis=1, inplace=True)
```

### 3.2.2 loan_amnt

Data Dictionary: "The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value."


```python
loans['loan_amnt'].describe()
```




    count    1.8603e+06
    mean     1.4588e+04
    std      8.9705e+03
    min      5.0000e+02
    25%      7.9500e+03
    50%      1.2000e+04
    75%      2.0000e+04
    max      4.0000e+04
    Name: loan_amnt, dtype: float64



Loan amounts range from \$500 to \$40,000, with a median of \$12,000.


```python
plot_var('loan_amnt', 'Loan Amount', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_53_1.png)
    


Charged-off loans tend to have higher loan amounts. Let's compare the summary statistics by loan status:


```python
loans.groupby('loan_status')['loan_amnt'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>15899.4960</td>
      <td>9083.2622</td>
      <td>900.0</td>
      <td>9250.0</td>
      <td>14700.0</td>
      <td>21000.0</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>14270.8825</td>
      <td>8914.0119</td>
      <td>500.0</td>
      <td>7350.0</td>
      <td>12000.0</td>
      <td>20000.0</td>
      <td>40000.0</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.3 term

Data Dictionary: "The number of payments on the loan. Values are in months and can be either 36 or 60."


```python
loans['term'].value_counts(dropna=False)
```




     36 months    1393299
     60 months     467032
    Name: term, dtype: int64



Convert `term` to integers.


```python
loans['term'] = loans['term'].apply(lambda s: np.int8(s.split()[0]))
loans['term'].value_counts(normalize=True)
```




    36    0.749
    60    0.251
    Name: term, dtype: float64



Compare the charge-off rate by loan period:


```python
loans.groupby('term')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
```




    term
    36    0.1562
    60    0.3102
    Name: loan_status, dtype: float64



About 75% of the completed loans have three-year periods, and the rest have five-year periods. Loans with five-year periods are more than twice as likely to charge-off as loans with three-year periods.

### 3.2.4 int_rate

Data Dictionary: "Interest Rate on the loan."


```python
# Let's remove % percent sign from `int_rate` and convert it into `float`
loans.int_rate = loans.int_rate.map(lambda x: x.replace('%','')).astype(np.float64)
```


```python
loans['int_rate'].describe()
```




    count    1.8603e+06
    mean     1.3170e+01
    std      4.8283e+00
    min      5.3100e+00
    25%      9.5600e+00
    50%      1.2690e+01
    75%      1.5990e+01
    max      3.0990e+01
    Name: int_rate, dtype: float64



Interest rates range from 5.32% to 31% (!) with a median of 13.2%.


```python
plot_var('int_rate', 'Interest Rate', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_69_1.png)
    


Charged-off loans tend to have much higher interest rates. Let's compare the summary statistics by loan status:


```python
loans.groupby('loan_status')['int_rate'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>15.6559</td>
      <td>5.0217</td>
      <td>5.31</td>
      <td>12.05</td>
      <td>14.99</td>
      <td>18.55</td>
      <td>30.99</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>12.5684</td>
      <td>4.5819</td>
      <td>5.31</td>
      <td>9.16</td>
      <td>11.99</td>
      <td>15.05</td>
      <td>30.99</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.5 installment
<a id="3.2.5"></a>

Data Dictionary: "The monthly payment owed by the borrower if the loan originates."


```python
loans['installment'].describe()
```




    count    1.8603e+06
    mean     4.4026e+02
    std      2.6691e+02
    min      4.9300e+00
    25%      2.4699e+02
    50%      3.7363e+02
    75%      5.8496e+02
    max      1.7198e+03
    Name: installment, dtype: float64



Installments range from \$4.93 to \$1,714, with a median of \$377.


```python
plot_var('installment', 'Installment', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_76_1.png)
    


Charged-off loans tend to have higher installments. Let's compare the summary statistics by loan status:


```python
loans.groupby('loan_status')['installment'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>472.7604</td>
      <td>268.4462</td>
      <td>21.62</td>
      <td>277.41</td>
      <td>407.26</td>
      <td>620.750</td>
      <td>1717.63</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>432.3879</td>
      <td>265.9396</td>
      <td>4.93</td>
      <td>238.17</td>
      <td>366.37</td>
      <td>575.105</td>
      <td>1719.83</td>
    </tr>
  </tbody>
</table>
</div>



Loans that charge off have \$30 higher installments on average.

### 3.2.6 grade, sub_grade

Data Dictionary for `grade`: "LendingClub assigned loan grade."

Data Dictionary for `sub_grade`: "LendingClub assigned loan subgrade."

What are the possible values of `grade` and `sub_grade`?


```python
print(sorted(loans['grade'].unique()))
```

    ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    


```python
print(sorted(loans['sub_grade'].unique()))
```

    ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5']
    

The grade is implied by the subgrade, so let's drop the grade column.


```python
loans.drop('grade', axis=1, inplace=True)
```


```python
plot_var('sub_grade', 'Subgrade', continuous=False)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    


    
![png](output_86_1.png)
    


There's a clear trend of higher probability of charge-off as the subgrade worsens.

### 3.2.7 emp_title

Data Dictionary: "The job title supplied by the Borrower when applying for the loan."


```python
loans['emp_title'].describe()
```




    count     1728877
    unique     456676
    top       Teacher
    freq        30748
    Name: emp_title, dtype: object



There are too many different job titles for this feature to be useful, so we drop it.


```python
loans.drop(labels='emp_title', axis=1, inplace=True)
```

### 3.2.8 emp_length

Data Dictionary: "Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years." The actual data does not match this description:


```python
loans['emp_length'].value_counts(dropna=False).sort_index()
```




    1 year       122924
    10+ years    611330
    2 years      168516
    3 years      149271
    4 years      112211
    5 years      115872
    6 years       85354
    7 years       78141
    8 years       78646
    9 years       67170
    < 1 year     153700
    NaN          117196
    Name: emp_length, dtype: int64



Note there are 117126 loans without data on the length of employment.

Convert `emp_length` to integers:


```python
loans['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
```


```python
loans['emp_length'].replace('< 1 year', '0 years', inplace=True)
```


```python
def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
```


```python
loans['emp_length'] = loans['emp_length'].apply(emp_length_to_int)
```


```python
loans['emp_length'].value_counts(dropna=False).sort_index()
```




    0.0     153700
    1.0     122924
    2.0     168516
    3.0     149271
    4.0     112211
    5.0     115872
    6.0      85354
    7.0      78141
    8.0      78646
    9.0      67170
    10.0    611330
    NaN     117196
    Name: emp_length, dtype: int64




```python
plot_var('emp_length', 'Employment Length', continuous=False)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    


    
![png](output_102_1.png)
    



```python
loans.drop('emp_length', axis=1, inplace=True)
```

Loan status does not appear to vary much with employment length on average, except for a small drop in charge-offs for borrowers with over 10 years of employment.

### 3.2.9 home_ownership

Data Dictionary: "The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER."


```python
loans['home_ownership'].value_counts(dropna=False)
```




    MORTGAGE    918028
    RENT        735358
    OWN         205578
    ANY           1174
    OTHER          144
    NONE            49
    Name: home_ownership, dtype: int64



Replace the values `ANY` and `NONE` with `OTHER`:


```python
loans['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
```


```python
loans['home_ownership'].value_counts(dropna=False)
```




    MORTGAGE    918028
    RENT        735358
    OWN         205578
    OTHER         1367
    Name: home_ownership, dtype: int64




```python
plot_var('home_ownership', 'Home Ownership', continuous=False)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    


    
![png](output_111_1.png)
    


There appear to be large differences in charge-off rates by home ownership status. Renters and homeowners have a higher probability of charge-off. Let's compare the charge-off rates:


```python
loans.groupby('home_ownership')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
```




    home_ownership
    MORTGAGE    0.1683
    OTHER       0.1997
    OWN         0.2011
    RENT        0.2263
    Name: loan_status, dtype: float64



### 3.2.10 annual_inc

Data Dictionary: "The self-reported annual income provided by the borrower during registration."


```python
loans['annual_inc'].describe()
```




    count    1.8603e+06
    mean     7.7370e+04
    std      1.1782e+05
    min      0.0000e+00
    25%      4.6000e+04
    50%      6.5000e+04
    75%      9.2000e+04
    max      1.1000e+08
    Name: annual_inc, dtype: float64



Annual income ranges from \$0 to \$9,550,000, with a median of \$65,000. Because of the large range of incomes, we should take a log transform of the annual income variable.


```python
loans['log_annual_inc'] = loans['annual_inc'].apply(lambda x: np.log10(x+1))
```


```python
loans.drop('annual_inc', axis=1, inplace=True)
```


```python
loans['log_annual_inc'].describe()
```




    count    1.8603e+06
    mean     4.8147e+00
    std      2.6617e-01
    min      0.0000e+00
    25%      4.6628e+00
    50%      4.8129e+00
    75%      4.9638e+00
    max      8.0414e+00
    Name: log_annual_inc, dtype: float64




```python
plot_var('log_annual_inc', 'Log Annual Income', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_121_1.png)
    


It appears that individuals with higher income are more likely to pay off their loans. Let's compare the summary statistics by loan status:


```python
loans.groupby('loan_status')['log_annual_inc'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>4.7852</td>
      <td>0.2630</td>
      <td>0.0</td>
      <td>4.6435</td>
      <td>4.7782</td>
      <td>4.9294</td>
      <td>8.0414</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>4.8219</td>
      <td>0.2664</td>
      <td>0.0</td>
      <td>4.6721</td>
      <td>4.8129</td>
      <td>4.9777</td>
      <td>7.7853</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.11 verification_status

Data Dictionary: "Indicates if income was verified by [Lending Club], not verified, or if the income source was verified."


```python
plot_var('verification_status', 'Verification Status', continuous=False)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    


    
![png](output_126_1.png)
    


### 3.2.12 issue_d

Data Dictionary: "The month which the loan was funded."

Because we're only using variables available to investors before the loan was funded, `issue_d` will not be included in the final model. We're keeping it for now just to perform the train/test split later, then we'll drop it.

### 3.2.13 purpose

Data Dictionary: "A category provided by the borrower for the loan request."


```python
loans['purpose'].value_counts()
```




    debt_consolidation    1063459
    credit_card            412844
    home_improvement       123861
    other                  113589
    major_purchase          41750
    medical                 22578
    small_business          20515
    car                     20215
    vacation                13333
    moving                  13203
    house                   11099
    wedding                  2298
    renewable_energy         1260
    educational               327
    Name: purpose, dtype: int64



Calculate the charge-off rates by purpose:


```python
loans.groupby('purpose')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off'].sort_values()
```




    purpose
    wedding               0.1218
    car                   0.1412
    credit_card           0.1685
    educational           0.1713
    home_improvement      0.1722
    vacation              0.1780
    major_purchase        0.1847
    other                 0.2002
    medical               0.2040
    debt_consolidation    0.2064
    renewable_energy      0.2167
    moving                0.2209
    house                 0.2217
    small_business        0.2858
    Name: loan_status, dtype: float64



Notice that only 12% of completed loans for weddings have charged-off, but 30% of completed small business loans have charged-off.

### 3.2.14 title

Data Dictionary: "The loan title provided by the borrower."


```python
loans['title'].describe()
```




    count                1838925
    unique                 61685
    top       Debt consolidation
    freq                  941169
    Name: title, dtype: object



View the top 10 loan titles, and their frequencies:


```python
loans['title'].value_counts().head(10)
```




    Debt consolidation         941169
    Credit card refinancing    366391
    Home improvement           111152
    Other                      102322
    Major purchase              36294
    Medical expenses            20548
    Business                    16806
    Car financing               16802
    Debt Consolidation          15695
    Vacation                    12273
    Name: title, dtype: int64



There are 60,298 different titles in the dataset, and based on the top 10 titles, the `purpose` variable appears to already contain this information. So we drop the `title` variable.


```python
loans.drop('title', axis=1, inplace=True)
```

### 3.2.15 zip_code, addr_state

Data Dictionary for `zip_code`: "The first 3 numbers of the zip code provided by the borrower in the loan application."

Data Dictionary for `addr_state`: "The state provided by the borrower in the loan application."


```python
loans['zip_code'].sample(5)
```




    66567     440xx
    68809     016xx
    110247    088xx
    28360     750xx
    30821     063xx
    Name: zip_code, dtype: object




```python
loans['zip_code'].nunique()
```




    949




```python
loans['addr_state'].sample(5)
```




    229346    TX
    200564    IL
    97998     OK
    132143    WA
    398716    CA
    Name: addr_state, dtype: object




```python
loans['addr_state'].nunique()
```




    51



There are a lot of different zip codes, so let's just keep the state column.


```python
loans.drop(labels='zip_code', axis=1, inplace=True)
```

Calculate the charge-off rates by address state:


```python
loans.groupby('addr_state')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off'].sort_values()
```




    addr_state
    ME    0.1282
    VT    0.1311
    DC    0.1350
    OR    0.1418
    IA    0.1429
    NH    0.1485
    WV    0.1491
    CO    0.1526
    WA    0.1534
    SC    0.1552
    MT    0.1604
    ID    0.1624
    WY    0.1627
    UT    0.1635
    KS    0.1658
    RI    0.1760
    IL    0.1780
    CT    0.1784
    ND    0.1831
    WI    0.1835
    GA    0.1844
    MA    0.1868
    DE    0.1891
    AK    0.1895
    AZ    0.1900
    MN    0.1920
    VA    0.1927
    OH    0.1930
    MI    0.1934
    TX    0.1945
    CA    0.1946
    KY    0.1985
    PA    0.1998
    NC    0.1998
    SD    0.2012
    HI    0.2021
    MO    0.2023
    NJ    0.2031
    TN    0.2034
    IN    0.2037
    NM    0.2059
    MD    0.2093
    FL    0.2141
    NV    0.2147
    NY    0.2160
    LA    0.2216
    NE    0.2248
    OK    0.2267
    AL    0.2271
    AR    0.2360
    MS    0.2489
    Name: loan_status, dtype: float64



The charge-off rate ranges from 13.0% in Washington, DC to 27.6% in Mississippi.

### 3.2.16 dti

Data Dictionary: "A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income."


```python
loans['dti'].describe()
```




    count    1.8592e+06
    mean     1.8568e+01
    std      1.3092e+01
    min     -1.0000e+00
    25%      1.1820e+01
    50%      1.7710e+01
    75%      2.4290e+01
    max      9.9900e+02
    Name: dti, dtype: float64



Note sure if the values of -1 and 999 make sense...

There are several outliers that mess up our default plots. Plot a histogram for `dti` less than 60:


```python
plt.figure(figsize=(8,3), dpi=90)
sns.distplot(loans.loc[loans['dti'].notnull() & (loans['dti']<60), 'dti'], kde=False)
plt.xlabel('Debt-to-income Ratio')
plt.ylabel('Count')
plt.title('Debt-to-income Ratio')
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    




    Text(0.5, 1.0, 'Debt-to-income Ratio')




    
![png](output_157_2.png)
    


How many of the `dti` values are "outliers" (above 60)?


```python
(loans['dti']>=60).sum()
```




    4874



Very few. Compare the summary statistics by loan status:


```python
loans.groupby('loan_status')['dti'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362325.0</td>
      <td>20.2683</td>
      <td>14.2722</td>
      <td>0.0</td>
      <td>13.27</td>
      <td>19.60</td>
      <td>26.31</td>
      <td>999.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1496898.0</td>
      <td>18.1561</td>
      <td>12.7553</td>
      <td>-1.0</td>
      <td>11.52</td>
      <td>17.29</td>
      <td>23.76</td>
      <td>999.0</td>
    </tr>
  </tbody>
</table>
</div>



Completed loans that are charged off tend to have higher debt-to-income ratios.

### 3.2.17 earliest_cr_line

Data Dictionary: "The month the borrower's earliest reported credit line was opened."


```python
loans['earliest_cr_line'].sample(5)
```




    67113     Apr-2002
    9966      Sep-1988
    143909    Dec-1980
    128824    Sep-2006
    33897     Dec-1994
    Name: earliest_cr_line, dtype: object




```python
loans['earliest_cr_line'].isnull().any()
```




    False



Let's just retain the year for simplicity:


```python
loans['earliest_cr_line'] = loans['earliest_cr_line'].apply(lambda s: int(s[-4:]))
```


```python
loans['earliest_cr_line'].describe()
```




    count    1.8603e+06
    mean     1.9993e+03
    std      7.7717e+00
    min      1.9330e+03
    25%      1.9950e+03
    50%      2.0010e+03
    75%      2.0050e+03
    max      2.0170e+03
    Name: earliest_cr_line, dtype: float64




```python
plot_var('earliest_cr_line', 'Year of Earliest Credit Line', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_170_1.png)
    


Borrowers who are charged-off tend to have shorter lines of credit.

### 3.2.18 fico_range_low, fico_range_high

Data Dictionary for `fico_range_low`: "The lower boundary range the borrower’s FICO at loan origination belongs to."

Data Dictionary for `fico_range_high`: "The upper boundary range the borrower’s FICO at loan origination belongs to."


```python
loans[['fico_range_low', 'fico_range_high']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.8603e+06</td>
      <td>1.8603e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.9768e+02</td>
      <td>7.0168e+02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.2709e+01</td>
      <td>3.2709e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.2500e+02</td>
      <td>6.2900e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.7000e+02</td>
      <td>6.7400e+02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.9000e+02</td>
      <td>6.9400e+02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.1500e+02</td>
      <td>7.1900e+02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.4500e+02</td>
      <td>8.5000e+02</td>
    </tr>
  </tbody>
</table>
</div>



Check the Pearson correlation between these values:


```python
loans[['fico_range_low','fico_range_high']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fico_range_low</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>fico_range_high</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



We only need to keep one of the FICO scores. We'll take the average of the two and call it `fico_score`:


```python
loans['fico_score'] = 0.5*loans['fico_range_low'] + 0.5*loans['fico_range_high']
```


```python
loans.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)
```


```python
plot_var('fico_score', 'FICO Score', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_180_1.png)
    


There is a noticeable difference in FICO scores between fully paid and charged-off loans. Compare the summary statistics:


```python
loans.groupby('loan_status')['fico_score'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>691.6664</td>
      <td>27.3203</td>
      <td>662.0</td>
      <td>672.0</td>
      <td>687.0</td>
      <td>707.0</td>
      <td>847.5</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>701.6180</td>
      <td>33.5988</td>
      <td>627.0</td>
      <td>677.0</td>
      <td>692.0</td>
      <td>717.0</td>
      <td>847.5</td>
    </tr>
  </tbody>
</table>
</div>



Loans that charge off have a FICO score 10 points lower on average.

### 3.2.19 open_acc

Data Dictionary: "The number of open credit lines in the borrower's credit file."


```python
plt.figure(figsize=(10,3), dpi=90)
sns.countplot(loans['open_acc'], order=sorted(loans['open_acc'].unique()), color='#5975A4', saturation=1)
_, _ = plt.xticks(np.arange(0, 90, 5), np.arange(0, 90, 5))
plt.title('Number of Open Credit Lines')
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    




    Text(0.5, 1.0, 'Number of Open Credit Lines')




    
![png](output_186_2.png)
    


Is there a difference in number of credit lines between fully paid loans and charged-off loans?


```python
loans.groupby('loan_status')['open_acc'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>11.8453</td>
      <td>5.7211</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>11.5480</td>
      <td>5.5381</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.20 pub_rec

Data Dictionary: "Number of derogatory public records."


```python
loans['pub_rec'].value_counts().sort_index()
```




    0.0     1552452
    1.0      260561
    2.0       30892
    3.0        9484
    4.0        3473
    5.0        1656
    6.0         830
    7.0         379
    8.0         214
    9.0         118
    10.0         77
    11.0         55
    12.0         32
    13.0         22
    14.0         10
    15.0         10
    16.0          9
    17.0          5
    18.0          6
    19.0          7
    20.0          3
    21.0          6
    22.0          3
    23.0          2
    24.0          3
    25.0          1
    28.0          4
    31.0          1
    32.0          1
    34.0          1
    37.0          1
    40.0          1
    44.0          1
    45.0          1
    46.0          1
    47.0          1
    49.0          2
    52.0          1
    54.0          1
    61.0          2
    63.0          1
    86.0          1
    Name: pub_rec, dtype: int64



Is there a difference in average public records between fully paid loans and charged-off loans?


```python
loans.groupby('loan_status')['pub_rec'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>0.2371</td>
      <td>0.6379</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>0.2013</td>
      <td>0.5778</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.21 revol_bal

Data Dictionary: "Total credit revolving balance."


```python
loans['revol_bal'].describe()
```




    count    1.8603e+06
    mean     1.6285e+04
    std      2.2468e+04
    min      0.0000e+00
    25%      5.8420e+03
    50%      1.1070e+04
    75%      1.9770e+04
    max      2.9048e+06
    Name: revol_bal, dtype: float64



Do a log transform:


```python
loans['log_revol_bal'] = loans['revol_bal'].apply(lambda x: np.log10(x+1))
```


```python
loans.drop('revol_bal', axis=1, inplace=True)
```


```python
plot_var('log_revol_bal', 'Log Revolving Credit Balance', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_200_1.png)
    



```python
loans.groupby('loan_status')['log_revol_bal'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>3.9733</td>
      <td>0.5392</td>
      <td>0.0</td>
      <td>3.7661</td>
      <td>4.0391</td>
      <td>4.2785</td>
      <td>6.2422</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>3.9877</td>
      <td>0.5519</td>
      <td>0.0</td>
      <td>3.7667</td>
      <td>4.0455</td>
      <td>4.3004</td>
      <td>6.4631</td>
    </tr>
  </tbody>
</table>
</div>



There isn't a large difference in the means.

### 3.2.22 revol_util

Data Dictionary: "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit."


```python
# Let's remove % percent sign from `revol_util` and convert it into `float`
loans.revol_util = loans.revol_util.map(lambda x: str(x).replace('%','')).astype(np.float64)
loans['revol_util'].head()
```




    0    83.7
    1     9.4
    2    98.5
    3    21.0
    4    53.9
    Name: revol_util, dtype: float64




```python
loans['revol_util'].describe()
```




    count    1.8589e+06
    mean     5.0464e+01
    std      2.4790e+01
    min      0.0000e+00
    25%      3.1600e+01
    50%      5.0500e+01
    75%      6.9600e+01
    max      8.9230e+02
    Name: revol_util, dtype: float64




```python
plot_var('revol_util', 'Revolving Line Utilization', continuous=True)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![png](output_207_1.png)
    



```python
loans.groupby('loan_status')['revol_util'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362279.0</td>
      <td>52.9553</td>
      <td>24.2451</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>53.5</td>
      <td>71.6</td>
      <td>366.6</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1496654.0</td>
      <td>49.8614</td>
      <td>24.8825</td>
      <td>0.0</td>
      <td>30.8</td>
      <td>49.8</td>
      <td>69.0</td>
      <td>892.3</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.23 total_acc

Data Dictionary: "The total number of credit lines currently in the borrower's credit file."


```python
plt.figure(figsize=(12,3), dpi=90)
sns.countplot(loans['total_acc'], order=sorted(loans['total_acc'].unique()), color='#5975A4', saturation=1)
_, _ = plt.xticks(np.arange(0, 176, 10), np.arange(0, 176, 10))
plt.title('Total Number of Credit Lines')
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    




    Text(0.5, 1.0, 'Total Number of Credit Lines')




    
![png](output_211_2.png)
    



```python
loans.groupby('loan_status')['total_acc'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>362548.0</td>
      <td>24.2129</td>
      <td>12.1633</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>22.0</td>
      <td>31.0</td>
      <td>176.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1497783.0</td>
      <td>24.6895</td>
      <td>12.0159</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>31.0</td>
      <td>165.0</td>
    </tr>
  </tbody>
</table>
</div>



No large differences here.

### 3.2.24 initial_list_status

Data Dictionary: "The initial listing status of the loan. Possible values are – W, F." I'm not sure what this means.


```python
plot_var('initial_list_status', 'Initial List Status', continuous=False)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    


    
![png](output_216_1.png)
    


### 3.2.25 application_type

Data Dictionary: "Indicates whether the loan is an individual application or a joint application with two co-borrowers."


```python
loans['application_type'].value_counts()
```




    Individual    1790838
    Joint App       69493
    Name: application_type, dtype: int64



Let's just compare the charge-off rates by application type:


```python
loans.groupby('application_type')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
```




    application_type
    Individual    0.1929
    Joint App     0.2462
    Name: loan_status, dtype: float64



Joint loans are slightly less likely to be charged-off.

### 3.2.26 mort_acc

Data Dictionary: "Number of mortgage accounts."


```python
loans['mort_acc'].describe()
```




    count    1.8130e+06
    mean     1.6066e+00
    std      1.9451e+00
    min      0.0000e+00
    25%      0.0000e+00
    50%      1.0000e+00
    75%      3.0000e+00
    max      6.1000e+01
    Name: mort_acc, dtype: float64



Not sure how someone can have 51 mortgage accounts...but apparently they do. Check the top 10 values:


```python
loans['mort_acc'].value_counts().head(10)
```




    0.0    747953
    1.0    319708
    2.0    266554
    3.0    191975
    4.0    127349
    5.0     75161
    6.0     41412
    7.0     21061
    8.0     10418
    9.0      5277
    Name: mort_acc, dtype: int64




```python
total_acc_avg = loans.groupby('total_acc').mean()['mort_acc'].fillna(0)

def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if mort_acc>=0:
        return total_acc_avg[total_acc]
    else:
        return mort_acc
```


```python
loans['mort_acc'] = loans.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
```

Compare the summary statistics by loan status:


```python
loans.groupby('loan_status')['mort_acc'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>355662.0</td>
      <td>1.5752</td>
      <td>0.7621</td>
      <td>0.0</td>
      <td>0.9295</td>
      <td>1.5731</td>
      <td>2.1987</td>
      <td>9.2857</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>1457388.0</td>
      <td>1.6142</td>
      <td>0.7517</td>
      <td>0.0</td>
      <td>1.0160</td>
      <td>1.6457</td>
      <td>2.2662</td>
      <td>11.0000</td>
    </tr>
  </tbody>
</table>
</div>



Individuals who pay off their loans are more likely to have several mortgage accounts.

### 3.2.27 pub_rec_bankruptcies

Data Dictionary: "Number of public record bankruptcies."


```python
loans['pub_rec_bankruptcies'].value_counts().sort_index()
```




    0.0     1628840
    1.0      218344
    2.0        9723
    3.0        1952
    4.0         505
    5.0         177
    6.0          58
    7.0          21
    8.0           9
    9.0           3
    11.0          1
    12.0          1
    Name: pub_rec_bankruptcies, dtype: int64




```python
plot_var('pub_rec_bankruptcies', 'Public Record Bankruptcies', continuous=False)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    


    
![png](output_236_1.png)
    


<a id="4"></a>
# 4. More Pre-processing

<a id="4.1"></a>
## 4.1 Convert loan status to 0/1 charge-off indicator

Change the response variable `loan_status` to a 0/1 variable, where 0 indicates fully paid and 1 indicates charge-off:


```python
loans['charged_off'] = (loans['loan_status'] == 'Charged Off').apply(np.uint8)
loans.drop('loan_status', axis=1, inplace=True)
```


```python
loans = loans.dropna().reset_index(drop=True)
```

<a id="4.2"></a>
## 4.2 Create dummy variables

How many variables do we currently have?


```python
loans.shape
```




    (1810610, 24)



If any categorical variables have missing values, we'll need to create NaN dummy variables for those. So first check which variables have missing data:


```python
missing_fractions = loans.isnull().mean().sort_values(ascending=False) # Fraction of data missing for each variable
```


```python
print(missing_fractions[missing_fractions > 0]) # Print variables that are missing data
```

    Series([], dtype: float64)
    

There are no categorical variables with missing values, and therefore we don't need any `NaN` dummy variables.

Create dummy variables for the categorical variables:


```python
print(loans.columns)
```

    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade',
           'home_ownership', 'verification_status', 'issue_d', 'purpose',
           'addr_state', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
           'revol_util', 'total_acc', 'initial_list_status', 'application_type',
           'mort_acc', 'pub_rec_bankruptcies', 'log_annual_inc', 'fico_score',
           'log_revol_bal', 'charged_off'],
          dtype='object')
    


```python
loans = pd.get_dummies(loans, columns=['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'application_type'], drop_first=True)
```

How many variables are there now?


```python
loans.shape
```




    (1810610, 121)



Check our data with the new dummy variables:


```python
loans.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>issue_d</th>
      <th>dti</th>
      <th>earliest_cr_line</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>log_annual_inc</th>
      <th>fico_score</th>
      <th>log_revol_bal</th>
      <th>charged_off</th>
      <th>sub_grade_A2</th>
      <th>sub_grade_A3</th>
      <th>sub_grade_A4</th>
      <th>sub_grade_A5</th>
      <th>sub_grade_B1</th>
      <th>sub_grade_B2</th>
      <th>sub_grade_B3</th>
      <th>sub_grade_B4</th>
      <th>sub_grade_B5</th>
      <th>sub_grade_C1</th>
      <th>sub_grade_C2</th>
      <th>sub_grade_C3</th>
      <th>sub_grade_C4</th>
      <th>sub_grade_C5</th>
      <th>sub_grade_D1</th>
      <th>sub_grade_D2</th>
      <th>sub_grade_D3</th>
      <th>sub_grade_D4</th>
      <th>sub_grade_D5</th>
      <th>sub_grade_E1</th>
      <th>sub_grade_E2</th>
      <th>sub_grade_E3</th>
      <th>sub_grade_E4</th>
      <th>sub_grade_E5</th>
      <th>sub_grade_F1</th>
      <th>sub_grade_F2</th>
      <th>sub_grade_F3</th>
      <th>sub_grade_F4</th>
      <th>sub_grade_F5</th>
      <th>sub_grade_G1</th>
      <th>sub_grade_G2</th>
      <th>sub_grade_G3</th>
      <th>sub_grade_G4</th>
      <th>sub_grade_G5</th>
      <th>home_ownership_OTHER</th>
      <th>home_ownership_OWN</th>
      <th>home_ownership_RENT</th>
      <th>verification_status_Source Verified</th>
      <th>verification_status_Verified</th>
      <th>purpose_credit_card</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_educational</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>addr_state_AL</th>
      <th>addr_state_AR</th>
      <th>addr_state_AZ</th>
      <th>addr_state_CA</th>
      <th>addr_state_CO</th>
      <th>addr_state_CT</th>
      <th>addr_state_DC</th>
      <th>addr_state_DE</th>
      <th>addr_state_FL</th>
      <th>addr_state_GA</th>
      <th>addr_state_HI</th>
      <th>addr_state_IA</th>
      <th>addr_state_ID</th>
      <th>addr_state_IL</th>
      <th>addr_state_IN</th>
      <th>addr_state_KS</th>
      <th>addr_state_KY</th>
      <th>addr_state_LA</th>
      <th>addr_state_MA</th>
      <th>addr_state_MD</th>
      <th>addr_state_ME</th>
      <th>addr_state_MI</th>
      <th>addr_state_MN</th>
      <th>addr_state_MO</th>
      <th>addr_state_MS</th>
      <th>addr_state_MT</th>
      <th>addr_state_NC</th>
      <th>addr_state_ND</th>
      <th>addr_state_NE</th>
      <th>addr_state_NH</th>
      <th>addr_state_NJ</th>
      <th>addr_state_NM</th>
      <th>addr_state_NV</th>
      <th>addr_state_NY</th>
      <th>addr_state_OH</th>
      <th>addr_state_OK</th>
      <th>addr_state_OR</th>
      <th>addr_state_PA</th>
      <th>addr_state_RI</th>
      <th>addr_state_SC</th>
      <th>addr_state_SD</th>
      <th>addr_state_TN</th>
      <th>addr_state_TX</th>
      <th>addr_state_UT</th>
      <th>addr_state_VA</th>
      <th>addr_state_VT</th>
      <th>addr_state_WA</th>
      <th>addr_state_WI</th>
      <th>addr_state_WV</th>
      <th>addr_state_WY</th>
      <th>initial_list_status_w</th>
      <th>application_type_Joint App</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57985</th>
      <td>9000.0</td>
      <td>36</td>
      <td>17.56</td>
      <td>323.39</td>
      <td>Aug-2013</td>
      <td>8.27</td>
      <td>2000</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>64.2</td>
      <td>19.0</td>
      <td>1.3075</td>
      <td>0.0</td>
      <td>4.6721</td>
      <td>672.0</td>
      <td>3.6695</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>535670</th>
      <td>6000.0</td>
      <td>36</td>
      <td>6.46</td>
      <td>183.79</td>
      <td>Feb-2019</td>
      <td>22.71</td>
      <td>2010</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>6.1</td>
      <td>25.0</td>
      <td>1.8113</td>
      <td>0.0</td>
      <td>5.0607</td>
      <td>777.0</td>
      <td>3.5250</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1409930</th>
      <td>23875.0</td>
      <td>36</td>
      <td>16.29</td>
      <td>842.80</td>
      <td>May-2016</td>
      <td>36.08</td>
      <td>1983</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>82.3</td>
      <td>21.0</td>
      <td>1.4771</td>
      <td>0.0</td>
      <td>4.6711</td>
      <td>672.0</td>
      <td>4.0123</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1464767</th>
      <td>4000.0</td>
      <td>36</td>
      <td>17.99</td>
      <td>144.59</td>
      <td>Sep-2016</td>
      <td>28.99</td>
      <td>2010</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>52.9</td>
      <td>8.0</td>
      <td>0.3260</td>
      <td>1.0</td>
      <td>4.6532</td>
      <td>667.0</td>
      <td>3.9789</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1274020</th>
      <td>18500.0</td>
      <td>36</td>
      <td>9.75</td>
      <td>594.78</td>
      <td>Mar-2016</td>
      <td>26.35</td>
      <td>1995</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>84.7</td>
      <td>20.0</td>
      <td>1.3781</td>
      <td>0.0</td>
      <td>4.6128</td>
      <td>722.0</td>
      <td>4.2244</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<a id="4.3"></a>
## 4.3 Train/test split


```python
# We'll make our modeling problem more realistic by performing the train/test split based on the month that the loan was funded. That is, we'll use loans funded on earlier dates to predict whether future loans will charge-off. The variable `issue_d` includes the month and year that the loan was funded.
# loans['issue_d'].sample(5)

# Are there any missing values?
# loans['issue_d'].isnull().any()

# No. Let's convert the issue dates to datetime objects:
# loans['issue_d'] = pd.to_datetime(loans['issue_d'])
# loans['issue_d'].sample(5)

# The new datetime values are all on the first day of the month. Check the summary statistics of the issue dates:
# loans['issue_d'].describe()
# There are only 154 unique issue dates over the 10-year period because we only have month/year information. In this particular dataset, the first loans were issued in June 2007, and the most recent loans were issued in March 2020. The busiest month was March 2016 with 57,553 loans funded in that month. What is the distribution of loans funded in each year?

# plt.figure(figsize=(6,3), dpi=90)
# loans['issue_d'].dt.year.value_counts().sort_index().plot.bar(color='darkblue')
# plt.xlabel('Year')
# plt.ylabel('Number of Loans Funded')
# plt.title('Loans Funded per Year')

# We'll form the test set from the most recent 10% of the loans.

# loans_train = loans.loc[loans['issue_d'] <  loans['issue_d'].quantile(0.9)]
# loans_test =  loans.loc[loans['issue_d'] >= loans['issue_d'].quantile(0.9)]
# Refer [this video](https://www.youtube.com/watch?v=idXEk3MjC9M) if you didn't understood the quantile. Check that we properly partitioned the loans:

# What is the test size?
# loans_test.shape[0] / loans.shape[0]
# About 10.8%. The partition looks good, so we can delete the original `loans` dataframe:
# The training set includes loans from June 2007 to April 2018. The test set includes loans from May 2018 to March 2020. Now we need to delete the `issue_d` variable, because it was not available before the loan was funded.

# del loans

# loans_train.drop('issue_d', axis=1, inplace=True)
# loans_test.drop('issue_d', axis=1, inplace=True)

# del loans_train, loans_test

# y_train = loans_train['charged_off']
# y_test = loans_test['charged_off']

# X_train = loans_train.drop('charged_off', axis=1)
# X_test = loans_test.drop('charged_off', axis=1)
```


```python
loans.drop('issue_d', axis=1, inplace=True)
```


```python
# Features
X = loans.drop('charged_off',axis=1)

# Label
y = loans['charged_off']
```


```python
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
```


```python
print("X_train.shape: ", X_train.shape)
print("X_test.shape:  ", X_test.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape:  ", y_test.shape)
```

    X_train.shape:  (1448488, 119)
    X_test.shape:   (362122, 119)
    y_train.shape:  (1448488,)
    y_test.shape:   (362122,)
    

<a id="5"></a>
# 5. Linear Dependence of Charge-off on the Predictors

On the training set, we compute the [Pearson correlation](https://www.youtube.com/watch?v=6fUYt1alA1U), $F$-statistic, and $p$ value of each predictor with the response variable `charged_off`.


```python
linear_dep = pd.DataFrame()
```

<a id="5.1"></a>
## 5.1 Pearson correlation
We use the Pearson correlation coefficient to examine the strength and direction of the linear relationship between two continuous variables.

The correlation coefficient can range in value from −1 to +1. The larger the absolute value of the coefficient, the stronger the relationship between the variables. For the Pearson correlation, an absolute value of 1 indicates a perfect linear relationship. A correlation close to 0 indicates no linear relationship between the variables. 

The sign of the coefficient indicates the direction of the relationship. If both variables tend to increase or decrease together, the coefficient is positive, and the line that represents the correlation slopes upward. If one variable tends to increase as the other decreases, the coefficient is negative, and the line that represents the correlation slopes downward.

* We can see a strong correlation between loan_amnt and installment. (The monthly payment owed by the borrower if the loan originates)


```python
for col in X_train.columns:
    linear_dep.loc[col, 'pearson_corr'] = X_train[col].corr(y_train)
linear_dep['abs_pearson_corr'] = abs(linear_dep['pearson_corr'])
```


```python
linear_dep
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pearson_corr</th>
      <th>abs_pearson_corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>0.0706</td>
      <td>0.0706</td>
    </tr>
    <tr>
      <th>term</th>
      <td>0.1690</td>
      <td>0.1690</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>0.2540</td>
      <td>0.2540</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>0.0587</td>
      <td>0.0587</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0.0639</td>
      <td>0.0639</td>
    </tr>
    <tr>
      <th>earliest_cr_line</th>
      <td>0.0459</td>
      <td>0.0459</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>0.0199</td>
      <td>0.0199</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>0.0230</td>
      <td>0.0230</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0.0484</td>
      <td>0.0484</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>-0.0165</td>
      <td>0.0165</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>-0.0208</td>
      <td>0.0208</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>0.0265</td>
      <td>0.0265</td>
    </tr>
    <tr>
      <th>log_annual_inc</th>
      <td>-0.0615</td>
      <td>0.0615</td>
    </tr>
    <tr>
      <th>fico_score</th>
      <td>-0.1194</td>
      <td>0.1194</td>
    </tr>
    <tr>
      <th>log_revol_bal</th>
      <td>-0.0126</td>
      <td>0.0126</td>
    </tr>
    <tr>
      <th>sub_grade_A2</th>
      <td>-0.0639</td>
      <td>0.0639</td>
    </tr>
    <tr>
      <th>sub_grade_A3</th>
      <td>-0.0604</td>
      <td>0.0604</td>
    </tr>
    <tr>
      <th>sub_grade_A4</th>
      <td>-0.0630</td>
      <td>0.0630</td>
    </tr>
    <tr>
      <th>sub_grade_A5</th>
      <td>-0.0606</td>
      <td>0.0606</td>
    </tr>
    <tr>
      <th>sub_grade_B1</th>
      <td>-0.0548</td>
      <td>0.0548</td>
    </tr>
    <tr>
      <th>sub_grade_B2</th>
      <td>-0.0477</td>
      <td>0.0477</td>
    </tr>
    <tr>
      <th>sub_grade_B3</th>
      <td>-0.0402</td>
      <td>0.0402</td>
    </tr>
    <tr>
      <th>sub_grade_B4</th>
      <td>-0.0307</td>
      <td>0.0307</td>
    </tr>
    <tr>
      <th>sub_grade_B5</th>
      <td>-0.0194</td>
      <td>0.0194</td>
    </tr>
    <tr>
      <th>sub_grade_C1</th>
      <td>-0.0052</td>
      <td>0.0052</td>
    </tr>
    <tr>
      <th>sub_grade_C2</th>
      <td>0.0070</td>
      <td>0.0070</td>
    </tr>
    <tr>
      <th>sub_grade_C3</th>
      <td>0.0186</td>
      <td>0.0186</td>
    </tr>
    <tr>
      <th>sub_grade_C4</th>
      <td>0.0321</td>
      <td>0.0321</td>
    </tr>
    <tr>
      <th>sub_grade_C5</th>
      <td>0.0387</td>
      <td>0.0387</td>
    </tr>
    <tr>
      <th>sub_grade_D1</th>
      <td>0.0388</td>
      <td>0.0388</td>
    </tr>
    <tr>
      <th>sub_grade_D2</th>
      <td>0.0460</td>
      <td>0.0460</td>
    </tr>
    <tr>
      <th>sub_grade_D3</th>
      <td>0.0483</td>
      <td>0.0483</td>
    </tr>
    <tr>
      <th>sub_grade_D4</th>
      <td>0.0520</td>
      <td>0.0520</td>
    </tr>
    <tr>
      <th>sub_grade_D5</th>
      <td>0.0540</td>
      <td>0.0540</td>
    </tr>
    <tr>
      <th>sub_grade_E1</th>
      <td>0.0481</td>
      <td>0.0481</td>
    </tr>
    <tr>
      <th>sub_grade_E2</th>
      <td>0.0503</td>
      <td>0.0503</td>
    </tr>
    <tr>
      <th>sub_grade_E3</th>
      <td>0.0535</td>
      <td>0.0535</td>
    </tr>
    <tr>
      <th>sub_grade_E4</th>
      <td>0.0524</td>
      <td>0.0524</td>
    </tr>
    <tr>
      <th>sub_grade_E5</th>
      <td>0.0580</td>
      <td>0.0580</td>
    </tr>
    <tr>
      <th>sub_grade_F1</th>
      <td>0.0445</td>
      <td>0.0445</td>
    </tr>
    <tr>
      <th>sub_grade_F2</th>
      <td>0.0425</td>
      <td>0.0425</td>
    </tr>
    <tr>
      <th>sub_grade_F3</th>
      <td>0.0385</td>
      <td>0.0385</td>
    </tr>
    <tr>
      <th>sub_grade_F4</th>
      <td>0.0379</td>
      <td>0.0379</td>
    </tr>
    <tr>
      <th>sub_grade_F5</th>
      <td>0.0359</td>
      <td>0.0359</td>
    </tr>
    <tr>
      <th>sub_grade_G1</th>
      <td>0.0306</td>
      <td>0.0306</td>
    </tr>
    <tr>
      <th>sub_grade_G2</th>
      <td>0.0255</td>
      <td>0.0255</td>
    </tr>
    <tr>
      <th>sub_grade_G3</th>
      <td>0.0238</td>
      <td>0.0238</td>
    </tr>
    <tr>
      <th>sub_grade_G4</th>
      <td>0.0238</td>
      <td>0.0238</td>
    </tr>
    <tr>
      <th>sub_grade_G5</th>
      <td>0.0226</td>
      <td>0.0226</td>
    </tr>
    <tr>
      <th>home_ownership_OTHER</th>
      <td>0.0003</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>home_ownership_OWN</th>
      <td>0.0047</td>
      <td>0.0047</td>
    </tr>
    <tr>
      <th>home_ownership_RENT</th>
      <td>0.0657</td>
      <td>0.0657</td>
    </tr>
    <tr>
      <th>verification_status_Source Verified</th>
      <td>0.0132</td>
      <td>0.0132</td>
    </tr>
    <tr>
      <th>verification_status_Verified</th>
      <td>0.0628</td>
      <td>0.0628</td>
    </tr>
    <tr>
      <th>purpose_credit_card</th>
      <td>-0.0364</td>
      <td>0.0364</td>
    </tr>
    <tr>
      <th>purpose_debt_consolidation</th>
      <td>0.0331</td>
      <td>0.0331</td>
    </tr>
    <tr>
      <th>purpose_educational</th>
      <td>-0.0004</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>purpose_home_improvement</th>
      <td>-0.0148</td>
      <td>0.0148</td>
    </tr>
    <tr>
      <th>purpose_house</th>
      <td>0.0057</td>
      <td>0.0057</td>
    </tr>
    <tr>
      <th>purpose_major_purchase</th>
      <td>-0.0024</td>
      <td>0.0024</td>
    </tr>
    <tr>
      <th>purpose_medical</th>
      <td>0.0022</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>purpose_moving</th>
      <td>0.0057</td>
      <td>0.0057</td>
    </tr>
    <tr>
      <th>purpose_other</th>
      <td>0.0040</td>
      <td>0.0040</td>
    </tr>
    <tr>
      <th>purpose_renewable_energy</th>
      <td>0.0015</td>
      <td>0.0015</td>
    </tr>
    <tr>
      <th>purpose_small_business</th>
      <td>0.0233</td>
      <td>0.0233</td>
    </tr>
    <tr>
      <th>purpose_vacation</th>
      <td>-0.0035</td>
      <td>0.0035</td>
    </tr>
    <tr>
      <th>purpose_wedding</th>
      <td>-0.0033</td>
      <td>0.0033</td>
    </tr>
    <tr>
      <th>addr_state_AL</th>
      <td>0.0097</td>
      <td>0.0097</td>
    </tr>
    <tr>
      <th>addr_state_AR</th>
      <td>0.0091</td>
      <td>0.0091</td>
    </tr>
    <tr>
      <th>addr_state_AZ</th>
      <td>-0.0020</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>addr_state_CA</th>
      <td>-0.0004</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>addr_state_CO</th>
      <td>-0.0164</td>
      <td>0.0164</td>
    </tr>
    <tr>
      <th>addr_state_CT</th>
      <td>-0.0047</td>
      <td>0.0047</td>
    </tr>
    <tr>
      <th>addr_state_DC</th>
      <td>-0.0068</td>
      <td>0.0068</td>
    </tr>
    <tr>
      <th>addr_state_DE</th>
      <td>-0.0011</td>
      <td>0.0011</td>
    </tr>
    <tr>
      <th>addr_state_FL</th>
      <td>0.0128</td>
      <td>0.0128</td>
    </tr>
    <tr>
      <th>addr_state_GA</th>
      <td>-0.0059</td>
      <td>0.0059</td>
    </tr>
    <tr>
      <th>addr_state_HI</th>
      <td>0.0016</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>addr_state_IA</th>
      <td>0.0017</td>
      <td>0.0017</td>
    </tr>
    <tr>
      <th>addr_state_ID</th>
      <td>-0.0033</td>
      <td>0.0033</td>
    </tr>
    <tr>
      <th>addr_state_IL</th>
      <td>-0.0086</td>
      <td>0.0086</td>
    </tr>
    <tr>
      <th>addr_state_IN</th>
      <td>0.0028</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <th>addr_state_KS</th>
      <td>-0.0062</td>
      <td>0.0062</td>
    </tr>
    <tr>
      <th>addr_state_KY</th>
      <td>0.0014</td>
      <td>0.0014</td>
    </tr>
    <tr>
      <th>addr_state_LA</th>
      <td>0.0079</td>
      <td>0.0079</td>
    </tr>
    <tr>
      <th>addr_state_MA</th>
      <td>-0.0032</td>
      <td>0.0032</td>
    </tr>
    <tr>
      <th>addr_state_MD</th>
      <td>0.0056</td>
      <td>0.0056</td>
    </tr>
    <tr>
      <th>addr_state_ME</th>
      <td>-0.0075</td>
      <td>0.0075</td>
    </tr>
    <tr>
      <th>addr_state_MI</th>
      <td>-0.0008</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>addr_state_MN</th>
      <td>-0.0011</td>
      <td>0.0011</td>
    </tr>
    <tr>
      <th>addr_state_MO</th>
      <td>0.0023</td>
      <td>0.0023</td>
    </tr>
    <tr>
      <th>addr_state_MS</th>
      <td>0.0097</td>
      <td>0.0097</td>
    </tr>
    <tr>
      <th>addr_state_MT</th>
      <td>-0.0044</td>
      <td>0.0044</td>
    </tr>
    <tr>
      <th>addr_state_NC</th>
      <td>0.0015</td>
      <td>0.0015</td>
    </tr>
    <tr>
      <th>addr_state_ND</th>
      <td>-0.0010</td>
      <td>0.0010</td>
    </tr>
    <tr>
      <th>addr_state_NE</th>
      <td>0.0041</td>
      <td>0.0041</td>
    </tr>
    <tr>
      <th>addr_state_NH</th>
      <td>-0.0084</td>
      <td>0.0084</td>
    </tr>
    <tr>
      <th>addr_state_NJ</th>
      <td>0.0036</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <th>addr_state_NM</th>
      <td>0.0023</td>
      <td>0.0023</td>
    </tr>
    <tr>
      <th>addr_state_NV</th>
      <td>0.0053</td>
      <td>0.0053</td>
    </tr>
    <tr>
      <th>addr_state_NY</th>
      <td>0.0173</td>
      <td>0.0173</td>
    </tr>
    <tr>
      <th>addr_state_OH</th>
      <td>-0.0010</td>
      <td>0.0010</td>
    </tr>
    <tr>
      <th>addr_state_OK</th>
      <td>0.0075</td>
      <td>0.0075</td>
    </tr>
    <tr>
      <th>addr_state_OR</th>
      <td>-0.0152</td>
      <td>0.0152</td>
    </tr>
    <tr>
      <th>addr_state_PA</th>
      <td>0.0032</td>
      <td>0.0032</td>
    </tr>
    <tr>
      <th>addr_state_RI</th>
      <td>-0.0034</td>
      <td>0.0034</td>
    </tr>
    <tr>
      <th>addr_state_SC</th>
      <td>-0.0116</td>
      <td>0.0116</td>
    </tr>
    <tr>
      <th>addr_state_SD</th>
      <td>0.0004</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>addr_state_TN</th>
      <td>0.0022</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>addr_state_TX</th>
      <td>0.0001</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>addr_state_UT</th>
      <td>-0.0064</td>
      <td>0.0064</td>
    </tr>
    <tr>
      <th>addr_state_VA</th>
      <td>-0.0008</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>addr_state_VT</th>
      <td>-0.0078</td>
      <td>0.0078</td>
    </tr>
    <tr>
      <th>addr_state_WA</th>
      <td>-0.0156</td>
      <td>0.0156</td>
    </tr>
    <tr>
      <th>addr_state_WI</th>
      <td>-0.0032</td>
      <td>0.0032</td>
    </tr>
    <tr>
      <th>addr_state_WV</th>
      <td>-0.0067</td>
      <td>0.0067</td>
    </tr>
    <tr>
      <th>addr_state_WY</th>
      <td>-0.0033</td>
      <td>0.0033</td>
    </tr>
    <tr>
      <th>initial_list_status_w</th>
      <td>0.0036</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <th>application_type_Joint App</th>
      <td>0.0255</td>
      <td>0.0255</td>
    </tr>
  </tbody>
</table>
</div>



$F$-statistics:


```python
from sklearn.feature_selection import f_classif
for col in X_train.columns:
    mask = X_train[col].notnull()
    (linear_dep.loc[col, 'F'], linear_dep.loc[col, 'p_value']) = f_classif(pd.DataFrame(X_train.loc[mask, col]), y_train.loc[mask])
```

Sort the results by the absolute value of the Pearson correlation:


```python
linear_dep.sort_values('abs_pearson_corr', ascending=False, inplace=True)
linear_dep.drop('abs_pearson_corr', axis=1, inplace=True)
```

Reset the index:


```python
linear_dep.reset_index(inplace=True)
linear_dep.rename(columns={'index':'variable'}, inplace=True)
```

View the results for the top 20 predictors most correlated with `charged_off`:


```python
linear_dep.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>pearson_corr</th>
      <th>F</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>int_rate</td>
      <td>0.2540</td>
      <td>99925.2512</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>0.1690</td>
      <td>42590.8883</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fico_score</td>
      <td>-0.1194</td>
      <td>20937.9775</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loan_amnt</td>
      <td>0.0706</td>
      <td>7248.7055</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>home_ownership_RENT</td>
      <td>0.0657</td>
      <td>6287.7696</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade_A2</td>
      <td>-0.0639</td>
      <td>5943.2444</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dti</td>
      <td>0.0639</td>
      <td>5938.5575</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sub_grade_A4</td>
      <td>-0.0630</td>
      <td>5775.0384</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>verification_status_Verified</td>
      <td>0.0628</td>
      <td>5740.1374</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>log_annual_inc</td>
      <td>-0.0615</td>
      <td>5500.9651</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sub_grade_A5</td>
      <td>-0.0606</td>
      <td>5331.7764</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sub_grade_A3</td>
      <td>-0.0604</td>
      <td>5297.2539</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>installment</td>
      <td>0.0587</td>
      <td>5008.4254</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sub_grade_E5</td>
      <td>0.0580</td>
      <td>4887.3351</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sub_grade_B1</td>
      <td>-0.0548</td>
      <td>4363.7033</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>sub_grade_D5</td>
      <td>0.0540</td>
      <td>4229.2016</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sub_grade_E3</td>
      <td>0.0535</td>
      <td>4150.8384</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sub_grade_E4</td>
      <td>0.0524</td>
      <td>3991.2166</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sub_grade_D4</td>
      <td>0.0520</td>
      <td>3920.7495</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>sub_grade_E2</td>
      <td>0.0503</td>
      <td>3670.0409</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>revol_util</td>
      <td>0.0484</td>
      <td>3402.4683</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>sub_grade_D3</td>
      <td>0.0483</td>
      <td>3384.2204</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>sub_grade_E1</td>
      <td>0.0481</td>
      <td>3354.9387</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>sub_grade_B2</td>
      <td>-0.0477</td>
      <td>3299.2963</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>sub_grade_D2</td>
      <td>0.0460</td>
      <td>3070.9537</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>earliest_cr_line</td>
      <td>0.0459</td>
      <td>3055.5958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>sub_grade_F1</td>
      <td>0.0445</td>
      <td>2870.9591</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>sub_grade_F2</td>
      <td>0.0425</td>
      <td>2619.4453</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>sub_grade_B3</td>
      <td>-0.0402</td>
      <td>2343.3164</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>sub_grade_D1</td>
      <td>0.0388</td>
      <td>2188.7583</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The variables most linearly correlated with `charged_off` are the interest rate, loan period (term), FICO score, debt-to-income ratio, number of mortgages, income, the loan grade, and the loan amount.

Now view the results for the 20 least correlated predictors:


```python
linear_dep.tail(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>pearson_corr</th>
      <th>F</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>addr_state_MO</td>
      <td>0.0023</td>
      <td>7.6653</td>
      <td>0.0056</td>
    </tr>
    <tr>
      <th>100</th>
      <td>addr_state_TN</td>
      <td>0.0022</td>
      <td>7.3217</td>
      <td>0.0068</td>
    </tr>
    <tr>
      <th>101</th>
      <td>purpose_medical</td>
      <td>0.0022</td>
      <td>7.1296</td>
      <td>0.0076</td>
    </tr>
    <tr>
      <th>102</th>
      <td>addr_state_AZ</td>
      <td>-0.0020</td>
      <td>5.9065</td>
      <td>0.0151</td>
    </tr>
    <tr>
      <th>103</th>
      <td>addr_state_IA</td>
      <td>0.0017</td>
      <td>4.1017</td>
      <td>0.0428</td>
    </tr>
    <tr>
      <th>104</th>
      <td>addr_state_HI</td>
      <td>0.0016</td>
      <td>3.5989</td>
      <td>0.0578</td>
    </tr>
    <tr>
      <th>105</th>
      <td>addr_state_NC</td>
      <td>0.0015</td>
      <td>3.4212</td>
      <td>0.0644</td>
    </tr>
    <tr>
      <th>106</th>
      <td>purpose_renewable_energy</td>
      <td>0.0015</td>
      <td>3.4174</td>
      <td>0.0645</td>
    </tr>
    <tr>
      <th>107</th>
      <td>addr_state_KY</td>
      <td>0.0014</td>
      <td>2.8849</td>
      <td>0.0894</td>
    </tr>
    <tr>
      <th>108</th>
      <td>addr_state_DE</td>
      <td>-0.0011</td>
      <td>1.7907</td>
      <td>0.1808</td>
    </tr>
    <tr>
      <th>109</th>
      <td>addr_state_MN</td>
      <td>-0.0011</td>
      <td>1.7868</td>
      <td>0.1813</td>
    </tr>
    <tr>
      <th>110</th>
      <td>addr_state_ND</td>
      <td>-0.0010</td>
      <td>1.5832</td>
      <td>0.2083</td>
    </tr>
    <tr>
      <th>111</th>
      <td>addr_state_OH</td>
      <td>-0.0010</td>
      <td>1.5772</td>
      <td>0.2092</td>
    </tr>
    <tr>
      <th>112</th>
      <td>addr_state_VA</td>
      <td>-0.0008</td>
      <td>0.9656</td>
      <td>0.3258</td>
    </tr>
    <tr>
      <th>113</th>
      <td>addr_state_MI</td>
      <td>-0.0008</td>
      <td>0.8900</td>
      <td>0.3455</td>
    </tr>
    <tr>
      <th>114</th>
      <td>addr_state_CA</td>
      <td>-0.0004</td>
      <td>0.2725</td>
      <td>0.6017</td>
    </tr>
    <tr>
      <th>115</th>
      <td>purpose_educational</td>
      <td>-0.0004</td>
      <td>0.2438</td>
      <td>0.6215</td>
    </tr>
    <tr>
      <th>116</th>
      <td>addr_state_SD</td>
      <td>0.0004</td>
      <td>0.2118</td>
      <td>0.6454</td>
    </tr>
    <tr>
      <th>117</th>
      <td>home_ownership_OTHER</td>
      <td>0.0003</td>
      <td>0.0912</td>
      <td>0.7626</td>
    </tr>
    <tr>
      <th>118</th>
      <td>addr_state_TX</td>
      <td>0.0001</td>
      <td>0.0233</td>
      <td>0.8788</td>
    </tr>
  </tbody>
</table>
</div>



It looks like the borrower's state of residence, the revolving balance, and several of the loan purposes are irrelevant for predicting charge-off.

<a id="6"></a>
# 6. Model Training and Testing

We implement machine learning pipelines consisting of one or more of the following steps, depending on the particular model:
1. Mean imputation of missing values
2. Dimension reduction using linear discriminant analysis (LDA)
3. Data standardization: rescaling to zero mean and unit variance
4. The chosen model

We will evaluate and compare the following models using a cross-validated AUROC score on the training set:
1. Logistic regression with SGD training
2. Random forest

We'll perform some hyperparameter tuning for each model to choose the most promising model, then more carefully tune the hyperparameters of the best-performing model.


```python
scaler = MinMaxScaler()

# fit and transfrom
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# everything has been scaled between 1 and 0
print('Max: ',X_train.max())
print('Min: ', X_train.min())
```

    Max:  1.0000000000000004
    Min:  0.0
    

<a id="6.1"></a>
## 6.1 Logistic regression with SGD training

The `SGDClassifier` estimator in scikit-learn implements linear classifiers (SVM, logistic regression, and others) with stochastic gradient descent (SGD) training. A particular linear classifier is chosen through the `loss` hyperparameter. Because we want to predict the probability of charge-off, we choose logistic regression (a probabilistic classifier) by setting `loss = 'log'`.

The machine learning pipeline:


```python
pipeline_sgdlogreg = Pipeline([
    ('imputer', SimpleImputer(copy=False)), # Mean imputation by default
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])
```

A small grid of hyperparameters to search over:


```python
param_grid_sgdlogreg = {
    'model__alpha': [10**-5, 10**-2, 10**1],
    'model__penalty': ['l1', 'l2']
}
```

Create the search grid object:


```python
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
```

Conduct the grid search and train the final model on the whole dataset:


```python
grid_sgdlogreg.fit(X_train, y_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed: 12.0min finished
    




    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('imputer', SimpleImputer(copy=False)),
                                           ('scaler', StandardScaler(copy=False)),
                                           ('model',
                                            SGDClassifier(loss='log',
                                                          random_state=1,
                                                          warm_start=True))]),
                 n_jobs=1,
                 param_grid={'model__alpha': [1e-05, 0.01, 10],
                             'model__penalty': ['l1', 'l2']},
                 pre_dispatch=1, scoring='roc_auc', verbose=1)



Mean cross-validated AUROC score of the best model:


```python
grid_sgdlogreg.best_score_
```




    0.705849170214228



Best hyperparameters:


```python
grid_sgdlogreg.best_params_
```




    {'model__alpha': 0.01, 'model__penalty': 'l2'}



<a id="6.2"></a>
## 6.2 Random forest classifier

Next we train a random forest model. Note that data standardization is not necessary for a random forest.


```python
pipeline_rfc = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])
```

The random forest takes very long to train, so we don't test different hyperparameter choices. We'll still use `GridSearchCV` for the sake of consistency.


```python
param_grid_rfc = {
    'model__n_estimators': [50] # The number of randomized trees to build
}
```

The AUROC will always improve (with decreasing gains) as the number of estimators increases, but it's not necessarily worth the extra training time and model complexity.


```python
grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
```


```python
grid_rfc.fit(X_train, y_train)
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:  6.8min finished
    




    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('imputer', SimpleImputer(copy=False)),
                                           ('model',
                                            RandomForestClassifier(n_jobs=-1,
                                                                   random_state=1))]),
                 n_jobs=1, param_grid={'model__n_estimators': [50]}, pre_dispatch=1,
                 scoring='roc_auc', verbose=1)



Mean cross-validated AUROC score of the random forest:


```python
grid_rfc.best_score_
```




    0.6922500368673525



Not quite as good as logistic regression, at least according to this metric.

<a id="6.4"></a>
## 6.4 Tune hyperparameters on the chosen model more finely

The three models performed quite similarly according to the AUROC:


```python
print('Cross-validated AUROC scores')
print(grid_sgdlogreg.best_score_, '- Logistic regression')
print(grid_rfc.best_score_, '- Random forest')
```

    Cross-validated AUROC scores
    0.705849170214228 - Logistic regression
    0.6922500368673525 - Random forest
    

Logistic regression squeaked out ahead, and coupled with the fact that `SGDClassifier` trains much faster than the other two models, we'll select logistic regression as our final model. Now we'll tune the hyperparameters more finely.


```python
param_grid_sgdlogreg = {
    'model__alpha': np.logspace(-4.5, 0.5, 11), # Fills in the gaps between 10^-5 and 10^1
    'model__penalty': ['l1', 'l2']
}

print(param_grid_sgdlogreg)
```

    {'model__alpha': array([3.16227766e-05, 1.00000000e-04, 3.16227766e-04, 1.00000000e-03,
           3.16227766e-03, 1.00000000e-02, 3.16227766e-02, 1.00000000e-01,
           3.16227766e-01, 1.00000000e+00, 3.16227766e+00]), 'model__penalty': ['l1', 'l2']}
    


```python
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
```


```python
grid_sgdlogreg.fit(X_train, y_train)
```

    Fitting 5 folds for each of 22 candidates, totalling 110 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 110 out of 110 | elapsed: 33.4min finished
    




    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('imputer', SimpleImputer(copy=False)),
                                           ('scaler', StandardScaler(copy=False)),
                                           ('model',
                                            SGDClassifier(loss='log',
                                                          random_state=1,
                                                          warm_start=True))]),
                 n_jobs=1,
                 param_grid={'model__alpha': array([3.16227766e-05, 1.00000000e-04, 3.16227766e-04, 1.00000000e-03,
           3.16227766e-03, 1.00000000e-02, 3.16227766e-02, 1.00000000e-01,
           3.16227766e-01, 1.00000000e+00, 3.16227766e+00]),
                             'model__penalty': ['l1', 'l2']},
                 pre_dispatch=1, scoring='roc_auc', verbose=1)



Mean cross-validated AUROC score of the best model:


```python
grid_sgdlogreg.best_score_
```




    0.7058472102563582



Best hyperparameters:


```python
grid_sgdlogreg.best_params_
```




    {'model__alpha': 0.01, 'model__penalty': 'l2'}



By some coincidence, the optimal hyperparameters here are the same as from our first grid search for logistic regression!

<a id="6.5"></a>
## 6.5 Test set evaluation

Now we can finally see how our chosen model performs on the test data (the most recent 10% of the loans).


```python
y_score = grid_sgdlogreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_score)
```




    0.6899233896732194



The test set AUROC score is somewhat lower than the cross-validated score (0.713).

## Shapley Value

### SHAP Values
- Shap values show how much a given feature changed our prediction (compared to if we made that prediction at some baseline value of that feature).
- In addition to this nice breakdown for each prediction, the Shap library offers great visualizations of groups of Shap values. These visualizations have conceptual similarities to permutation importance and partial dependence plots.


```python
from xgboost import XGBClassifier
```


```python
%%time
my_model = XGBClassifier(n_estimators=100).fit(X_train, y_train)
```

    C:\Users\ykamd\.conda\envs\ve1\lib\site-packages\xgboost\sklearn.py:888: UserWarning:
    
    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    
    

    [23:23:49] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Wall time: 3min 17s
    


```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>dti</th>
      <th>earliest_cr_line</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>log_annual_inc</th>
      <th>fico_score</th>
      <th>log_revol_bal</th>
      <th>sub_grade_A2</th>
      <th>sub_grade_A3</th>
      <th>sub_grade_A4</th>
      <th>sub_grade_A5</th>
      <th>sub_grade_B1</th>
      <th>sub_grade_B2</th>
      <th>sub_grade_B3</th>
      <th>sub_grade_B4</th>
      <th>sub_grade_B5</th>
      <th>sub_grade_C1</th>
      <th>sub_grade_C2</th>
      <th>sub_grade_C3</th>
      <th>sub_grade_C4</th>
      <th>sub_grade_C5</th>
      <th>sub_grade_D1</th>
      <th>sub_grade_D2</th>
      <th>sub_grade_D3</th>
      <th>sub_grade_D4</th>
      <th>sub_grade_D5</th>
      <th>sub_grade_E1</th>
      <th>sub_grade_E2</th>
      <th>sub_grade_E3</th>
      <th>sub_grade_E4</th>
      <th>sub_grade_E5</th>
      <th>sub_grade_F1</th>
      <th>sub_grade_F2</th>
      <th>sub_grade_F3</th>
      <th>sub_grade_F4</th>
      <th>sub_grade_F5</th>
      <th>sub_grade_G1</th>
      <th>sub_grade_G2</th>
      <th>sub_grade_G3</th>
      <th>sub_grade_G4</th>
      <th>sub_grade_G5</th>
      <th>home_ownership_OTHER</th>
      <th>home_ownership_OWN</th>
      <th>home_ownership_RENT</th>
      <th>verification_status_Source Verified</th>
      <th>verification_status_Verified</th>
      <th>purpose_credit_card</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_educational</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>addr_state_AL</th>
      <th>addr_state_AR</th>
      <th>addr_state_AZ</th>
      <th>addr_state_CA</th>
      <th>addr_state_CO</th>
      <th>addr_state_CT</th>
      <th>addr_state_DC</th>
      <th>addr_state_DE</th>
      <th>addr_state_FL</th>
      <th>addr_state_GA</th>
      <th>addr_state_HI</th>
      <th>addr_state_IA</th>
      <th>addr_state_ID</th>
      <th>addr_state_IL</th>
      <th>addr_state_IN</th>
      <th>addr_state_KS</th>
      <th>addr_state_KY</th>
      <th>addr_state_LA</th>
      <th>addr_state_MA</th>
      <th>addr_state_MD</th>
      <th>addr_state_ME</th>
      <th>addr_state_MI</th>
      <th>addr_state_MN</th>
      <th>addr_state_MO</th>
      <th>addr_state_MS</th>
      <th>addr_state_MT</th>
      <th>addr_state_NC</th>
      <th>addr_state_ND</th>
      <th>addr_state_NE</th>
      <th>addr_state_NH</th>
      <th>addr_state_NJ</th>
      <th>addr_state_NM</th>
      <th>addr_state_NV</th>
      <th>addr_state_NY</th>
      <th>addr_state_OH</th>
      <th>addr_state_OK</th>
      <th>addr_state_OR</th>
      <th>addr_state_PA</th>
      <th>addr_state_RI</th>
      <th>addr_state_SC</th>
      <th>addr_state_SD</th>
      <th>addr_state_TN</th>
      <th>addr_state_TX</th>
      <th>addr_state_UT</th>
      <th>addr_state_VA</th>
      <th>addr_state_VT</th>
      <th>addr_state_WA</th>
      <th>addr_state_WI</th>
      <th>addr_state_WV</th>
      <th>addr_state_WY</th>
      <th>initial_list_status_w</th>
      <th>application_type_Joint App</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1174390</th>
      <td>15000.0</td>
      <td>60</td>
      <td>15.61</td>
      <td>361.67</td>
      <td>9.96</td>
      <td>2002</td>
      <td>24.0</td>
      <td>1.0</td>
      <td>49.5</td>
      <td>30.0</td>
      <td>2.1438</td>
      <td>1.0</td>
      <td>4.9031</td>
      <td>707.0</td>
      <td>3.3883</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>304775</th>
      <td>14000.0</td>
      <td>60</td>
      <td>10.42</td>
      <td>300.37</td>
      <td>1.73</td>
      <td>2006</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>61.0</td>
      <td>19.0</td>
      <td>1.3075</td>
      <td>1.0</td>
      <td>5.3979</td>
      <td>747.0</td>
      <td>4.4535</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1714051</th>
      <td>3600.0</td>
      <td>36</td>
      <td>15.99</td>
      <td>126.55</td>
      <td>28.63</td>
      <td>2002</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>63.3</td>
      <td>37.0</td>
      <td>2.4758</td>
      <td>0.0</td>
      <td>5.0253</td>
      <td>682.0</td>
      <td>4.4527</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1485291</th>
      <td>25600.0</td>
      <td>36</td>
      <td>14.49</td>
      <td>881.06</td>
      <td>16.90</td>
      <td>1985</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>69.9</td>
      <td>15.0</td>
      <td>0.9295</td>
      <td>0.0</td>
      <td>5.1206</td>
      <td>687.0</td>
      <td>4.3912</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>365102</th>
      <td>21000.0</td>
      <td>60</td>
      <td>18.06</td>
      <td>533.95</td>
      <td>20.48</td>
      <td>2005</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>47.6</td>
      <td>28.0</td>
      <td>2.0141</td>
      <td>0.0</td>
      <td>4.7993</td>
      <td>687.0</td>
      <td>4.3768</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Summary Plots
One of the most basic questions we might ask of a model is: What features have the biggest impact on predictions?

This concept is called feature importance.

There are multiple ways to measure feature importance. Some approaches answer subtly different versions of the question above. Other approaches have documented shortcomings.

Shap's Summary Plots are created on the basis of permutation importance. Compared to most other approaches, permutation importance is:

- fast to calculate,
- widely used and understood, and
- consistent with properties we would want a feature importance measure to have.

Link for detailed explanation: https://www.kaggle.com/dansbecker/permutation-importance    


Permutation importance is great because it created simple numeric measures to see which features mattered to a model. This helped us make comparisons between features easily, and you can present the resulting graphs to non-technical audiences.

But it doesn't tell you how each features matter. If a feature has medium permutation importance, that could mean it has

- a large effect for a few predictions, but no effect in general, or
- a medium effect for all predictions.

SHAP summary plots give us a birds-eye view of feature importance and what is driving it. Below we have plotted the <strong> Summary Plot </strong>. 
The <strong>"Feature Number"</strong> corresponds to the <strong>column number</strong> in our Test dataset.


```python
%%time
import shap
shap.initjs()
# compute SHAP values
explainer = shap.Explainer(my_model)
shap_values = explainer(X_test)
#summary_plot
shap.plots.beeswarm(shap_values)
```


<div align='center'><img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAWCAYAAAA1vze2AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAdxJREFUeNq0Vt1Rg0AQJjcpgBJiBWIFkgoMFYhPPAIVECogPuYpdJBYgXQQrMCUkA50V7+d2ZwXuXPGm9khHLu3f9+3l1nkWNvtNqfHLgpfQ1EUS3tz5nAQ0+NIsiAZSc6eDlI8M3J00B/mDuUKDk6kfOebAgW3pkdD0pFcODGW4gKKvOrAUm04MA4QDt1OEIXU9hDigfS5rC1eS5T90gltck1Xrizo257kgySZcNRzgCSxCvgiE9nckPJo2b/B2AcEkk2OwL8bD8gmOKR1GPbaCUqxEgTq0tLvgb6zfo7+DgYGkkWL2tqLDV4RSITfbHPPfJKIrWz4nJQTMPAWA7IbD6imcNaDeDfgk+4No+wZr40BL3g9eQJJCFqRQ54KiSt72lsLpE3o3MCBSxDuq4yOckU2hKXRuwBH3OyMR4g1UpyTYw6mlmBqNdUXRM1NfyF5EPI6JkcpIDBIX8jX6DR/6ckAZJ0wEAdLR8DEk6OfC1Pp8BKo6TQIwPJbvJ6toK5lmuvJoRtfK6Ym1iRYIarRo2UyYHvRN5qpakR3yoizWrouoyuXXQqI185LCw07op5ZyCRGL99h24InP0e9xdQukEKVmhzrqZuRIfwISB//cP3Wk3f8f/yR+BRgAHu00HjLcEQBAAAAAElFTkSuQmCC' /></div><script charset='utf-8'>!function(t){function e(r){if(n[r])return n[r].exports;var i=n[r]={i:r,l:!1,exports:{}};return t[r].call(i.exports,i,i.exports,e),i.l=!0,i.exports}var n={};return e.m=t,e.c=n,e.i=function(t){return t},e.d=function(t,n,r){e.o(t,n)||Object.defineProperty(t,n,{configurable:!1,enumerable:!0,get:r})},e.n=function(t){var n=t&&t.__esModule?function(){return t.default}:function(){return t};return e.d(n,"a",n),n},e.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},e.p="",e(e.s=410)}([function(t,e,n){"use strict";function r(t,e,n,r,o,a,u,c){if(i(e),!t){var s;if(void 0===e)s=new Error("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");else{var l=[n,r,o,a,u,c],f=0;s=new Error(e.replace(/%s/g,function(){return l[f++]})),s.name="Invariant Violation"}throw s.framesToPop=1,s}}var i=function(t){};t.exports=r},function(t,e,n){"use strict";var r=n(8),i=r;t.exports=i},function(t,e,n){"use strict";function r(t){for(var e=arguments.length-1,n="Minified React error #"+t+"; visit http://facebook.github.io/react/docs/error-decoder.html?invariant="+t,r=0;r<e;r++)n+="&args[]="+encodeURIComponent(arguments[r+1]);n+=" for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";var i=new Error(n);throw i.name="Invariant Violation",i.framesToPop=1,i}t.exports=r},function(t,e,n){"use strict";function r(t){if(null===t||void 0===t)throw new TypeError("Object.assign cannot be called with null or undefined");return Object(t)}function i(){try{if(!Object.assign)return!1;var t=new String("abc");if(t[5]="de","5"===Object.getOwnPropertyNames(t)[0])return!1;for(var e={},n=0;n<10;n++)e["_"+String.fromCharCode(n)]=n;var r=Object.getOwnPropertyNames(e).map(function(t){return e[t]});if("0123456789"!==r.join(""))return!1;var i={};return"abcdefghijklmnopqrst".split("").forEach(function(t){i[t]=t}),"abcdefghijklmnopqrst"===Object.keys(Object.assign({},i)).join("")}catch(t){return!1}}/*
object-assign
(c) Sindre Sorhus
@license MIT
*/
var o=Object.getOwnPropertySymbols,a=Object.prototype.hasOwnProperty,u=Object.prototype.propertyIsEnumerable;t.exports=i()?Object.assign:function(t,e){for(var n,i,c=r(t),s=1;s<arguments.length;s++){n=Object(arguments[s]);for(var l in n)a.call(n,l)&&(c[l]=n[l]);if(o){i=o(n);for(var f=0;f<i.length;f++)u.call(n,i[f])&&(c[i[f]]=n[i[f]])}}return c}},function(t,e,n){"use strict";function r(t,e){return 1===t.nodeType&&t.getAttribute(d)===String(e)||8===t.nodeType&&t.nodeValue===" react-text: "+e+" "||8===t.nodeType&&t.nodeValue===" react-empty: "+e+" "}function i(t){for(var e;e=t._renderedComponent;)t=e;return t}function o(t,e){var n=i(t);n._hostNode=e,e[g]=n}function a(t){var e=t._hostNode;e&&(delete e[g],t._hostNode=null)}function u(t,e){if(!(t._flags&v.hasCachedChildNodes)){var n=t._renderedChildren,a=e.firstChild;t:for(var u in n)if(n.hasOwnProperty(u)){var c=n[u],s=i(c)._domID;if(0!==s){for(;null!==a;a=a.nextSibling)if(r(a,s)){o(c,a);continue t}f("32",s)}}t._flags|=v.hasCachedChildNodes}}function c(t){if(t[g])return t[g];for(var e=[];!t[g];){if(e.push(t),!t.parentNode)return null;t=t.parentNode}for(var n,r;t&&(r=t[g]);t=e.pop())n=r,e.length&&u(r,t);return n}function s(t){var e=c(t);return null!=e&&e._hostNode===t?e:null}function l(t){if(void 0===t._hostNode?f("33"):void 0,t._hostNode)return t._hostNode;for(var e=[];!t._hostNode;)e.push(t),t._hostParent?void 0:f("34"),t=t._hostParent;for(;e.length;t=e.pop())u(t,t._hostNode);return t._hostNode}var f=n(2),p=n(21),h=n(157),d=(n(0),p.ID_ATTRIBUTE_NAME),v=h,g="__reactInternalInstance$"+Math.random().toString(36).slice(2),m={getClosestInstanceFromNode:c,getInstanceFromNode:s,getNodeFromInstance:l,precacheChildNodes:u,precacheNode:o,uncacheNode:a};t.exports=m},function(t,e,n){"use strict";function r(t,e,n,a){function u(e){return t(e=new Date(+e)),e}return u.floor=u,u.ceil=function(n){return t(n=new Date(n-1)),e(n,1),t(n),n},u.round=function(t){var e=u(t),n=u.ceil(t);return t-e<n-t?e:n},u.offset=function(t,n){return e(t=new Date(+t),null==n?1:Math.floor(n)),t},u.range=function(n,r,i){var o=[];if(n=u.ceil(n),i=null==i?1:Math.floor(i),!(n<r&&i>0))return o;do o.push(new Date(+n));while(e(n,i),t(n),n<r);return o},u.filter=function(n){return r(function(e){if(e>=e)for(;t(e),!n(e);)e.setTime(e-1)},function(t,r){if(t>=t)for(;--r>=0;)for(;e(t,1),!n(t););})},n&&(u.count=function(e,r){return i.setTime(+e),o.setTime(+r),t(i),t(o),Math.floor(n(i,o))},u.every=function(t){return t=Math.floor(t),isFinite(t)&&t>0?t>1?u.filter(a?function(e){return a(e)%t===0}:function(e){return u.count(0,e)%t===0}):u:null}),u}e.a=r;var i=new Date,o=new Date},function(t,e,n){"use strict";var r=!("undefined"==typeof window||!window.document||!window.document.createElement),i={canUseDOM:r,canUseWorkers:"undefined"!=typeof Worker,canUseEventListeners:r&&!(!window.addEventListener&&!window.attachEvent),canUseViewport:r&&!!window.screen,isInWorker:!r};t.exports=i},function(t,e,n){"use strict";function r(t,e){this._groups=t,this._parents=e}function i(){return new r([[document.documentElement]],D)}var o=n(272),a=n(273),u=n(261),c=n(255),s=n(131),l=n(260),f=n(265),p=n(268),h=n(275),d=n(253),v=n(267),g=n(266),m=n(274),y=n(259),_=n(258),b=n(252),x=n(276),w=n(269),C=n(254),M=n(277),k=n(262),E=n(270),T=n(264),S=n(251),P=n(263),N=n(271),A=n(256),O=n(70),I=n(257);n.d(e,"c",function(){return D}),e.b=r;var D=[null];r.prototype=i.prototype={constructor:r,select:o.a,selectAll:a.a,filter:u.a,data:c.a,enter:s.a,exit:l.a,merge:f.a,order:p.a,sort:h.a,call:d.a,nodes:v.a,node:g.a,size:m.a,empty:y.a,each:_.a,attr:b.a,style:x.a,property:w.a,classed:C.a,text:M.a,html:k.a,raise:E.a,lower:T.a,append:S.a,insert:P.a,remove:N.a,datum:A.a,on:O.c,dispatch:I.a},e.a=i},function(t,e,n){"use strict";function r(t){return function(){return t}}var i=function(){};i.thatReturns=r,i.thatReturnsFalse=r(!1),i.thatReturnsTrue=r(!0),i.thatReturnsNull=r(null),i.thatReturnsThis=function(){return this},i.thatReturnsArgument=function(t){return t},t.exports=i},function(t,e,n){"use strict";var r=null;t.exports={debugTool:r}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(59);n.d(e,"color",function(){return r.a}),n.d(e,"rgb",function(){return r.b}),n.d(e,"hsl",function(){return r.c});var i=n(210);n.d(e,"lab",function(){return i.a}),n.d(e,"hcl",function(){return i.b});var o=n(209);n.d(e,"cubehelix",function(){return o.a})},function(t,e,n){"use strict";function r(){T.ReactReconcileTransaction&&x?void 0:l("123")}function i(){this.reinitializeTransaction(),this.dirtyComponentsLength=null,this.callbackQueue=p.getPooled(),this.reconcileTransaction=T.ReactReconcileTransaction.getPooled(!0)}function o(t,e,n,i,o,a){return r(),x.batchedUpdates(t,e,n,i,o,a)}function a(t,e){return t._mountOrder-e._mountOrder}function u(t){var e=t.dirtyComponentsLength;e!==m.length?l("124",e,m.length):void 0,m.sort(a),y++;for(var n=0;n<e;n++){var r=m[n],i=r._pendingCallbacks;r._pendingCallbacks=null;var o;if(d.logTopLevelRenders){var u=r;r._currentElement.type.isReactTopLevelWrapper&&(u=r._renderedComponent),o="React update: "+u.getName(),console.time(o)}if(v.performUpdateIfNecessary(r,t.reconcileTransaction,y),o&&console.timeEnd(o),i)for(var c=0;c<i.length;c++)t.callbackQueue.enqueue(i[c],r.getPublicInstance())}}function c(t){return r(),x.isBatchingUpdates?(m.push(t),void(null==t._updateBatchNumber&&(t._updateBatchNumber=y+1))):void x.batchedUpdates(c,t)}function s(t,e){x.isBatchingUpdates?void 0:l("125"),_.enqueue(t,e),b=!0}var l=n(2),f=n(3),p=n(155),h=n(17),d=n(160),v=n(24),g=n(53),m=(n(0),[]),y=0,_=p.getPooled(),b=!1,x=null,w={initialize:function(){this.dirtyComponentsLength=m.length},close:function(){this.dirtyComponentsLength!==m.length?(m.splice(0,this.dirtyComponentsLength),k()):m.length=0}},C={initialize:function(){this.callbackQueue.reset()},close:function(){this.callbackQueue.notifyAll()}},M=[w,C];f(i.prototype,g,{getTransactionWrappers:function(){return M},destructor:function(){this.dirtyComponentsLength=null,p.release(this.callbackQueue),this.callbackQueue=null,T.ReactReconcileTransaction.release(this.reconcileTransaction),this.reconcileTransaction=null},perform:function(t,e,n){return g.perform.call(this,this.reconcileTransaction.perform,this.reconcileTransaction,t,e,n)}}),h.addPoolingTo(i);var k=function(){for(;m.length||b;){if(m.length){var t=i.getPooled();t.perform(u,null,t),i.release(t)}if(b){b=!1;var e=_;_=p.getPooled(),e.notifyAll(),p.release(e)}}},E={injectReconcileTransaction:function(t){t?void 0:l("126"),T.ReactReconcileTransaction=t},injectBatchingStrategy:function(t){t?void 0:l("127"),"function"!=typeof t.batchedUpdates?l("128"):void 0,"boolean"!=typeof t.isBatchingUpdates?l("129"):void 0,x=t}},T={ReactReconcileTransaction:null,batchedUpdates:o,enqueueUpdate:c,flushBatchedUpdates:k,injection:E,asap:s};t.exports=T},function(t,e,n){"use strict";var r=n(102);n.d(e,"c",function(){return r.a});var i=n(18);n.d(e,"f",function(){return i.a});var o=n(103);n.d(e,"d",function(){return o.a});var a=(n(185),n(104),n(105),n(186),n(197),n(198),n(108),n(188),n(189),n(190),n(191),n(106),n(192),n(193),n(57));n.d(e,"e",function(){return a.a});var u=n(107);n.d(e,"g",function(){return u.a});var c=(n(194),n(195),n(196),n(109));n.d(e,"a",function(){return c.a}),n.d(e,"b",function(){return c.b});n(110),n(111),n(199)},function(t,e,n){"use strict";n.d(e,"e",function(){return r}),n.d(e,"d",function(){return i}),n.d(e,"c",function(){return o}),n.d(e,"b",function(){return a}),n.d(e,"a",function(){return u});var r=1e3,i=6e4,o=36e5,a=864e5,u=6048e5},function(t,e,n){"use strict";function r(t,e,n,r){this.dispatchConfig=t,this._targetInst=e,this.nativeEvent=n;var i=this.constructor.Interface;for(var o in i)if(i.hasOwnProperty(o)){var u=i[o];u?this[o]=u(n):"target"===o?this.target=r:this[o]=n[o]}var c=null!=n.defaultPrevented?n.defaultPrevented:n.returnValue===!1;return c?this.isDefaultPrevented=a.thatReturnsTrue:this.isDefaultPrevented=a.thatReturnsFalse,this.isPropagationStopped=a.thatReturnsFalse,this}var i=n(3),o=n(17),a=n(8),u=(n(1),"function"==typeof Proxy,["dispatchConfig","_targetInst","nativeEvent","isDefaultPrevented","isPropagationStopped","_dispatchListeners","_dispatchInstances"]),c={type:null,target:null,currentTarget:a.thatReturnsNull,eventPhase:null,bubbles:null,cancelable:null,timeStamp:function(t){return t.timeStamp||Date.now()},defaultPrevented:null,isTrusted:null};i(r.prototype,{preventDefault:function(){this.defaultPrevented=!0;var t=this.nativeEvent;t&&(t.preventDefault?t.preventDefault():"unknown"!=typeof t.returnValue&&(t.returnValue=!1),this.isDefaultPrevented=a.thatReturnsTrue)},stopPropagation:function(){var t=this.nativeEvent;t&&(t.stopPropagation?t.stopPropagation():"unknown"!=typeof t.cancelBubble&&(t.cancelBubble=!0),this.isPropagationStopped=a.thatReturnsTrue)},persist:function(){this.isPersistent=a.thatReturnsTrue},isPersistent:a.thatReturnsFalse,destructor:function(){var t=this.constructor.Interface;for(var e in t)this[e]=null;for(var n=0;n<u.length;n++)this[u[n]]=null}}),r.Interface=c,r.augmentClass=function(t,e){var n=this,r=function(){};r.prototype=n.prototype;var a=new r;i(a,t.prototype),t.prototype=a,t.prototype.constructor=t,t.Interface=i({},n.Interface,e),t.augmentClass=n.augmentClass,o.addPoolingTo(t,o.fourArgumentPooler)},o.addPoolingTo(r,o.fourArgumentPooler),t.exports=r},function(t,e,n){"use strict";var r={current:null};t.exports=r},function(t,e,n){"use strict";n.d(e,"a",function(){return i}),n.d(e,"b",function(){return o});var r=Array.prototype,i=r.map,o=r.slice},function(t,e,n){"use strict";var r=n(2),i=(n(0),function(t){var e=this;if(e.instancePool.length){var n=e.instancePool.pop();return e.call(n,t),n}return new e(t)}),o=function(t,e){var n=this;if(n.instancePool.length){var r=n.instancePool.pop();return n.call(r,t,e),r}return new n(t,e)},a=function(t,e,n){var r=this;if(r.instancePool.length){var i=r.instancePool.pop();return r.call(i,t,e,n),i}return new r(t,e,n)},u=function(t,e,n,r){var i=this;if(i.instancePool.length){var o=i.instancePool.pop();return i.call(o,t,e,n,r),o}return new i(t,e,n,r)},c=function(t){var e=this;t instanceof e?void 0:r("25"),t.destructor(),e.instancePool.length<e.poolSize&&e.instancePool.push(t)},s=10,l=i,f=function(t,e){var n=t;return n.instancePool=[],n.getPooled=e||l,n.poolSize||(n.poolSize=s),n.release=c,n},p={addPoolingTo:f,oneArgumentPooler:i,twoArgumentPooler:o,threeArgumentPooler:a,fourArgumentPooler:u};t.exports=p},function(t,e,n){"use strict";e.a=function(t,e){return t<e?-1:t>e?1:t>=e?0:NaN}},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";function r(t){if(g){var e=t.node,n=t.children;if(n.length)for(var r=0;r<n.length;r++)m(e,n[r],null);else null!=t.html?f(e,t.html):null!=t.text&&h(e,t.text)}}function i(t,e){t.parentNode.replaceChild(e.node,t),r(e)}function o(t,e){g?t.children.push(e):t.node.appendChild(e.node)}function a(t,e){g?t.html=e:f(t.node,e)}function u(t,e){g?t.text=e:h(t.node,e)}function c(){return this.node.nodeName}function s(t){return{node:t,children:[],html:null,text:null,toString:c}}var l=n(82),f=n(55),p=n(90),h=n(171),d=1,v=11,g="undefined"!=typeof document&&"number"==typeof document.documentMode||"undefined"!=typeof navigator&&"string"==typeof navigator.userAgent&&/\bEdge\/\d/.test(navigator.userAgent),m=p(function(t,e,n){e.node.nodeType===v||e.node.nodeType===d&&"object"===e.node.nodeName.toLowerCase()&&(null==e.node.namespaceURI||e.node.namespaceURI===l.html)?(r(e),t.insertBefore(e.node,n)):(t.insertBefore(e.node,n),r(e))});s.insertTreeBefore=m,s.replaceChildWithTree=i,s.queueChild=o,s.queueHTML=a,s.queueText=u,t.exports=s},function(t,e,n){"use strict";function r(t,e){return(t&e)===e}var i=n(2),o=(n(0),{MUST_USE_PROPERTY:1,HAS_BOOLEAN_VALUE:4,HAS_NUMERIC_VALUE:8,HAS_POSITIVE_NUMERIC_VALUE:24,HAS_OVERLOADED_BOOLEAN_VALUE:32,injectDOMPropertyConfig:function(t){var e=o,n=t.Properties||{},a=t.DOMAttributeNamespaces||{},c=t.DOMAttributeNames||{},s=t.DOMPropertyNames||{},l=t.DOMMutationMethods||{};t.isCustomAttribute&&u._isCustomAttributeFunctions.push(t.isCustomAttribute);for(var f in n){u.properties.hasOwnProperty(f)?i("48",f):void 0;var p=f.toLowerCase(),h=n[f],d={attributeName:p,attributeNamespace:null,propertyName:f,mutationMethod:null,mustUseProperty:r(h,e.MUST_USE_PROPERTY),hasBooleanValue:r(h,e.HAS_BOOLEAN_VALUE),hasNumericValue:r(h,e.HAS_NUMERIC_VALUE),hasPositiveNumericValue:r(h,e.HAS_POSITIVE_NUMERIC_VALUE),hasOverloadedBooleanValue:r(h,e.HAS_OVERLOADED_BOOLEAN_VALUE)};if(d.hasBooleanValue+d.hasNumericValue+d.hasOverloadedBooleanValue<=1?void 0:i("50",f),c.hasOwnProperty(f)){var v=c[f];d.attributeName=v}a.hasOwnProperty(f)&&(d.attributeNamespace=a[f]),s.hasOwnProperty(f)&&(d.propertyName=s[f]),l.hasOwnProperty(f)&&(d.mutationMethod=l[f]),u.properties[f]=d}}}),a=":A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD",u={ID_ATTRIBUTE_NAME:"data-reactid",ROOT_ATTRIBUTE_NAME:"data-reactroot",ATTRIBUTE_NAME_START_CHAR:a,ATTRIBUTE_NAME_CHAR:a+"\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040",properties:{},getPossibleStandardName:null,_isCustomAttributeFunctions:[],isCustomAttribute:function(t){for(var e=0;e<u._isCustomAttributeFunctions.length;e++){var n=u._isCustomAttributeFunctions[e];if(n(t))return!0}return!1},injection:o};t.exports=u},function(t,e,n){"use strict";function r(t){return"button"===t||"input"===t||"select"===t||"textarea"===t}function i(t,e,n){switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":return!(!n.disabled||!r(e));default:return!1}}var o=n(2),a=n(83),u=n(50),c=n(87),s=n(165),l=n(166),f=(n(0),{}),p=null,h=function(t,e){t&&(u.executeDispatchesInOrder(t,e),t.isPersistent()||t.constructor.release(t))},d=function(t){return h(t,!0)},v=function(t){return h(t,!1)},g=function(t){return"."+t._rootNodeID},m={injection:{injectEventPluginOrder:a.injectEventPluginOrder,injectEventPluginsByName:a.injectEventPluginsByName},putListener:function(t,e,n){"function"!=typeof n?o("94",e,typeof n):void 0;var r=g(t),i=f[e]||(f[e]={});i[r]=n;var u=a.registrationNameModules[e];u&&u.didPutListener&&u.didPutListener(t,e,n)},getListener:function(t,e){var n=f[e];if(i(e,t._currentElement.type,t._currentElement.props))return null;var r=g(t);return n&&n[r]},deleteListener:function(t,e){var n=a.registrationNameModules[e];n&&n.willDeleteListener&&n.willDeleteListener(t,e);var r=f[e];if(r){var i=g(t);delete r[i]}},deleteAllListeners:function(t){var e=g(t);for(var n in f)if(f.hasOwnProperty(n)&&f[n][e]){var r=a.registrationNameModules[n];r&&r.willDeleteListener&&r.willDeleteListener(t,n),delete f[n][e]}},extractEvents:function(t,e,n,r){for(var i,o=a.plugins,u=0;u<o.length;u++){var c=o[u];if(c){var l=c.extractEvents(t,e,n,r);l&&(i=s(i,l))}}return i},enqueueEvents:function(t){t&&(p=s(p,t))},processEventQueue:function(t){var e=p;p=null,t?l(e,d):l(e,v),p?o("95"):void 0,c.rethrowCaughtError()},__purge:function(){f={}},__getListenerBank:function(){return f}};t.exports=m},function(t,e,n){"use strict";function r(t,e,n){var r=e.dispatchConfig.phasedRegistrationNames[n];return m(t,r)}function i(t,e,n){var i=r(t,n,e);i&&(n._dispatchListeners=v(n._dispatchListeners,i),n._dispatchInstances=v(n._dispatchInstances,t))}function o(t){t&&t.dispatchConfig.phasedRegistrationNames&&d.traverseTwoPhase(t._targetInst,i,t)}function a(t){if(t&&t.dispatchConfig.phasedRegistrationNames){var e=t._targetInst,n=e?d.getParentInstance(e):null;d.traverseTwoPhase(n,i,t)}}function u(t,e,n){if(n&&n.dispatchConfig.registrationName){var r=n.dispatchConfig.registrationName,i=m(t,r);i&&(n._dispatchListeners=v(n._dispatchListeners,i),n._dispatchInstances=v(n._dispatchInstances,t))}}function c(t){t&&t.dispatchConfig.registrationName&&u(t._targetInst,null,t)}function s(t){g(t,o)}function l(t){g(t,a)}function f(t,e,n,r){d.traverseEnterLeave(n,r,u,t,e)}function p(t){g(t,c)}var h=n(22),d=n(50),v=n(165),g=n(166),m=(n(1),h.getListener),y={accumulateTwoPhaseDispatches:s,accumulateTwoPhaseDispatchesSkipTarget:l,accumulateDirectDispatches:p,accumulateEnterLeaveDispatches:f};t.exports=y},function(t,e,n){"use strict";function r(){i.attachRefs(this,this._currentElement)}var i=n(368),o=(n(9),n(1),{mountComponent:function(t,e,n,i,o,a){var u=t.mountComponent(e,n,i,o,a);return t._currentElement&&null!=t._currentElement.ref&&e.getReactMountReady().enqueue(r,t),u},getHostNode:function(t){return t.getHostNode()},unmountComponent:function(t,e){i.detachRefs(t,t._currentElement),t.unmountComponent(e)},receiveComponent:function(t,e,n,o){var a=t._currentElement;if(e!==a||o!==t._context){var u=i.shouldUpdateRefs(a,e);u&&i.detachRefs(t,a),t.receiveComponent(e,n,o),u&&t._currentElement&&null!=t._currentElement.ref&&n.getReactMountReady().enqueue(r,t)}},performUpdateIfNecessary:function(t,e,n){t._updateBatchNumber===n&&t.performUpdateIfNecessary(e)}});t.exports=o},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o=n(93),a={view:function(t){if(t.view)return t.view;var e=o(t);if(e.window===e)return e;var n=e.ownerDocument;return n?n.defaultView||n.parentWindow:window},detail:function(t){return t.detail||0}};i.augmentClass(r,a),t.exports=r},function(t,e,n){"use strict";var r=n(3),i=n(401),o=n(97),a=n(406),u=n(402),c=n(403),s=n(27),l=n(404),f=n(407),p=n(408),h=(n(1),s.createElement),d=s.createFactory,v=s.cloneElement,g=r,m={Children:{map:i.map,forEach:i.forEach,count:i.count,toArray:i.toArray,only:p},Component:o,PureComponent:a,createElement:h,cloneElement:v,isValidElement:s.isValidElement,PropTypes:l,createClass:u.createClass,createFactory:d,createMixin:function(t){return t},DOM:c,version:f,__spread:g};t.exports=m},function(t,e,n){"use strict";function r(t){return void 0!==t.ref}function i(t){return void 0!==t.key}var o=n(3),a=n(15),u=(n(1),n(176),Object.prototype.hasOwnProperty),c=n(174),s={key:!0,ref:!0,__self:!0,__source:!0},l=function(t,e,n,r,i,o,a){var u={$$typeof:c,type:t,key:e,ref:n,props:a,_owner:o};return u};l.createElement=function(t,e,n){var o,c={},f=null,p=null,h=null,d=null;if(null!=e){r(e)&&(p=e.ref),i(e)&&(f=""+e.key),h=void 0===e.__self?null:e.__self,d=void 0===e.__source?null:e.__source;for(o in e)u.call(e,o)&&!s.hasOwnProperty(o)&&(c[o]=e[o])}var v=arguments.length-2;if(1===v)c.children=n;else if(v>1){for(var g=Array(v),m=0;m<v;m++)g[m]=arguments[m+2];c.children=g}if(t&&t.defaultProps){var y=t.defaultProps;for(o in y)void 0===c[o]&&(c[o]=y[o])}return l(t,f,p,h,d,a.current,c)},l.createFactory=function(t){var e=l.createElement.bind(null,t);return e.type=t,e},l.cloneAndReplaceKey=function(t,e){var n=l(t.type,e,t.ref,t._self,t._source,t._owner,t.props);return n},l.cloneElement=function(t,e,n){var c,f=o({},t.props),p=t.key,h=t.ref,d=t._self,v=t._source,g=t._owner;if(null!=e){r(e)&&(h=e.ref,g=a.current),i(e)&&(p=""+e.key);var m;t.type&&t.type.defaultProps&&(m=t.type.defaultProps);for(c in e)u.call(e,c)&&!s.hasOwnProperty(c)&&(void 0===e[c]&&void 0!==m?f[c]=m[c]:f[c]=e[c])}var y=arguments.length-2;if(1===y)f.children=n;else if(y>1){for(var _=Array(y),b=0;b<y;b++)_[b]=arguments[b+2];f.children=_}return l(t.type,p,h,d,v,g,f)},l.isValidElement=function(t){return"object"==typeof t&&null!==t&&t.$$typeof===c},t.exports=l},function(t,e,n){"use strict";function r(t){for(var e=arguments.length-1,n="Minified React error #"+t+"; visit http://facebook.github.io/react/docs/error-decoder.html?invariant="+t,r=0;r<e;r++)n+="&args[]="+encodeURIComponent(arguments[r+1]);n+=" for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";var i=new Error(n);throw i.name="Invariant Violation",i.framesToPop=1,i}t.exports=r},function(t,e,n){"use strict";e.a=function(t){return null===t?NaN:+t}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(211);n.d(e,"formatDefaultLocale",function(){return r.a}),n.d(e,"format",function(){return r.b}),n.d(e,"formatPrefix",function(){return r.c});var i=n(117);n.d(e,"formatLocale",function(){return i.a});var o=n(115);n.d(e,"formatSpecifier",function(){return o.a});var a=n(215);n.d(e,"precisionFixed",function(){return a.a});var u=n(216);n.d(e,"precisionPrefix",function(){return u.a});var c=n(217);n.d(e,"precisionRound",function(){return c.a})},function(t,e,n){"use strict";var r=n(63);n.d(e,"b",function(){return r.a});var i=(n(118),n(62),n(119),n(121),n(43));n.d(e,"a",function(){return i.a});var o=(n(122),n(223));n.d(e,"c",function(){return o.a});var a=(n(124),n(225),n(227),n(123),n(220),n(221),n(219),n(218));n.d(e,"d",function(){return a.a});n(222)},function(t,e,n){"use strict";function r(t,e){return function(n){return t+n*e}}function i(t,e,n){return t=Math.pow(t,n),e=Math.pow(e,n)-t,n=1/n,function(r){return Math.pow(t+r*e,n)}}function o(t,e){var i=e-t;return i?r(t,i>180||i<-180?i-360*Math.round(i/360):i):n.i(c.a)(isNaN(t)?e:t)}function a(t){return 1===(t=+t)?u:function(e,r){return r-e?i(e,r,t):n.i(c.a)(isNaN(e)?r:e)}}function u(t,e){var i=e-t;return i?r(t,i):n.i(c.a)(isNaN(t)?e:t)}var c=n(120);e.b=o,e.c=a,e.a=u},function(t,e,n){"use strict";e.a=function(t){return t.match(/.{6}/g).map(function(t){return"#"+t})}},function(t,e,n){"use strict";function r(t){var e=t.domain;return t.ticks=function(t){var r=e();return n.i(o.a)(r[0],r[r.length-1],null==t?10:t)},t.tickFormat=function(t,r){return n.i(c.a)(e(),t,r)},t.nice=function(r){var i=e(),a=i.length-1,u=null==r?10:r,c=i[0],s=i[a],l=n.i(o.b)(c,s,u);return l&&(l=n.i(o.b)(Math.floor(c/l)*l,Math.ceil(s/l)*l,u),i[0]=Math.floor(c/l)*l,i[a]=Math.ceil(s/l)*l,e(i)),t},t}function i(){var t=n.i(u.a)(u.b,a.a);return t.copy=function(){return n.i(u.c)(t,i())},r(t)}var o=n(12),a=n(31),u=n(45),c=n(243);e.b=r,e.a=i},function(t,e,n){"use strict";n.d(e,"a",function(){return r}),n.d(e,"b",function(){return i}),n.d(e,"d",function(){return o}),n.d(e,"c",function(){return a});var r=1e-12,i=Math.PI,o=i/2,a=2*i},function(t,e,n){"use strict";e.a=function(t,e){if((r=t.length)>1)for(var n,r,i=1,o=t[e[0]],a=o.length;i<r;++i){n=o,o=t[e[i]];for(var u=0;u<a;++u)o[u][1]+=o[u][0]=isNaN(n[u][1])?n[u][0]:n[u][1]}}},function(t,e,n){"use strict";e.a=function(t){for(var e=t.length,n=new Array(e);--e>=0;)n[e]=e;return n}},function(t,e,n){"use strict";var r={};t.exports=r},function(t,e,n){(function(t,r){var i;(function(){function o(t,e){return t.set(e[0],e[1]),t}function a(t,e){return t.add(e),t}function u(t,e,n){switch(n.length){case 0:return t.call(e);case 1:return t.call(e,n[0]);case 2:return t.call(e,n[0],n[1]);case 3:return t.call(e,n[0],n[1],n[2])}return t.apply(e,n)}function c(t,e,n,r){for(var i=-1,o=null==t?0:t.length;++i<o;){var a=t[i];e(r,a,n(a),t)}return r}function s(t,e){for(var n=-1,r=null==t?0:t.length;++n<r&&e(t[n],n,t)!==!1;);return t}function l(t,e){for(var n=null==t?0:t.length;n--&&e(t[n],n,t)!==!1;);return t}function f(t,e){for(var n=-1,r=null==t?0:t.length;++n<r;)if(!e(t[n],n,t))return!1;return!0}function p(t,e){for(var n=-1,r=null==t?0:t.length,i=0,o=[];++n<r;){var a=t[n];e(a,n,t)&&(o[i++]=a)}return o}function h(t,e){var n=null==t?0:t.length;return!!n&&M(t,e,0)>-1}function d(t,e,n){for(var r=-1,i=null==t?0:t.length;++r<i;)if(n(e,t[r]))return!0;return!1}function v(t,e){for(var n=-1,r=null==t?0:t.length,i=Array(r);++n<r;)i[n]=e(t[n],n,t);return i}function g(t,e){for(var n=-1,r=e.length,i=t.length;++n<r;)t[i+n]=e[n];return t}function m(t,e,n,r){var i=-1,o=null==t?0:t.length;for(r&&o&&(n=t[++i]);++i<o;)n=e(n,t[i],i,t);return n}function y(t,e,n,r){var i=null==t?0:t.length;for(r&&i&&(n=t[--i]);i--;)n=e(n,t[i],i,t);return n}function _(t,e){for(var n=-1,r=null==t?0:t.length;++n<r;)if(e(t[n],n,t))return!0;return!1}function b(t){return t.split("")}function x(t){return t.match(ze)||[]}function w(t,e,n){var r;return n(t,function(t,n,i){if(e(t,n,i))return r=n,!1}),r}function C(t,e,n,r){for(var i=t.length,o=n+(r?1:-1);r?o--:++o<i;)if(e(t[o],o,t))return o;return-1}function M(t,e,n){return e===e?Z(t,e,n):C(t,E,n)}function k(t,e,n,r){for(var i=n-1,o=t.length;++i<o;)if(r(t[i],e))return i;return-1}function E(t){return t!==t}function T(t,e){var n=null==t?0:t.length;return n?O(t,e)/n:Ut}function S(t){return function(e){return null==e?it:e[t]}}function P(t){return function(e){return null==t?it:t[e]}}function N(t,e,n,r,i){return i(t,function(t,i,o){n=r?(r=!1,t):e(n,t,i,o)}),n}function A(t,e){var n=t.length;for(t.sort(e);n--;)t[n]=t[n].value;return t}function O(t,e){for(var n,r=-1,i=t.length;++r<i;){var o=e(t[r]);o!==it&&(n=n===it?o:n+o)}return n}function I(t,e){for(var n=-1,r=Array(t);++n<t;)r[n]=e(n);return r}function D(t,e){return v(e,function(e){return[e,t[e]]})}function R(t){return function(e){return t(e)}}function L(t,e){return v(e,function(e){return t[e]})}function U(t,e){return t.has(e)}function F(t,e){for(var n=-1,r=t.length;++n<r&&M(e,t[n],0)>-1;);return n}function j(t,e){for(var n=t.length;n--&&M(e,t[n],0)>-1;);return n}function B(t,e){for(var n=t.length,r=0;n--;)t[n]===e&&++r;return r}function W(t){return"\\"+nr[t]}function V(t,e){return null==t?it:t[e]}function z(t){return Kn.test(t)}function H(t){return Gn.test(t)}function q(t){for(var e,n=[];!(e=t.next()).done;)n.push(e.value);return n}function Y(t){var e=-1,n=Array(t.size);return t.forEach(function(t,r){n[++e]=[r,t]}),n}function K(t,e){return function(n){return t(e(n))}}function G(t,e){for(var n=-1,r=t.length,i=0,o=[];++n<r;){var a=t[n];a!==e&&a!==ft||(t[n]=ft,o[i++]=n)}return o}function $(t){var e=-1,n=Array(t.size);return t.forEach(function(t){n[++e]=t}),n}function X(t){var e=-1,n=Array(t.size);return t.forEach(function(t){n[++e]=[t,t]}),n}function Z(t,e,n){for(var r=n-1,i=t.length;++r<i;)if(t[r]===e)return r;return-1}function Q(t,e,n){for(var r=n+1;r--;)if(t[r]===e)return r;return r}function J(t){return z(t)?et(t):_r(t)}function tt(t){return z(t)?nt(t):b(t)}function et(t){for(var e=qn.lastIndex=0;qn.test(t);)++e;return e}function nt(t){return t.match(qn)||[]}function rt(t){return t.match(Yn)||[]}var it,ot="4.17.4",at=200,ut="Unsupported core-js use. Try https://npms.io/search?q=ponyfill.",ct="Expected a function",st="__lodash_hash_undefined__",lt=500,ft="__lodash_placeholder__",pt=1,ht=2,dt=4,vt=1,gt=2,mt=1,yt=2,_t=4,bt=8,xt=16,wt=32,Ct=64,Mt=128,kt=256,Et=512,Tt=30,St="...",Pt=800,Nt=16,At=1,Ot=2,It=3,Dt=1/0,Rt=9007199254740991,Lt=1.7976931348623157e308,Ut=NaN,Ft=4294967295,jt=Ft-1,Bt=Ft>>>1,Wt=[["ary",Mt],["bind",mt],["bindKey",yt],["curry",bt],["curryRight",xt],["flip",Et],["partial",wt],["partialRight",Ct],["rearg",kt]],Vt="[object Arguments]",zt="[object Array]",Ht="[object AsyncFunction]",qt="[object Boolean]",Yt="[object Date]",Kt="[object DOMException]",Gt="[object Error]",$t="[object Function]",Xt="[object GeneratorFunction]",Zt="[object Map]",Qt="[object Number]",Jt="[object Null]",te="[object Object]",ee="[object Promise]",ne="[object Proxy]",re="[object RegExp]",ie="[object Set]",oe="[object String]",ae="[object Symbol]",ue="[object Undefined]",ce="[object WeakMap]",se="[object WeakSet]",le="[object ArrayBuffer]",fe="[object DataView]",pe="[object Float32Array]",he="[object Float64Array]",de="[object Int8Array]",ve="[object Int16Array]",ge="[object Int32Array]",me="[object Uint8Array]",ye="[object Uint8ClampedArray]",_e="[object Uint16Array]",be="[object Uint32Array]",xe=/\b__p \+= '';/g,we=/\b(__p \+=) '' \+/g,Ce=/(__e\(.*?\)|\b__t\)) \+\n'';/g,Me=/&(?:amp|lt|gt|quot|#39);/g,ke=/[&<>"']/g,Ee=RegExp(Me.source),Te=RegExp(ke.source),Se=/<%-([\s\S]+?)%>/g,Pe=/<%([\s\S]+?)%>/g,Ne=/<%=([\s\S]+?)%>/g,Ae=/\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/,Oe=/^\w*$/,Ie=/^\./,De=/[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g,Re=/[\\^$.*+?()[\]{}|]/g,Le=RegExp(Re.source),Ue=/^\s+|\s+$/g,Fe=/^\s+/,je=/\s+$/,Be=/\{(?:\n\/\* \[wrapped with .+\] \*\/)?\n?/,We=/\{\n\/\* \[wrapped with (.+)\] \*/,Ve=/,? & /,ze=/[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g,He=/\\(\\)?/g,qe=/\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g,Ye=/\w*$/,Ke=/^[-+]0x[0-9a-f]+$/i,Ge=/^0b[01]+$/i,$e=/^\[object .+?Constructor\]$/,Xe=/^0o[0-7]+$/i,Ze=/^(?:0|[1-9]\d*)$/,Qe=/[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g,Je=/($^)/,tn=/['\n\r\u2028\u2029\\]/g,en="\\ud800-\\udfff",nn="\\u0300-\\u036f",rn="\\ufe20-\\ufe2f",on="\\u20d0-\\u20ff",an=nn+rn+on,un="\\u2700-\\u27bf",cn="a-z\\xdf-\\xf6\\xf8-\\xff",sn="\\xac\\xb1\\xd7\\xf7",ln="\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf",fn="\\u2000-\\u206f",pn=" \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000",hn="A-Z\\xc0-\\xd6\\xd8-\\xde",dn="\\ufe0e\\ufe0f",vn=sn+ln+fn+pn,gn="['’]",mn="["+en+"]",yn="["+vn+"]",_n="["+an+"]",bn="\\d+",xn="["+un+"]",wn="["+cn+"]",Cn="[^"+en+vn+bn+un+cn+hn+"]",Mn="\\ud83c[\\udffb-\\udfff]",kn="(?:"+_n+"|"+Mn+")",En="[^"+en+"]",Tn="(?:\\ud83c[\\udde6-\\uddff]){2}",Sn="[\\ud800-\\udbff][\\udc00-\\udfff]",Pn="["+hn+"]",Nn="\\u200d",An="(?:"+wn+"|"+Cn+")",On="(?:"+Pn+"|"+Cn+")",In="(?:"+gn+"(?:d|ll|m|re|s|t|ve))?",Dn="(?:"+gn+"(?:D|LL|M|RE|S|T|VE))?",Rn=kn+"?",Ln="["+dn+"]?",Un="(?:"+Nn+"(?:"+[En,Tn,Sn].join("|")+")"+Ln+Rn+")*",Fn="\\d*(?:(?:1st|2nd|3rd|(?![123])\\dth)\\b)",jn="\\d*(?:(?:1ST|2ND|3RD|(?![123])\\dTH)\\b)",Bn=Ln+Rn+Un,Wn="(?:"+[xn,Tn,Sn].join("|")+")"+Bn,Vn="(?:"+[En+_n+"?",_n,Tn,Sn,mn].join("|")+")",zn=RegExp(gn,"g"),Hn=RegExp(_n,"g"),qn=RegExp(Mn+"(?="+Mn+")|"+Vn+Bn,"g"),Yn=RegExp([Pn+"?"+wn+"+"+In+"(?="+[yn,Pn,"$"].join("|")+")",On+"+"+Dn+"(?="+[yn,Pn+An,"$"].join("|")+")",Pn+"?"+An+"+"+In,Pn+"+"+Dn,jn,Fn,bn,Wn].join("|"),"g"),Kn=RegExp("["+Nn+en+an+dn+"]"),Gn=/[a-z][A-Z]|[A-Z]{2,}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/,$n=["Array","Buffer","DataView","Date","Error","Float32Array","Float64Array","Function","Int8Array","Int16Array","Int32Array","Map","Math","Object","Promise","RegExp","Set","String","Symbol","TypeError","Uint8Array","Uint8ClampedArray","Uint16Array","Uint32Array","WeakMap","_","clearTimeout","isFinite","parseInt","setTimeout"],Xn=-1,Zn={};Zn[pe]=Zn[he]=Zn[de]=Zn[ve]=Zn[ge]=Zn[me]=Zn[ye]=Zn[_e]=Zn[be]=!0,Zn[Vt]=Zn[zt]=Zn[le]=Zn[qt]=Zn[fe]=Zn[Yt]=Zn[Gt]=Zn[$t]=Zn[Zt]=Zn[Qt]=Zn[te]=Zn[re]=Zn[ie]=Zn[oe]=Zn[ce]=!1;var Qn={};Qn[Vt]=Qn[zt]=Qn[le]=Qn[fe]=Qn[qt]=Qn[Yt]=Qn[pe]=Qn[he]=Qn[de]=Qn[ve]=Qn[ge]=Qn[Zt]=Qn[Qt]=Qn[te]=Qn[re]=Qn[ie]=Qn[oe]=Qn[ae]=Qn[me]=Qn[ye]=Qn[_e]=Qn[be]=!0,Qn[Gt]=Qn[$t]=Qn[ce]=!1;var Jn={"À":"A","Á":"A","Â":"A","Ã":"A","Ä":"A","Å":"A","à":"a","á":"a","â":"a","ã":"a","ä":"a","å":"a","Ç":"C","ç":"c","Ð":"D","ð":"d","È":"E","É":"E","Ê":"E","Ë":"E","è":"e","é":"e","ê":"e","ë":"e","Ì":"I","Í":"I","Î":"I","Ï":"I","ì":"i","í":"i","î":"i","ï":"i","Ñ":"N","ñ":"n","Ò":"O","Ó":"O","Ô":"O","Õ":"O","Ö":"O","Ø":"O","ò":"o","ó":"o","ô":"o","õ":"o","ö":"o","ø":"o","Ù":"U","Ú":"U","Û":"U","Ü":"U","ù":"u","ú":"u","û":"u","ü":"u","Ý":"Y","ý":"y","ÿ":"y","Æ":"Ae","æ":"ae","Þ":"Th","þ":"th","ß":"ss","Ā":"A","Ă":"A","Ą":"A","ā":"a","ă":"a","ą":"a","Ć":"C","Ĉ":"C","Ċ":"C","Č":"C","ć":"c","ĉ":"c","ċ":"c","č":"c","Ď":"D","Đ":"D","ď":"d","đ":"d","Ē":"E","Ĕ":"E","Ė":"E","Ę":"E","Ě":"E","ē":"e","ĕ":"e","ė":"e","ę":"e","ě":"e","Ĝ":"G","Ğ":"G","Ġ":"G","Ģ":"G","ĝ":"g","ğ":"g","ġ":"g","ģ":"g","Ĥ":"H","Ħ":"H","ĥ":"h","ħ":"h","Ĩ":"I","Ī":"I","Ĭ":"I","Į":"I","İ":"I","ĩ":"i","ī":"i","ĭ":"i","į":"i","ı":"i","Ĵ":"J","ĵ":"j","Ķ":"K","ķ":"k","ĸ":"k","Ĺ":"L","Ļ":"L","Ľ":"L","Ŀ":"L","Ł":"L","ĺ":"l","ļ":"l","ľ":"l","ŀ":"l","ł":"l","Ń":"N","Ņ":"N","Ň":"N","Ŋ":"N","ń":"n","ņ":"n","ň":"n","ŋ":"n","Ō":"O","Ŏ":"O","Ő":"O","ō":"o","ŏ":"o","ő":"o","Ŕ":"R","Ŗ":"R","Ř":"R","ŕ":"r","ŗ":"r","ř":"r","Ś":"S","Ŝ":"S","Ş":"S","Š":"S","ś":"s","ŝ":"s","ş":"s","š":"s","Ţ":"T","Ť":"T","Ŧ":"T","ţ":"t","ť":"t","ŧ":"t","Ũ":"U","Ū":"U","Ŭ":"U","Ů":"U","Ű":"U","Ų":"U","ũ":"u","ū":"u","ŭ":"u","ů":"u","ű":"u","ų":"u","Ŵ":"W","ŵ":"w","Ŷ":"Y","ŷ":"y","Ÿ":"Y","Ź":"Z","Ż":"Z","Ž":"Z","ź":"z","ż":"z","ž":"z","Ĳ":"IJ",
"ĳ":"ij","Œ":"Oe","œ":"oe","ŉ":"'n","ſ":"s"},tr={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"},er={"&amp;":"&","&lt;":"<","&gt;":">","&quot;":'"',"&#39;":"'"},nr={"\\":"\\","'":"'","\n":"n","\r":"r","\u2028":"u2028","\u2029":"u2029"},rr=parseFloat,ir=parseInt,or="object"==typeof t&&t&&t.Object===Object&&t,ar="object"==typeof self&&self&&self.Object===Object&&self,ur=or||ar||Function("return this")(),cr="object"==typeof e&&e&&!e.nodeType&&e,sr=cr&&"object"==typeof r&&r&&!r.nodeType&&r,lr=sr&&sr.exports===cr,fr=lr&&or.process,pr=function(){try{return fr&&fr.binding&&fr.binding("util")}catch(t){}}(),hr=pr&&pr.isArrayBuffer,dr=pr&&pr.isDate,vr=pr&&pr.isMap,gr=pr&&pr.isRegExp,mr=pr&&pr.isSet,yr=pr&&pr.isTypedArray,_r=S("length"),br=P(Jn),xr=P(tr),wr=P(er),Cr=function t(e){function n(t){if(sc(t)&&!xp(t)&&!(t instanceof b)){if(t instanceof i)return t;if(bl.call(t,"__wrapped__"))return aa(t)}return new i(t)}function r(){}function i(t,e){this.__wrapped__=t,this.__actions__=[],this.__chain__=!!e,this.__index__=0,this.__values__=it}function b(t){this.__wrapped__=t,this.__actions__=[],this.__dir__=1,this.__filtered__=!1,this.__iteratees__=[],this.__takeCount__=Ft,this.__views__=[]}function P(){var t=new b(this.__wrapped__);return t.__actions__=Bi(this.__actions__),t.__dir__=this.__dir__,t.__filtered__=this.__filtered__,t.__iteratees__=Bi(this.__iteratees__),t.__takeCount__=this.__takeCount__,t.__views__=Bi(this.__views__),t}function Z(){if(this.__filtered__){var t=new b(this);t.__dir__=-1,t.__filtered__=!0}else t=this.clone(),t.__dir__*=-1;return t}function et(){var t=this.__wrapped__.value(),e=this.__dir__,n=xp(t),r=e<0,i=n?t.length:0,o=No(0,i,this.__views__),a=o.start,u=o.end,c=u-a,s=r?u:a-1,l=this.__iteratees__,f=l.length,p=0,h=Xl(c,this.__takeCount__);if(!n||!r&&i==c&&h==c)return xi(t,this.__actions__);var d=[];t:for(;c--&&p<h;){s+=e;for(var v=-1,g=t[s];++v<f;){var m=l[v],y=m.iteratee,_=m.type,b=y(g);if(_==Ot)g=b;else if(!b){if(_==At)continue t;break t}}d[p++]=g}return d}function nt(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function ze(){this.__data__=uf?uf(null):{},this.size=0}function en(t){var e=this.has(t)&&delete this.__data__[t];return this.size-=e?1:0,e}function nn(t){var e=this.__data__;if(uf){var n=e[t];return n===st?it:n}return bl.call(e,t)?e[t]:it}function rn(t){var e=this.__data__;return uf?e[t]!==it:bl.call(e,t)}function on(t,e){var n=this.__data__;return this.size+=this.has(t)?0:1,n[t]=uf&&e===it?st:e,this}function an(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function un(){this.__data__=[],this.size=0}function cn(t){var e=this.__data__,n=In(e,t);if(n<0)return!1;var r=e.length-1;return n==r?e.pop():Dl.call(e,n,1),--this.size,!0}function sn(t){var e=this.__data__,n=In(e,t);return n<0?it:e[n][1]}function ln(t){return In(this.__data__,t)>-1}function fn(t,e){var n=this.__data__,r=In(n,t);return r<0?(++this.size,n.push([t,e])):n[r][1]=e,this}function pn(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function hn(){this.size=0,this.__data__={hash:new nt,map:new(nf||an),string:new nt}}function dn(t){var e=Eo(this,t).delete(t);return this.size-=e?1:0,e}function vn(t){return Eo(this,t).get(t)}function gn(t){return Eo(this,t).has(t)}function mn(t,e){var n=Eo(this,t),r=n.size;return n.set(t,e),this.size+=n.size==r?0:1,this}function yn(t){var e=-1,n=null==t?0:t.length;for(this.__data__=new pn;++e<n;)this.add(t[e])}function _n(t){return this.__data__.set(t,st),this}function bn(t){return this.__data__.has(t)}function xn(t){var e=this.__data__=new an(t);this.size=e.size}function wn(){this.__data__=new an,this.size=0}function Cn(t){var e=this.__data__,n=e.delete(t);return this.size=e.size,n}function Mn(t){return this.__data__.get(t)}function kn(t){return this.__data__.has(t)}function En(t,e){var n=this.__data__;if(n instanceof an){var r=n.__data__;if(!nf||r.length<at-1)return r.push([t,e]),this.size=++n.size,this;n=this.__data__=new pn(r)}return n.set(t,e),this.size=n.size,this}function Tn(t,e){var n=xp(t),r=!n&&bp(t),i=!n&&!r&&Cp(t),o=!n&&!r&&!i&&Sp(t),a=n||r||i||o,u=a?I(t.length,hl):[],c=u.length;for(var s in t)!e&&!bl.call(t,s)||a&&("length"==s||i&&("offset"==s||"parent"==s)||o&&("buffer"==s||"byteLength"==s||"byteOffset"==s)||Fo(s,c))||u.push(s);return u}function Sn(t){var e=t.length;return e?t[ni(0,e-1)]:it}function Pn(t,e){return na(Bi(t),jn(e,0,t.length))}function Nn(t){return na(Bi(t))}function An(t,e,n){(n===it||$u(t[e],n))&&(n!==it||e in t)||Un(t,e,n)}function On(t,e,n){var r=t[e];bl.call(t,e)&&$u(r,n)&&(n!==it||e in t)||Un(t,e,n)}function In(t,e){for(var n=t.length;n--;)if($u(t[n][0],e))return n;return-1}function Dn(t,e,n,r){return _f(t,function(t,i,o){e(r,t,n(t),o)}),r}function Rn(t,e){return t&&Wi(e,Hc(e),t)}function Ln(t,e){return t&&Wi(e,qc(e),t)}function Un(t,e,n){"__proto__"==e&&Fl?Fl(t,e,{configurable:!0,enumerable:!0,value:n,writable:!0}):t[e]=n}function Fn(t,e){for(var n=-1,r=e.length,i=al(r),o=null==t;++n<r;)i[n]=o?it:Wc(t,e[n]);return i}function jn(t,e,n){return t===t&&(n!==it&&(t=t<=n?t:n),e!==it&&(t=t>=e?t:e)),t}function Bn(t,e,n,r,i,o){var a,u=e&pt,c=e&ht,l=e&dt;if(n&&(a=i?n(t,r,i,o):n(t)),a!==it)return a;if(!cc(t))return t;var f=xp(t);if(f){if(a=Io(t),!u)return Bi(t,a)}else{var p=Af(t),h=p==$t||p==Xt;if(Cp(t))return Si(t,u);if(p==te||p==Vt||h&&!i){if(a=c||h?{}:Do(t),!u)return c?zi(t,Ln(a,t)):Vi(t,Rn(a,t))}else{if(!Qn[p])return i?t:{};a=Ro(t,p,Bn,u)}}o||(o=new xn);var d=o.get(t);if(d)return d;o.set(t,a);var v=l?c?wo:xo:c?qc:Hc,g=f?it:v(t);return s(g||t,function(r,i){g&&(i=r,r=t[i]),On(a,i,Bn(r,e,n,i,t,o))}),a}function Wn(t){var e=Hc(t);return function(n){return Vn(n,t,e)}}function Vn(t,e,n){var r=n.length;if(null==t)return!r;for(t=fl(t);r--;){var i=n[r],o=e[i],a=t[i];if(a===it&&!(i in t)||!o(a))return!1}return!0}function qn(t,e,n){if("function"!=typeof t)throw new dl(ct);return Df(function(){t.apply(it,n)},e)}function Yn(t,e,n,r){var i=-1,o=h,a=!0,u=t.length,c=[],s=e.length;if(!u)return c;n&&(e=v(e,R(n))),r?(o=d,a=!1):e.length>=at&&(o=U,a=!1,e=new yn(e));t:for(;++i<u;){var l=t[i],f=null==n?l:n(l);if(l=r||0!==l?l:0,a&&f===f){for(var p=s;p--;)if(e[p]===f)continue t;c.push(l)}else o(e,f,r)||c.push(l)}return c}function Kn(t,e){var n=!0;return _f(t,function(t,r,i){return n=!!e(t,r,i)}),n}function Gn(t,e,n){for(var r=-1,i=t.length;++r<i;){var o=t[r],a=e(o);if(null!=a&&(u===it?a===a&&!bc(a):n(a,u)))var u=a,c=o}return c}function Jn(t,e,n,r){var i=t.length;for(n=Ec(n),n<0&&(n=-n>i?0:i+n),r=r===it||r>i?i:Ec(r),r<0&&(r+=i),r=n>r?0:Tc(r);n<r;)t[n++]=e;return t}function tr(t,e){var n=[];return _f(t,function(t,r,i){e(t,r,i)&&n.push(t)}),n}function er(t,e,n,r,i){var o=-1,a=t.length;for(n||(n=Uo),i||(i=[]);++o<a;){var u=t[o];e>0&&n(u)?e>1?er(u,e-1,n,r,i):g(i,u):r||(i[i.length]=u)}return i}function nr(t,e){return t&&xf(t,e,Hc)}function or(t,e){return t&&wf(t,e,Hc)}function ar(t,e){return p(e,function(e){return oc(t[e])})}function cr(t,e){e=Ei(e,t);for(var n=0,r=e.length;null!=t&&n<r;)t=t[ra(e[n++])];return n&&n==r?t:it}function sr(t,e,n){var r=e(t);return xp(t)?r:g(r,n(t))}function fr(t){return null==t?t===it?ue:Jt:Ul&&Ul in fl(t)?Po(t):Xo(t)}function pr(t,e){return t>e}function _r(t,e){return null!=t&&bl.call(t,e)}function Cr(t,e){return null!=t&&e in fl(t)}function kr(t,e,n){return t>=Xl(e,n)&&t<$l(e,n)}function Er(t,e,n){for(var r=n?d:h,i=t[0].length,o=t.length,a=o,u=al(o),c=1/0,s=[];a--;){var l=t[a];a&&e&&(l=v(l,R(e))),c=Xl(l.length,c),u[a]=!n&&(e||i>=120&&l.length>=120)?new yn(a&&l):it}l=t[0];var f=-1,p=u[0];t:for(;++f<i&&s.length<c;){var g=l[f],m=e?e(g):g;if(g=n||0!==g?g:0,!(p?U(p,m):r(s,m,n))){for(a=o;--a;){var y=u[a];if(!(y?U(y,m):r(t[a],m,n)))continue t}p&&p.push(m),s.push(g)}}return s}function Tr(t,e,n,r){return nr(t,function(t,i,o){e(r,n(t),i,o)}),r}function Sr(t,e,n){e=Ei(e,t),t=Qo(t,e);var r=null==t?t:t[ra(ka(e))];return null==r?it:u(r,t,n)}function Pr(t){return sc(t)&&fr(t)==Vt}function Nr(t){return sc(t)&&fr(t)==le}function Ar(t){return sc(t)&&fr(t)==Yt}function Or(t,e,n,r,i){return t===e||(null==t||null==e||!sc(t)&&!sc(e)?t!==t&&e!==e:Ir(t,e,n,r,Or,i))}function Ir(t,e,n,r,i,o){var a=xp(t),u=xp(e),c=a?zt:Af(t),s=u?zt:Af(e);c=c==Vt?te:c,s=s==Vt?te:s;var l=c==te,f=s==te,p=c==s;if(p&&Cp(t)){if(!Cp(e))return!1;a=!0,l=!1}if(p&&!l)return o||(o=new xn),a||Sp(t)?mo(t,e,n,r,i,o):yo(t,e,c,n,r,i,o);if(!(n&vt)){var h=l&&bl.call(t,"__wrapped__"),d=f&&bl.call(e,"__wrapped__");if(h||d){var v=h?t.value():t,g=d?e.value():e;return o||(o=new xn),i(v,g,n,r,o)}}return!!p&&(o||(o=new xn),_o(t,e,n,r,i,o))}function Dr(t){return sc(t)&&Af(t)==Zt}function Rr(t,e,n,r){var i=n.length,o=i,a=!r;if(null==t)return!o;for(t=fl(t);i--;){var u=n[i];if(a&&u[2]?u[1]!==t[u[0]]:!(u[0]in t))return!1}for(;++i<o;){u=n[i];var c=u[0],s=t[c],l=u[1];if(a&&u[2]){if(s===it&&!(c in t))return!1}else{var f=new xn;if(r)var p=r(s,l,c,t,e,f);if(!(p===it?Or(l,s,vt|gt,r,f):p))return!1}}return!0}function Lr(t){if(!cc(t)||zo(t))return!1;var e=oc(t)?El:$e;return e.test(ia(t))}function Ur(t){return sc(t)&&fr(t)==re}function Fr(t){return sc(t)&&Af(t)==ie}function jr(t){return sc(t)&&uc(t.length)&&!!Zn[fr(t)]}function Br(t){return"function"==typeof t?t:null==t?Ds:"object"==typeof t?xp(t)?Yr(t[0],t[1]):qr(t):Vs(t)}function Wr(t){if(!Ho(t))return Gl(t);var e=[];for(var n in fl(t))bl.call(t,n)&&"constructor"!=n&&e.push(n);return e}function Vr(t){if(!cc(t))return $o(t);var e=Ho(t),n=[];for(var r in t)("constructor"!=r||!e&&bl.call(t,r))&&n.push(r);return n}function zr(t,e){return t<e}function Hr(t,e){var n=-1,r=Xu(t)?al(t.length):[];return _f(t,function(t,i,o){r[++n]=e(t,i,o)}),r}function qr(t){var e=To(t);return 1==e.length&&e[0][2]?Yo(e[0][0],e[0][1]):function(n){return n===t||Rr(n,t,e)}}function Yr(t,e){return Bo(t)&&qo(e)?Yo(ra(t),e):function(n){var r=Wc(n,t);return r===it&&r===e?zc(n,t):Or(e,r,vt|gt)}}function Kr(t,e,n,r,i){t!==e&&xf(e,function(o,a){if(cc(o))i||(i=new xn),Gr(t,e,a,n,Kr,r,i);else{var u=r?r(t[a],o,a+"",t,e,i):it;u===it&&(u=o),An(t,a,u)}},qc)}function Gr(t,e,n,r,i,o,a){var u=t[n],c=e[n],s=a.get(c);if(s)return void An(t,n,s);var l=o?o(u,c,n+"",t,e,a):it,f=l===it;if(f){var p=xp(c),h=!p&&Cp(c),d=!p&&!h&&Sp(c);l=c,p||h||d?xp(u)?l=u:Zu(u)?l=Bi(u):h?(f=!1,l=Si(c,!0)):d?(f=!1,l=Ri(c,!0)):l=[]:mc(c)||bp(c)?(l=u,bp(u)?l=Pc(u):(!cc(u)||r&&oc(u))&&(l=Do(c))):f=!1}f&&(a.set(c,l),i(l,c,r,o,a),a.delete(c)),An(t,n,l)}function $r(t,e){var n=t.length;if(n)return e+=e<0?n:0,Fo(e,n)?t[e]:it}function Xr(t,e,n){var r=-1;e=v(e.length?e:[Ds],R(ko()));var i=Hr(t,function(t,n,i){var o=v(e,function(e){return e(t)});return{criteria:o,index:++r,value:t}});return A(i,function(t,e){return Ui(t,e,n)})}function Zr(t,e){return Qr(t,e,function(e,n){return zc(t,n)})}function Qr(t,e,n){for(var r=-1,i=e.length,o={};++r<i;){var a=e[r],u=cr(t,a);n(u,a)&&ci(o,Ei(a,t),u)}return o}function Jr(t){return function(e){return cr(e,t)}}function ti(t,e,n,r){var i=r?k:M,o=-1,a=e.length,u=t;for(t===e&&(e=Bi(e)),n&&(u=v(t,R(n)));++o<a;)for(var c=0,s=e[o],l=n?n(s):s;(c=i(u,l,c,r))>-1;)u!==t&&Dl.call(u,c,1),Dl.call(t,c,1);return t}function ei(t,e){for(var n=t?e.length:0,r=n-1;n--;){var i=e[n];if(n==r||i!==o){var o=i;Fo(i)?Dl.call(t,i,1):yi(t,i)}}return t}function ni(t,e){return t+zl(Jl()*(e-t+1))}function ri(t,e,n,r){for(var i=-1,o=$l(Vl((e-t)/(n||1)),0),a=al(o);o--;)a[r?o:++i]=t,t+=n;return a}function ii(t,e){var n="";if(!t||e<1||e>Rt)return n;do e%2&&(n+=t),e=zl(e/2),e&&(t+=t);while(e);return n}function oi(t,e){return Rf(Zo(t,e,Ds),t+"")}function ai(t){return Sn(rs(t))}function ui(t,e){var n=rs(t);return na(n,jn(e,0,n.length))}function ci(t,e,n,r){if(!cc(t))return t;e=Ei(e,t);for(var i=-1,o=e.length,a=o-1,u=t;null!=u&&++i<o;){var c=ra(e[i]),s=n;if(i!=a){var l=u[c];s=r?r(l,c,u):it,s===it&&(s=cc(l)?l:Fo(e[i+1])?[]:{})}On(u,c,s),u=u[c]}return t}function si(t){return na(rs(t))}function li(t,e,n){var r=-1,i=t.length;e<0&&(e=-e>i?0:i+e),n=n>i?i:n,n<0&&(n+=i),i=e>n?0:n-e>>>0,e>>>=0;for(var o=al(i);++r<i;)o[r]=t[r+e];return o}function fi(t,e){var n;return _f(t,function(t,r,i){return n=e(t,r,i),!n}),!!n}function pi(t,e,n){var r=0,i=null==t?r:t.length;if("number"==typeof e&&e===e&&i<=Bt){for(;r<i;){var o=r+i>>>1,a=t[o];null!==a&&!bc(a)&&(n?a<=e:a<e)?r=o+1:i=o}return i}return hi(t,e,Ds,n)}function hi(t,e,n,r){e=n(e);for(var i=0,o=null==t?0:t.length,a=e!==e,u=null===e,c=bc(e),s=e===it;i<o;){var l=zl((i+o)/2),f=n(t[l]),p=f!==it,h=null===f,d=f===f,v=bc(f);if(a)var g=r||d;else g=s?d&&(r||p):u?d&&p&&(r||!h):c?d&&p&&!h&&(r||!v):!h&&!v&&(r?f<=e:f<e);g?i=l+1:o=l}return Xl(o,jt)}function di(t,e){for(var n=-1,r=t.length,i=0,o=[];++n<r;){var a=t[n],u=e?e(a):a;if(!n||!$u(u,c)){var c=u;o[i++]=0===a?0:a}}return o}function vi(t){return"number"==typeof t?t:bc(t)?Ut:+t}function gi(t){if("string"==typeof t)return t;if(xp(t))return v(t,gi)+"";if(bc(t))return mf?mf.call(t):"";var e=t+"";return"0"==e&&1/t==-Dt?"-0":e}function mi(t,e,n){var r=-1,i=h,o=t.length,a=!0,u=[],c=u;if(n)a=!1,i=d;else if(o>=at){var s=e?null:Tf(t);if(s)return $(s);a=!1,i=U,c=new yn}else c=e?[]:u;t:for(;++r<o;){var l=t[r],f=e?e(l):l;if(l=n||0!==l?l:0,a&&f===f){for(var p=c.length;p--;)if(c[p]===f)continue t;e&&c.push(f),u.push(l)}else i(c,f,n)||(c!==u&&c.push(f),u.push(l))}return u}function yi(t,e){return e=Ei(e,t),t=Qo(t,e),null==t||delete t[ra(ka(e))]}function _i(t,e,n,r){return ci(t,e,n(cr(t,e)),r)}function bi(t,e,n,r){for(var i=t.length,o=r?i:-1;(r?o--:++o<i)&&e(t[o],o,t););return n?li(t,r?0:o,r?o+1:i):li(t,r?o+1:0,r?i:o)}function xi(t,e){var n=t;return n instanceof b&&(n=n.value()),m(e,function(t,e){return e.func.apply(e.thisArg,g([t],e.args))},n)}function wi(t,e,n){var r=t.length;if(r<2)return r?mi(t[0]):[];for(var i=-1,o=al(r);++i<r;)for(var a=t[i],u=-1;++u<r;)u!=i&&(o[i]=Yn(o[i]||a,t[u],e,n));return mi(er(o,1),e,n)}function Ci(t,e,n){for(var r=-1,i=t.length,o=e.length,a={};++r<i;){var u=r<o?e[r]:it;n(a,t[r],u)}return a}function Mi(t){return Zu(t)?t:[]}function ki(t){return"function"==typeof t?t:Ds}function Ei(t,e){return xp(t)?t:Bo(t,e)?[t]:Lf(Ac(t))}function Ti(t,e,n){var r=t.length;return n=n===it?r:n,!e&&n>=r?t:li(t,e,n)}function Si(t,e){if(e)return t.slice();var n=t.length,r=Nl?Nl(n):new t.constructor(n);return t.copy(r),r}function Pi(t){var e=new t.constructor(t.byteLength);return new Pl(e).set(new Pl(t)),e}function Ni(t,e){var n=e?Pi(t.buffer):t.buffer;return new t.constructor(n,t.byteOffset,t.byteLength)}function Ai(t,e,n){var r=e?n(Y(t),pt):Y(t);return m(r,o,new t.constructor)}function Oi(t){var e=new t.constructor(t.source,Ye.exec(t));return e.lastIndex=t.lastIndex,e}function Ii(t,e,n){var r=e?n($(t),pt):$(t);return m(r,a,new t.constructor)}function Di(t){return gf?fl(gf.call(t)):{}}function Ri(t,e){var n=e?Pi(t.buffer):t.buffer;return new t.constructor(n,t.byteOffset,t.length)}function Li(t,e){if(t!==e){var n=t!==it,r=null===t,i=t===t,o=bc(t),a=e!==it,u=null===e,c=e===e,s=bc(e);if(!u&&!s&&!o&&t>e||o&&a&&c&&!u&&!s||r&&a&&c||!n&&c||!i)return 1;if(!r&&!o&&!s&&t<e||s&&n&&i&&!r&&!o||u&&n&&i||!a&&i||!c)return-1}return 0}function Ui(t,e,n){for(var r=-1,i=t.criteria,o=e.criteria,a=i.length,u=n.length;++r<a;){var c=Li(i[r],o[r]);if(c){if(r>=u)return c;var s=n[r];return c*("desc"==s?-1:1)}}return t.index-e.index}function Fi(t,e,n,r){for(var i=-1,o=t.length,a=n.length,u=-1,c=e.length,s=$l(o-a,0),l=al(c+s),f=!r;++u<c;)l[u]=e[u];for(;++i<a;)(f||i<o)&&(l[n[i]]=t[i]);for(;s--;)l[u++]=t[i++];return l}function ji(t,e,n,r){for(var i=-1,o=t.length,a=-1,u=n.length,c=-1,s=e.length,l=$l(o-u,0),f=al(l+s),p=!r;++i<l;)f[i]=t[i];for(var h=i;++c<s;)f[h+c]=e[c];for(;++a<u;)(p||i<o)&&(f[h+n[a]]=t[i++]);return f}function Bi(t,e){var n=-1,r=t.length;for(e||(e=al(r));++n<r;)e[n]=t[n];return e}function Wi(t,e,n,r){var i=!n;n||(n={});for(var o=-1,a=e.length;++o<a;){var u=e[o],c=r?r(n[u],t[u],u,n,t):it;c===it&&(c=t[u]),i?Un(n,u,c):On(n,u,c)}return n}function Vi(t,e){return Wi(t,Pf(t),e)}function zi(t,e){return Wi(t,Nf(t),e)}function Hi(t,e){return function(n,r){var i=xp(n)?c:Dn,o=e?e():{};return i(n,t,ko(r,2),o)}}function qi(t){return oi(function(e,n){var r=-1,i=n.length,o=i>1?n[i-1]:it,a=i>2?n[2]:it;for(o=t.length>3&&"function"==typeof o?(i--,o):it,a&&jo(n[0],n[1],a)&&(o=i<3?it:o,i=1),e=fl(e);++r<i;){var u=n[r];u&&t(e,u,r,o)}return e})}function Yi(t,e){return function(n,r){if(null==n)return n;if(!Xu(n))return t(n,r);for(var i=n.length,o=e?i:-1,a=fl(n);(e?o--:++o<i)&&r(a[o],o,a)!==!1;);return n}}function Ki(t){return function(e,n,r){for(var i=-1,o=fl(e),a=r(e),u=a.length;u--;){var c=a[t?u:++i];if(n(o[c],c,o)===!1)break}return e}}function Gi(t,e,n){function r(){var e=this&&this!==ur&&this instanceof r?o:t;return e.apply(i?n:this,arguments)}var i=e&mt,o=Zi(t);return r}function $i(t){return function(e){e=Ac(e);var n=z(e)?tt(e):it,r=n?n[0]:e.charAt(0),i=n?Ti(n,1).join(""):e.slice(1);return r[t]()+i}}function Xi(t){return function(e){return m(Ps(ss(e).replace(zn,"")),t,"")}}function Zi(t){return function(){var e=arguments;switch(e.length){case 0:return new t;case 1:return new t(e[0]);case 2:return new t(e[0],e[1]);case 3:return new t(e[0],e[1],e[2]);case 4:return new t(e[0],e[1],e[2],e[3]);case 5:return new t(e[0],e[1],e[2],e[3],e[4]);case 6:return new t(e[0],e[1],e[2],e[3],e[4],e[5]);case 7:return new t(e[0],e[1],e[2],e[3],e[4],e[5],e[6])}var n=yf(t.prototype),r=t.apply(n,e);return cc(r)?r:n}}function Qi(t,e,n){function r(){for(var o=arguments.length,a=al(o),c=o,s=Mo(r);c--;)a[c]=arguments[c];var l=o<3&&a[0]!==s&&a[o-1]!==s?[]:G(a,s);if(o-=l.length,o<n)return so(t,e,eo,r.placeholder,it,a,l,it,it,n-o);var f=this&&this!==ur&&this instanceof r?i:t;return u(f,this,a)}var i=Zi(t);return r}function Ji(t){return function(e,n,r){var i=fl(e);if(!Xu(e)){var o=ko(n,3);e=Hc(e),n=function(t){return o(i[t],t,i)}}var a=t(e,n,r);return a>-1?i[o?e[a]:a]:it}}function to(t){return bo(function(e){var n=e.length,r=n,o=i.prototype.thru;for(t&&e.reverse();r--;){var a=e[r];if("function"!=typeof a)throw new dl(ct);if(o&&!u&&"wrapper"==Co(a))var u=new i([],!0)}for(r=u?r:n;++r<n;){a=e[r];var c=Co(a),s="wrapper"==c?Sf(a):it;u=s&&Vo(s[0])&&s[1]==(Mt|bt|wt|kt)&&!s[4].length&&1==s[9]?u[Co(s[0])].apply(u,s[3]):1==a.length&&Vo(a)?u[c]():u.thru(a)}return function(){var t=arguments,r=t[0];if(u&&1==t.length&&xp(r))return u.plant(r).value();for(var i=0,o=n?e[i].apply(this,t):r;++i<n;)o=e[i].call(this,o);return o}})}function eo(t,e,n,r,i,o,a,u,c,s){function l(){for(var m=arguments.length,y=al(m),_=m;_--;)y[_]=arguments[_];if(d)var b=Mo(l),x=B(y,b);if(r&&(y=Fi(y,r,i,d)),o&&(y=ji(y,o,a,d)),m-=x,d&&m<s){var w=G(y,b);return so(t,e,eo,l.placeholder,n,y,w,u,c,s-m)}var C=p?n:this,M=h?C[t]:t;return m=y.length,u?y=Jo(y,u):v&&m>1&&y.reverse(),f&&c<m&&(y.length=c),this&&this!==ur&&this instanceof l&&(M=g||Zi(M)),M.apply(C,y)}var f=e&Mt,p=e&mt,h=e&yt,d=e&(bt|xt),v=e&Et,g=h?it:Zi(t);return l}function no(t,e){return function(n,r){return Tr(n,t,e(r),{})}}function ro(t,e){return function(n,r){var i;if(n===it&&r===it)return e;if(n!==it&&(i=n),r!==it){if(i===it)return r;"string"==typeof n||"string"==typeof r?(n=gi(n),r=gi(r)):(n=vi(n),r=vi(r)),i=t(n,r)}return i}}function io(t){return bo(function(e){return e=v(e,R(ko())),oi(function(n){var r=this;return t(e,function(t){return u(t,r,n)})})})}function oo(t,e){e=e===it?" ":gi(e);var n=e.length;if(n<2)return n?ii(e,t):e;var r=ii(e,Vl(t/J(e)));return z(e)?Ti(tt(r),0,t).join(""):r.slice(0,t)}function ao(t,e,n,r){function i(){for(var e=-1,c=arguments.length,s=-1,l=r.length,f=al(l+c),p=this&&this!==ur&&this instanceof i?a:t;++s<l;)f[s]=r[s];for(;c--;)f[s++]=arguments[++e];return u(p,o?n:this,f)}var o=e&mt,a=Zi(t);return i}function uo(t){return function(e,n,r){return r&&"number"!=typeof r&&jo(e,n,r)&&(n=r=it),e=kc(e),n===it?(n=e,e=0):n=kc(n),r=r===it?e<n?1:-1:kc(r),ri(e,n,r,t)}}function co(t){return function(e,n){return"string"==typeof e&&"string"==typeof n||(e=Sc(e),n=Sc(n)),t(e,n)}}function so(t,e,n,r,i,o,a,u,c,s){var l=e&bt,f=l?a:it,p=l?it:a,h=l?o:it,d=l?it:o;e|=l?wt:Ct,e&=~(l?Ct:wt),e&_t||(e&=~(mt|yt));var v=[t,e,i,h,f,d,p,u,c,s],g=n.apply(it,v);return Vo(t)&&If(g,v),g.placeholder=r,ta(g,t,e)}function lo(t){var e=ll[t];return function(t,n){if(t=Sc(t),n=null==n?0:Xl(Ec(n),292)){var r=(Ac(t)+"e").split("e"),i=e(r[0]+"e"+(+r[1]+n));return r=(Ac(i)+"e").split("e"),+(r[0]+"e"+(+r[1]-n))}return e(t)}}function fo(t){return function(e){var n=Af(e);return n==Zt?Y(e):n==ie?X(e):D(e,t(e))}}function po(t,e,n,r,i,o,a,u){var c=e&yt;if(!c&&"function"!=typeof t)throw new dl(ct);var s=r?r.length:0;if(s||(e&=~(wt|Ct),r=i=it),a=a===it?a:$l(Ec(a),0),u=u===it?u:Ec(u),s-=i?i.length:0,e&Ct){var l=r,f=i;r=i=it}var p=c?it:Sf(t),h=[t,e,n,r,i,l,f,o,a,u];if(p&&Go(h,p),t=h[0],e=h[1],n=h[2],r=h[3],i=h[4],u=h[9]=h[9]===it?c?0:t.length:$l(h[9]-s,0),!u&&e&(bt|xt)&&(e&=~(bt|xt)),e&&e!=mt)d=e==bt||e==xt?Qi(t,e,u):e!=wt&&e!=(mt|wt)||i.length?eo.apply(it,h):ao(t,e,n,r);else var d=Gi(t,e,n);var v=p?Cf:If;return ta(v(d,h),t,e)}function ho(t,e,n,r){return t===it||$u(t,ml[n])&&!bl.call(r,n)?e:t}function vo(t,e,n,r,i,o){return cc(t)&&cc(e)&&(o.set(e,t),Kr(t,e,it,vo,o),o.delete(e)),t}function go(t){return mc(t)?it:t}function mo(t,e,n,r,i,o){var a=n&vt,u=t.length,c=e.length;if(u!=c&&!(a&&c>u))return!1;var s=o.get(t);if(s&&o.get(e))return s==e;var l=-1,f=!0,p=n&gt?new yn:it;for(o.set(t,e),o.set(e,t);++l<u;){var h=t[l],d=e[l];if(r)var v=a?r(d,h,l,e,t,o):r(h,d,l,t,e,o);if(v!==it){if(v)continue;f=!1;break}if(p){if(!_(e,function(t,e){if(!U(p,e)&&(h===t||i(h,t,n,r,o)))return p.push(e)})){f=!1;break}}else if(h!==d&&!i(h,d,n,r,o)){f=!1;break}}return o.delete(t),o.delete(e),f}function yo(t,e,n,r,i,o,a){switch(n){case fe:if(t.byteLength!=e.byteLength||t.byteOffset!=e.byteOffset)return!1;t=t.buffer,e=e.buffer;case le:return!(t.byteLength!=e.byteLength||!o(new Pl(t),new Pl(e)));case qt:case Yt:case Qt:return $u(+t,+e);case Gt:return t.name==e.name&&t.message==e.message;case re:case oe:return t==e+"";case Zt:var u=Y;case ie:var c=r&vt;if(u||(u=$),t.size!=e.size&&!c)return!1;var s=a.get(t);if(s)return s==e;r|=gt,a.set(t,e);var l=mo(u(t),u(e),r,i,o,a);return a.delete(t),l;case ae:if(gf)return gf.call(t)==gf.call(e)}return!1}function _o(t,e,n,r,i,o){var a=n&vt,u=xo(t),c=u.length,s=xo(e),l=s.length;if(c!=l&&!a)return!1;for(var f=c;f--;){var p=u[f];if(!(a?p in e:bl.call(e,p)))return!1}var h=o.get(t);if(h&&o.get(e))return h==e;var d=!0;o.set(t,e),o.set(e,t);for(var v=a;++f<c;){p=u[f];var g=t[p],m=e[p];if(r)var y=a?r(m,g,p,e,t,o):r(g,m,p,t,e,o);if(!(y===it?g===m||i(g,m,n,r,o):y)){d=!1;break}v||(v="constructor"==p)}if(d&&!v){var _=t.constructor,b=e.constructor;_!=b&&"constructor"in t&&"constructor"in e&&!("function"==typeof _&&_ instanceof _&&"function"==typeof b&&b instanceof b)&&(d=!1)}return o.delete(t),o.delete(e),d}function bo(t){return Rf(Zo(t,it,ma),t+"")}function xo(t){return sr(t,Hc,Pf)}function wo(t){return sr(t,qc,Nf)}function Co(t){for(var e=t.name+"",n=sf[e],r=bl.call(sf,e)?n.length:0;r--;){var i=n[r],o=i.func;if(null==o||o==t)return i.name}return e}function Mo(t){var e=bl.call(n,"placeholder")?n:t;return e.placeholder}function ko(){var t=n.iteratee||Rs;return t=t===Rs?Br:t,arguments.length?t(arguments[0],arguments[1]):t}function Eo(t,e){var n=t.__data__;return Wo(e)?n["string"==typeof e?"string":"hash"]:n.map}function To(t){for(var e=Hc(t),n=e.length;n--;){var r=e[n],i=t[r];e[n]=[r,i,qo(i)]}return e}function So(t,e){var n=V(t,e);return Lr(n)?n:it}function Po(t){var e=bl.call(t,Ul),n=t[Ul];try{t[Ul]=it;var r=!0}catch(t){}var i=Cl.call(t);return r&&(e?t[Ul]=n:delete t[Ul]),i}function No(t,e,n){for(var r=-1,i=n.length;++r<i;){var o=n[r],a=o.size;switch(o.type){case"drop":t+=a;break;case"dropRight":e-=a;break;case"take":e=Xl(e,t+a);break;case"takeRight":t=$l(t,e-a)}}return{start:t,end:e}}function Ao(t){var e=t.match(We);return e?e[1].split(Ve):[]}function Oo(t,e,n){e=Ei(e,t);for(var r=-1,i=e.length,o=!1;++r<i;){var a=ra(e[r]);if(!(o=null!=t&&n(t,a)))break;t=t[a]}return o||++r!=i?o:(i=null==t?0:t.length,!!i&&uc(i)&&Fo(a,i)&&(xp(t)||bp(t)))}function Io(t){var e=t.length,n=t.constructor(e);return e&&"string"==typeof t[0]&&bl.call(t,"index")&&(n.index=t.index,n.input=t.input),n}function Do(t){return"function"!=typeof t.constructor||Ho(t)?{}:yf(Al(t))}function Ro(t,e,n,r){var i=t.constructor;switch(e){case le:return Pi(t);case qt:case Yt:return new i(+t);case fe:return Ni(t,r);case pe:case he:case de:case ve:case ge:case me:case ye:case _e:case be:return Ri(t,r);case Zt:return Ai(t,r,n);case Qt:case oe:return new i(t);case re:return Oi(t);case ie:return Ii(t,r,n);case ae:return Di(t)}}function Lo(t,e){var n=e.length;if(!n)return t;var r=n-1;return e[r]=(n>1?"& ":"")+e[r],e=e.join(n>2?", ":" "),t.replace(Be,"{\n/* [wrapped with "+e+"] */\n")}function Uo(t){return xp(t)||bp(t)||!!(Rl&&t&&t[Rl])}function Fo(t,e){return e=null==e?Rt:e,!!e&&("number"==typeof t||Ze.test(t))&&t>-1&&t%1==0&&t<e}function jo(t,e,n){if(!cc(n))return!1;var r=typeof e;return!!("number"==r?Xu(n)&&Fo(e,n.length):"string"==r&&e in n)&&$u(n[e],t)}function Bo(t,e){if(xp(t))return!1;var n=typeof t;return!("number"!=n&&"symbol"!=n&&"boolean"!=n&&null!=t&&!bc(t))||(Oe.test(t)||!Ae.test(t)||null!=e&&t in fl(e))}function Wo(t){var e=typeof t;return"string"==e||"number"==e||"symbol"==e||"boolean"==e?"__proto__"!==t:null===t}function Vo(t){var e=Co(t),r=n[e];if("function"!=typeof r||!(e in b.prototype))return!1;if(t===r)return!0;var i=Sf(r);return!!i&&t===i[0]}function zo(t){return!!wl&&wl in t}function Ho(t){var e=t&&t.constructor,n="function"==typeof e&&e.prototype||ml;return t===n}function qo(t){return t===t&&!cc(t)}function Yo(t,e){return function(n){return null!=n&&(n[t]===e&&(e!==it||t in fl(n)))}}function Ko(t){var e=Ru(t,function(t){return n.size===lt&&n.clear(),t}),n=e.cache;return e}function Go(t,e){var n=t[1],r=e[1],i=n|r,o=i<(mt|yt|Mt),a=r==Mt&&n==bt||r==Mt&&n==kt&&t[7].length<=e[8]||r==(Mt|kt)&&e[7].length<=e[8]&&n==bt;if(!o&&!a)return t;r&mt&&(t[2]=e[2],i|=n&mt?0:_t);var u=e[3];if(u){var c=t[3];t[3]=c?Fi(c,u,e[4]):u,t[4]=c?G(t[3],ft):e[4]}return u=e[5],u&&(c=t[5],t[5]=c?ji(c,u,e[6]):u,t[6]=c?G(t[5],ft):e[6]),u=e[7],u&&(t[7]=u),r&Mt&&(t[8]=null==t[8]?e[8]:Xl(t[8],e[8])),null==t[9]&&(t[9]=e[9]),t[0]=e[0],t[1]=i,t}function $o(t){var e=[];if(null!=t)for(var n in fl(t))e.push(n);return e}function Xo(t){return Cl.call(t)}function Zo(t,e,n){return e=$l(e===it?t.length-1:e,0),function(){for(var r=arguments,i=-1,o=$l(r.length-e,0),a=al(o);++i<o;)a[i]=r[e+i];i=-1;for(var c=al(e+1);++i<e;)c[i]=r[i];return c[e]=n(a),u(t,this,c)}}function Qo(t,e){return e.length<2?t:cr(t,li(e,0,-1))}function Jo(t,e){for(var n=t.length,r=Xl(e.length,n),i=Bi(t);r--;){var o=e[r];t[r]=Fo(o,n)?i[o]:it}return t}function ta(t,e,n){var r=e+"";return Rf(t,Lo(r,oa(Ao(r),n)))}function ea(t){var e=0,n=0;return function(){var r=Zl(),i=Nt-(r-n);if(n=r,i>0){if(++e>=Pt)return arguments[0]}else e=0;return t.apply(it,arguments)}}function na(t,e){var n=-1,r=t.length,i=r-1;for(e=e===it?r:e;++n<e;){var o=ni(n,i),a=t[o];t[o]=t[n],t[n]=a}return t.length=e,t}function ra(t){if("string"==typeof t||bc(t))return t;var e=t+"";return"0"==e&&1/t==-Dt?"-0":e}function ia(t){if(null!=t){try{return _l.call(t)}catch(t){}try{return t+""}catch(t){}}return""}function oa(t,e){return s(Wt,function(n){var r="_."+n[0];e&n[1]&&!h(t,r)&&t.push(r)}),t.sort()}function aa(t){if(t instanceof b)return t.clone();var e=new i(t.__wrapped__,t.__chain__);return e.__actions__=Bi(t.__actions__),e.__index__=t.__index__,e.__values__=t.__values__,e}function ua(t,e,n){e=(n?jo(t,e,n):e===it)?1:$l(Ec(e),0);var r=null==t?0:t.length;if(!r||e<1)return[];for(var i=0,o=0,a=al(Vl(r/e));i<r;)a[o++]=li(t,i,i+=e);return a}function ca(t){for(var e=-1,n=null==t?0:t.length,r=0,i=[];++e<n;){var o=t[e];o&&(i[r++]=o)}return i}function sa(){var t=arguments.length;if(!t)return[];for(var e=al(t-1),n=arguments[0],r=t;r--;)e[r-1]=arguments[r];return g(xp(n)?Bi(n):[n],er(e,1))}function la(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===it?1:Ec(e),li(t,e<0?0:e,r)):[]}function fa(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===it?1:Ec(e),e=r-e,li(t,0,e<0?0:e)):[]}function pa(t,e){return t&&t.length?bi(t,ko(e,3),!0,!0):[]}function ha(t,e){return t&&t.length?bi(t,ko(e,3),!0):[]}function da(t,e,n,r){var i=null==t?0:t.length;return i?(n&&"number"!=typeof n&&jo(t,e,n)&&(n=0,r=i),Jn(t,e,n,r)):[]}function va(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=null==n?0:Ec(n);return i<0&&(i=$l(r+i,0)),C(t,ko(e,3),i)}function ga(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=r-1;return n!==it&&(i=Ec(n),i=n<0?$l(r+i,0):Xl(i,r-1)),C(t,ko(e,3),i,!0)}function ma(t){var e=null==t?0:t.length;return e?er(t,1):[]}function ya(t){var e=null==t?0:t.length;return e?er(t,Dt):[]}function _a(t,e){var n=null==t?0:t.length;return n?(e=e===it?1:Ec(e),er(t,e)):[]}function ba(t){for(var e=-1,n=null==t?0:t.length,r={};++e<n;){var i=t[e];r[i[0]]=i[1]}return r}function xa(t){return t&&t.length?t[0]:it}function wa(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=null==n?0:Ec(n);return i<0&&(i=$l(r+i,0)),M(t,e,i)}function Ca(t){var e=null==t?0:t.length;return e?li(t,0,-1):[]}function Ma(t,e){return null==t?"":Kl.call(t,e)}function ka(t){var e=null==t?0:t.length;return e?t[e-1]:it}function Ea(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=r;return n!==it&&(i=Ec(n),i=i<0?$l(r+i,0):Xl(i,r-1)),e===e?Q(t,e,i):C(t,E,i,!0)}function Ta(t,e){return t&&t.length?$r(t,Ec(e)):it}function Sa(t,e){return t&&t.length&&e&&e.length?ti(t,e):t}function Pa(t,e,n){return t&&t.length&&e&&e.length?ti(t,e,ko(n,2)):t}function Na(t,e,n){return t&&t.length&&e&&e.length?ti(t,e,it,n):t}function Aa(t,e){var n=[];if(!t||!t.length)return n;var r=-1,i=[],o=t.length;for(e=ko(e,3);++r<o;){var a=t[r];e(a,r,t)&&(n.push(a),i.push(r))}return ei(t,i),n}function Oa(t){return null==t?t:tf.call(t)}function Ia(t,e,n){var r=null==t?0:t.length;return r?(n&&"number"!=typeof n&&jo(t,e,n)?(e=0,n=r):(e=null==e?0:Ec(e),n=n===it?r:Ec(n)),li(t,e,n)):[]}function Da(t,e){return pi(t,e)}function Ra(t,e,n){return hi(t,e,ko(n,2))}function La(t,e){var n=null==t?0:t.length;if(n){var r=pi(t,e);if(r<n&&$u(t[r],e))return r}return-1}function Ua(t,e){return pi(t,e,!0)}function Fa(t,e,n){return hi(t,e,ko(n,2),!0)}function ja(t,e){var n=null==t?0:t.length;if(n){var r=pi(t,e,!0)-1;if($u(t[r],e))return r}return-1}function Ba(t){return t&&t.length?di(t):[]}function Wa(t,e){return t&&t.length?di(t,ko(e,2)):[]}function Va(t){var e=null==t?0:t.length;return e?li(t,1,e):[]}function za(t,e,n){return t&&t.length?(e=n||e===it?1:Ec(e),li(t,0,e<0?0:e)):[]}function Ha(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===it?1:Ec(e),e=r-e,li(t,e<0?0:e,r)):[]}function qa(t,e){return t&&t.length?bi(t,ko(e,3),!1,!0):[]}function Ya(t,e){return t&&t.length?bi(t,ko(e,3)):[]}function Ka(t){return t&&t.length?mi(t):[]}function Ga(t,e){return t&&t.length?mi(t,ko(e,2)):[]}function $a(t,e){return e="function"==typeof e?e:it,t&&t.length?mi(t,it,e):[]}function Xa(t){if(!t||!t.length)return[];var e=0;return t=p(t,function(t){if(Zu(t))return e=$l(t.length,e),!0}),I(e,function(e){return v(t,S(e))})}function Za(t,e){if(!t||!t.length)return[];var n=Xa(t);return null==e?n:v(n,function(t){return u(e,it,t)})}function Qa(t,e){return Ci(t||[],e||[],On)}function Ja(t,e){return Ci(t||[],e||[],ci)}function tu(t){var e=n(t);return e.__chain__=!0,e}function eu(t,e){return e(t),t}function nu(t,e){return e(t)}function ru(){return tu(this)}function iu(){return new i(this.value(),this.__chain__)}function ou(){this.__values__===it&&(this.__values__=Mc(this.value()));var t=this.__index__>=this.__values__.length,e=t?it:this.__values__[this.__index__++];return{done:t,value:e}}function au(){return this}function uu(t){for(var e,n=this;n instanceof r;){var i=aa(n);i.__index__=0,i.__values__=it,e?o.__wrapped__=i:e=i;var o=i;n=n.__wrapped__}return o.__wrapped__=t,e}function cu(){var t=this.__wrapped__;if(t instanceof b){var e=t;return this.__actions__.length&&(e=new b(this)),e=e.reverse(),e.__actions__.push({func:nu,args:[Oa],thisArg:it}),new i(e,this.__chain__)}return this.thru(Oa)}function su(){return xi(this.__wrapped__,this.__actions__)}function lu(t,e,n){
var r=xp(t)?f:Kn;return n&&jo(t,e,n)&&(e=it),r(t,ko(e,3))}function fu(t,e){var n=xp(t)?p:tr;return n(t,ko(e,3))}function pu(t,e){return er(yu(t,e),1)}function hu(t,e){return er(yu(t,e),Dt)}function du(t,e,n){return n=n===it?1:Ec(n),er(yu(t,e),n)}function vu(t,e){var n=xp(t)?s:_f;return n(t,ko(e,3))}function gu(t,e){var n=xp(t)?l:bf;return n(t,ko(e,3))}function mu(t,e,n,r){t=Xu(t)?t:rs(t),n=n&&!r?Ec(n):0;var i=t.length;return n<0&&(n=$l(i+n,0)),_c(t)?n<=i&&t.indexOf(e,n)>-1:!!i&&M(t,e,n)>-1}function yu(t,e){var n=xp(t)?v:Hr;return n(t,ko(e,3))}function _u(t,e,n,r){return null==t?[]:(xp(e)||(e=null==e?[]:[e]),n=r?it:n,xp(n)||(n=null==n?[]:[n]),Xr(t,e,n))}function bu(t,e,n){var r=xp(t)?m:N,i=arguments.length<3;return r(t,ko(e,4),n,i,_f)}function xu(t,e,n){var r=xp(t)?y:N,i=arguments.length<3;return r(t,ko(e,4),n,i,bf)}function wu(t,e){var n=xp(t)?p:tr;return n(t,Lu(ko(e,3)))}function Cu(t){var e=xp(t)?Sn:ai;return e(t)}function Mu(t,e,n){e=(n?jo(t,e,n):e===it)?1:Ec(e);var r=xp(t)?Pn:ui;return r(t,e)}function ku(t){var e=xp(t)?Nn:si;return e(t)}function Eu(t){if(null==t)return 0;if(Xu(t))return _c(t)?J(t):t.length;var e=Af(t);return e==Zt||e==ie?t.size:Wr(t).length}function Tu(t,e,n){var r=xp(t)?_:fi;return n&&jo(t,e,n)&&(e=it),r(t,ko(e,3))}function Su(t,e){if("function"!=typeof e)throw new dl(ct);return t=Ec(t),function(){if(--t<1)return e.apply(this,arguments)}}function Pu(t,e,n){return e=n?it:e,e=t&&null==e?t.length:e,po(t,Mt,it,it,it,it,e)}function Nu(t,e){var n;if("function"!=typeof e)throw new dl(ct);return t=Ec(t),function(){return--t>0&&(n=e.apply(this,arguments)),t<=1&&(e=it),n}}function Au(t,e,n){e=n?it:e;var r=po(t,bt,it,it,it,it,it,e);return r.placeholder=Au.placeholder,r}function Ou(t,e,n){e=n?it:e;var r=po(t,xt,it,it,it,it,it,e);return r.placeholder=Ou.placeholder,r}function Iu(t,e,n){function r(e){var n=p,r=h;return p=h=it,y=e,v=t.apply(r,n)}function i(t){return y=t,g=Df(u,e),_?r(t):v}function o(t){var n=t-m,r=t-y,i=e-n;return b?Xl(i,d-r):i}function a(t){var n=t-m,r=t-y;return m===it||n>=e||n<0||b&&r>=d}function u(){var t=sp();return a(t)?c(t):void(g=Df(u,o(t)))}function c(t){return g=it,x&&p?r(t):(p=h=it,v)}function s(){g!==it&&Ef(g),y=0,p=m=h=g=it}function l(){return g===it?v:c(sp())}function f(){var t=sp(),n=a(t);if(p=arguments,h=this,m=t,n){if(g===it)return i(m);if(b)return g=Df(u,e),r(m)}return g===it&&(g=Df(u,e)),v}var p,h,d,v,g,m,y=0,_=!1,b=!1,x=!0;if("function"!=typeof t)throw new dl(ct);return e=Sc(e)||0,cc(n)&&(_=!!n.leading,b="maxWait"in n,d=b?$l(Sc(n.maxWait)||0,e):d,x="trailing"in n?!!n.trailing:x),f.cancel=s,f.flush=l,f}function Du(t){return po(t,Et)}function Ru(t,e){if("function"!=typeof t||null!=e&&"function"!=typeof e)throw new dl(ct);var n=function(){var r=arguments,i=e?e.apply(this,r):r[0],o=n.cache;if(o.has(i))return o.get(i);var a=t.apply(this,r);return n.cache=o.set(i,a)||o,a};return n.cache=new(Ru.Cache||pn),n}function Lu(t){if("function"!=typeof t)throw new dl(ct);return function(){var e=arguments;switch(e.length){case 0:return!t.call(this);case 1:return!t.call(this,e[0]);case 2:return!t.call(this,e[0],e[1]);case 3:return!t.call(this,e[0],e[1],e[2])}return!t.apply(this,e)}}function Uu(t){return Nu(2,t)}function Fu(t,e){if("function"!=typeof t)throw new dl(ct);return e=e===it?e:Ec(e),oi(t,e)}function ju(t,e){if("function"!=typeof t)throw new dl(ct);return e=null==e?0:$l(Ec(e),0),oi(function(n){var r=n[e],i=Ti(n,0,e);return r&&g(i,r),u(t,this,i)})}function Bu(t,e,n){var r=!0,i=!0;if("function"!=typeof t)throw new dl(ct);return cc(n)&&(r="leading"in n?!!n.leading:r,i="trailing"in n?!!n.trailing:i),Iu(t,e,{leading:r,maxWait:e,trailing:i})}function Wu(t){return Pu(t,1)}function Vu(t,e){return vp(ki(e),t)}function zu(){if(!arguments.length)return[];var t=arguments[0];return xp(t)?t:[t]}function Hu(t){return Bn(t,dt)}function qu(t,e){return e="function"==typeof e?e:it,Bn(t,dt,e)}function Yu(t){return Bn(t,pt|dt)}function Ku(t,e){return e="function"==typeof e?e:it,Bn(t,pt|dt,e)}function Gu(t,e){return null==e||Vn(t,e,Hc(e))}function $u(t,e){return t===e||t!==t&&e!==e}function Xu(t){return null!=t&&uc(t.length)&&!oc(t)}function Zu(t){return sc(t)&&Xu(t)}function Qu(t){return t===!0||t===!1||sc(t)&&fr(t)==qt}function Ju(t){return sc(t)&&1===t.nodeType&&!mc(t)}function tc(t){if(null==t)return!0;if(Xu(t)&&(xp(t)||"string"==typeof t||"function"==typeof t.splice||Cp(t)||Sp(t)||bp(t)))return!t.length;var e=Af(t);if(e==Zt||e==ie)return!t.size;if(Ho(t))return!Wr(t).length;for(var n in t)if(bl.call(t,n))return!1;return!0}function ec(t,e){return Or(t,e)}function nc(t,e,n){n="function"==typeof n?n:it;var r=n?n(t,e):it;return r===it?Or(t,e,it,n):!!r}function rc(t){if(!sc(t))return!1;var e=fr(t);return e==Gt||e==Kt||"string"==typeof t.message&&"string"==typeof t.name&&!mc(t)}function ic(t){return"number"==typeof t&&Yl(t)}function oc(t){if(!cc(t))return!1;var e=fr(t);return e==$t||e==Xt||e==Ht||e==ne}function ac(t){return"number"==typeof t&&t==Ec(t)}function uc(t){return"number"==typeof t&&t>-1&&t%1==0&&t<=Rt}function cc(t){var e=typeof t;return null!=t&&("object"==e||"function"==e)}function sc(t){return null!=t&&"object"==typeof t}function lc(t,e){return t===e||Rr(t,e,To(e))}function fc(t,e,n){return n="function"==typeof n?n:it,Rr(t,e,To(e),n)}function pc(t){return gc(t)&&t!=+t}function hc(t){if(Of(t))throw new cl(ut);return Lr(t)}function dc(t){return null===t}function vc(t){return null==t}function gc(t){return"number"==typeof t||sc(t)&&fr(t)==Qt}function mc(t){if(!sc(t)||fr(t)!=te)return!1;var e=Al(t);if(null===e)return!0;var n=bl.call(e,"constructor")&&e.constructor;return"function"==typeof n&&n instanceof n&&_l.call(n)==Ml}function yc(t){return ac(t)&&t>=-Rt&&t<=Rt}function _c(t){return"string"==typeof t||!xp(t)&&sc(t)&&fr(t)==oe}function bc(t){return"symbol"==typeof t||sc(t)&&fr(t)==ae}function xc(t){return t===it}function wc(t){return sc(t)&&Af(t)==ce}function Cc(t){return sc(t)&&fr(t)==se}function Mc(t){if(!t)return[];if(Xu(t))return _c(t)?tt(t):Bi(t);if(Ll&&t[Ll])return q(t[Ll]());var e=Af(t),n=e==Zt?Y:e==ie?$:rs;return n(t)}function kc(t){if(!t)return 0===t?t:0;if(t=Sc(t),t===Dt||t===-Dt){var e=t<0?-1:1;return e*Lt}return t===t?t:0}function Ec(t){var e=kc(t),n=e%1;return e===e?n?e-n:e:0}function Tc(t){return t?jn(Ec(t),0,Ft):0}function Sc(t){if("number"==typeof t)return t;if(bc(t))return Ut;if(cc(t)){var e="function"==typeof t.valueOf?t.valueOf():t;t=cc(e)?e+"":e}if("string"!=typeof t)return 0===t?t:+t;t=t.replace(Ue,"");var n=Ge.test(t);return n||Xe.test(t)?ir(t.slice(2),n?2:8):Ke.test(t)?Ut:+t}function Pc(t){return Wi(t,qc(t))}function Nc(t){return t?jn(Ec(t),-Rt,Rt):0===t?t:0}function Ac(t){return null==t?"":gi(t)}function Oc(t,e){var n=yf(t);return null==e?n:Rn(n,e)}function Ic(t,e){return w(t,ko(e,3),nr)}function Dc(t,e){return w(t,ko(e,3),or)}function Rc(t,e){return null==t?t:xf(t,ko(e,3),qc)}function Lc(t,e){return null==t?t:wf(t,ko(e,3),qc)}function Uc(t,e){return t&&nr(t,ko(e,3))}function Fc(t,e){return t&&or(t,ko(e,3))}function jc(t){return null==t?[]:ar(t,Hc(t))}function Bc(t){return null==t?[]:ar(t,qc(t))}function Wc(t,e,n){var r=null==t?it:cr(t,e);return r===it?n:r}function Vc(t,e){return null!=t&&Oo(t,e,_r)}function zc(t,e){return null!=t&&Oo(t,e,Cr)}function Hc(t){return Xu(t)?Tn(t):Wr(t)}function qc(t){return Xu(t)?Tn(t,!0):Vr(t)}function Yc(t,e){var n={};return e=ko(e,3),nr(t,function(t,r,i){Un(n,e(t,r,i),t)}),n}function Kc(t,e){var n={};return e=ko(e,3),nr(t,function(t,r,i){Un(n,r,e(t,r,i))}),n}function Gc(t,e){return $c(t,Lu(ko(e)))}function $c(t,e){if(null==t)return{};var n=v(wo(t),function(t){return[t]});return e=ko(e),Qr(t,n,function(t,n){return e(t,n[0])})}function Xc(t,e,n){e=Ei(e,t);var r=-1,i=e.length;for(i||(i=1,t=it);++r<i;){var o=null==t?it:t[ra(e[r])];o===it&&(r=i,o=n),t=oc(o)?o.call(t):o}return t}function Zc(t,e,n){return null==t?t:ci(t,e,n)}function Qc(t,e,n,r){return r="function"==typeof r?r:it,null==t?t:ci(t,e,n,r)}function Jc(t,e,n){var r=xp(t),i=r||Cp(t)||Sp(t);if(e=ko(e,4),null==n){var o=t&&t.constructor;n=i?r?new o:[]:cc(t)&&oc(o)?yf(Al(t)):{}}return(i?s:nr)(t,function(t,r,i){return e(n,t,r,i)}),n}function ts(t,e){return null==t||yi(t,e)}function es(t,e,n){return null==t?t:_i(t,e,ki(n))}function ns(t,e,n,r){return r="function"==typeof r?r:it,null==t?t:_i(t,e,ki(n),r)}function rs(t){return null==t?[]:L(t,Hc(t))}function is(t){return null==t?[]:L(t,qc(t))}function os(t,e,n){return n===it&&(n=e,e=it),n!==it&&(n=Sc(n),n=n===n?n:0),e!==it&&(e=Sc(e),e=e===e?e:0),jn(Sc(t),e,n)}function as(t,e,n){return e=kc(e),n===it?(n=e,e=0):n=kc(n),t=Sc(t),kr(t,e,n)}function us(t,e,n){if(n&&"boolean"!=typeof n&&jo(t,e,n)&&(e=n=it),n===it&&("boolean"==typeof e?(n=e,e=it):"boolean"==typeof t&&(n=t,t=it)),t===it&&e===it?(t=0,e=1):(t=kc(t),e===it?(e=t,t=0):e=kc(e)),t>e){var r=t;t=e,e=r}if(n||t%1||e%1){var i=Jl();return Xl(t+i*(e-t+rr("1e-"+((i+"").length-1))),e)}return ni(t,e)}function cs(t){return th(Ac(t).toLowerCase())}function ss(t){return t=Ac(t),t&&t.replace(Qe,br).replace(Hn,"")}function ls(t,e,n){t=Ac(t),e=gi(e);var r=t.length;n=n===it?r:jn(Ec(n),0,r);var i=n;return n-=e.length,n>=0&&t.slice(n,i)==e}function fs(t){return t=Ac(t),t&&Te.test(t)?t.replace(ke,xr):t}function ps(t){return t=Ac(t),t&&Le.test(t)?t.replace(Re,"\\$&"):t}function hs(t,e,n){t=Ac(t),e=Ec(e);var r=e?J(t):0;if(!e||r>=e)return t;var i=(e-r)/2;return oo(zl(i),n)+t+oo(Vl(i),n)}function ds(t,e,n){t=Ac(t),e=Ec(e);var r=e?J(t):0;return e&&r<e?t+oo(e-r,n):t}function vs(t,e,n){t=Ac(t),e=Ec(e);var r=e?J(t):0;return e&&r<e?oo(e-r,n)+t:t}function gs(t,e,n){return n||null==e?e=0:e&&(e=+e),Ql(Ac(t).replace(Fe,""),e||0)}function ms(t,e,n){return e=(n?jo(t,e,n):e===it)?1:Ec(e),ii(Ac(t),e)}function ys(){var t=arguments,e=Ac(t[0]);return t.length<3?e:e.replace(t[1],t[2])}function _s(t,e,n){return n&&"number"!=typeof n&&jo(t,e,n)&&(e=n=it),(n=n===it?Ft:n>>>0)?(t=Ac(t),t&&("string"==typeof e||null!=e&&!Ep(e))&&(e=gi(e),!e&&z(t))?Ti(tt(t),0,n):t.split(e,n)):[]}function bs(t,e,n){return t=Ac(t),n=null==n?0:jn(Ec(n),0,t.length),e=gi(e),t.slice(n,n+e.length)==e}function xs(t,e,r){var i=n.templateSettings;r&&jo(t,e,r)&&(e=it),t=Ac(t),e=Ip({},e,i,ho);var o,a,u=Ip({},e.imports,i.imports,ho),c=Hc(u),s=L(u,c),l=0,f=e.interpolate||Je,p="__p += '",h=pl((e.escape||Je).source+"|"+f.source+"|"+(f===Ne?qe:Je).source+"|"+(e.evaluate||Je).source+"|$","g"),d="//# sourceURL="+("sourceURL"in e?e.sourceURL:"lodash.templateSources["+ ++Xn+"]")+"\n";t.replace(h,function(e,n,r,i,u,c){return r||(r=i),p+=t.slice(l,c).replace(tn,W),n&&(o=!0,p+="' +\n__e("+n+") +\n'"),u&&(a=!0,p+="';\n"+u+";\n__p += '"),r&&(p+="' +\n((__t = ("+r+")) == null ? '' : __t) +\n'"),l=c+e.length,e}),p+="';\n";var v=e.variable;v||(p="with (obj) {\n"+p+"\n}\n"),p=(a?p.replace(xe,""):p).replace(we,"$1").replace(Ce,"$1;"),p="function("+(v||"obj")+") {\n"+(v?"":"obj || (obj = {});\n")+"var __t, __p = ''"+(o?", __e = _.escape":"")+(a?", __j = Array.prototype.join;\nfunction print() { __p += __j.call(arguments, '') }\n":";\n")+p+"return __p\n}";var g=eh(function(){return sl(c,d+"return "+p).apply(it,s)});if(g.source=p,rc(g))throw g;return g}function ws(t){return Ac(t).toLowerCase()}function Cs(t){return Ac(t).toUpperCase()}function Ms(t,e,n){if(t=Ac(t),t&&(n||e===it))return t.replace(Ue,"");if(!t||!(e=gi(e)))return t;var r=tt(t),i=tt(e),o=F(r,i),a=j(r,i)+1;return Ti(r,o,a).join("")}function ks(t,e,n){if(t=Ac(t),t&&(n||e===it))return t.replace(je,"");if(!t||!(e=gi(e)))return t;var r=tt(t),i=j(r,tt(e))+1;return Ti(r,0,i).join("")}function Es(t,e,n){if(t=Ac(t),t&&(n||e===it))return t.replace(Fe,"");if(!t||!(e=gi(e)))return t;var r=tt(t),i=F(r,tt(e));return Ti(r,i).join("")}function Ts(t,e){var n=Tt,r=St;if(cc(e)){var i="separator"in e?e.separator:i;n="length"in e?Ec(e.length):n,r="omission"in e?gi(e.omission):r}t=Ac(t);var o=t.length;if(z(t)){var a=tt(t);o=a.length}if(n>=o)return t;var u=n-J(r);if(u<1)return r;var c=a?Ti(a,0,u).join(""):t.slice(0,u);if(i===it)return c+r;if(a&&(u+=c.length-u),Ep(i)){if(t.slice(u).search(i)){var s,l=c;for(i.global||(i=pl(i.source,Ac(Ye.exec(i))+"g")),i.lastIndex=0;s=i.exec(l);)var f=s.index;c=c.slice(0,f===it?u:f)}}else if(t.indexOf(gi(i),u)!=u){var p=c.lastIndexOf(i);p>-1&&(c=c.slice(0,p))}return c+r}function Ss(t){return t=Ac(t),t&&Ee.test(t)?t.replace(Me,wr):t}function Ps(t,e,n){return t=Ac(t),e=n?it:e,e===it?H(t)?rt(t):x(t):t.match(e)||[]}function Ns(t){var e=null==t?0:t.length,n=ko();return t=e?v(t,function(t){if("function"!=typeof t[1])throw new dl(ct);return[n(t[0]),t[1]]}):[],oi(function(n){for(var r=-1;++r<e;){var i=t[r];if(u(i[0],this,n))return u(i[1],this,n)}})}function As(t){return Wn(Bn(t,pt))}function Os(t){return function(){return t}}function Is(t,e){return null==t||t!==t?e:t}function Ds(t){return t}function Rs(t){return Br("function"==typeof t?t:Bn(t,pt))}function Ls(t){return qr(Bn(t,pt))}function Us(t,e){return Yr(t,Bn(e,pt))}function Fs(t,e,n){var r=Hc(e),i=ar(e,r);null!=n||cc(e)&&(i.length||!r.length)||(n=e,e=t,t=this,i=ar(e,Hc(e)));var o=!(cc(n)&&"chain"in n&&!n.chain),a=oc(t);return s(i,function(n){var r=e[n];t[n]=r,a&&(t.prototype[n]=function(){var e=this.__chain__;if(o||e){var n=t(this.__wrapped__),i=n.__actions__=Bi(this.__actions__);return i.push({func:r,args:arguments,thisArg:t}),n.__chain__=e,n}return r.apply(t,g([this.value()],arguments))})}),t}function js(){return ur._===this&&(ur._=kl),this}function Bs(){}function Ws(t){return t=Ec(t),oi(function(e){return $r(e,t)})}function Vs(t){return Bo(t)?S(ra(t)):Jr(t)}function zs(t){return function(e){return null==t?it:cr(t,e)}}function Hs(){return[]}function qs(){return!1}function Ys(){return{}}function Ks(){return""}function Gs(){return!0}function $s(t,e){if(t=Ec(t),t<1||t>Rt)return[];var n=Ft,r=Xl(t,Ft);e=ko(e),t-=Ft;for(var i=I(r,e);++n<t;)e(n);return i}function Xs(t){return xp(t)?v(t,ra):bc(t)?[t]:Bi(Lf(Ac(t)))}function Zs(t){var e=++xl;return Ac(t)+e}function Qs(t){return t&&t.length?Gn(t,Ds,pr):it}function Js(t,e){return t&&t.length?Gn(t,ko(e,2),pr):it}function tl(t){return T(t,Ds)}function el(t,e){return T(t,ko(e,2))}function nl(t){return t&&t.length?Gn(t,Ds,zr):it}function rl(t,e){return t&&t.length?Gn(t,ko(e,2),zr):it}function il(t){return t&&t.length?O(t,Ds):0}function ol(t,e){return t&&t.length?O(t,ko(e,2)):0}e=null==e?ur:Mr.defaults(ur.Object(),e,Mr.pick(ur,$n));var al=e.Array,ul=e.Date,cl=e.Error,sl=e.Function,ll=e.Math,fl=e.Object,pl=e.RegExp,hl=e.String,dl=e.TypeError,vl=al.prototype,gl=sl.prototype,ml=fl.prototype,yl=e["__core-js_shared__"],_l=gl.toString,bl=ml.hasOwnProperty,xl=0,wl=function(){var t=/[^.]+$/.exec(yl&&yl.keys&&yl.keys.IE_PROTO||"");return t?"Symbol(src)_1."+t:""}(),Cl=ml.toString,Ml=_l.call(fl),kl=ur._,El=pl("^"+_l.call(bl).replace(Re,"\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$"),Tl=lr?e.Buffer:it,Sl=e.Symbol,Pl=e.Uint8Array,Nl=Tl?Tl.allocUnsafe:it,Al=K(fl.getPrototypeOf,fl),Ol=fl.create,Il=ml.propertyIsEnumerable,Dl=vl.splice,Rl=Sl?Sl.isConcatSpreadable:it,Ll=Sl?Sl.iterator:it,Ul=Sl?Sl.toStringTag:it,Fl=function(){try{var t=So(fl,"defineProperty");return t({},"",{}),t}catch(t){}}(),jl=e.clearTimeout!==ur.clearTimeout&&e.clearTimeout,Bl=ul&&ul.now!==ur.Date.now&&ul.now,Wl=e.setTimeout!==ur.setTimeout&&e.setTimeout,Vl=ll.ceil,zl=ll.floor,Hl=fl.getOwnPropertySymbols,ql=Tl?Tl.isBuffer:it,Yl=e.isFinite,Kl=vl.join,Gl=K(fl.keys,fl),$l=ll.max,Xl=ll.min,Zl=ul.now,Ql=e.parseInt,Jl=ll.random,tf=vl.reverse,ef=So(e,"DataView"),nf=So(e,"Map"),rf=So(e,"Promise"),of=So(e,"Set"),af=So(e,"WeakMap"),uf=So(fl,"create"),cf=af&&new af,sf={},lf=ia(ef),ff=ia(nf),pf=ia(rf),hf=ia(of),df=ia(af),vf=Sl?Sl.prototype:it,gf=vf?vf.valueOf:it,mf=vf?vf.toString:it,yf=function(){function t(){}return function(e){if(!cc(e))return{};if(Ol)return Ol(e);t.prototype=e;var n=new t;return t.prototype=it,n}}();n.templateSettings={escape:Se,evaluate:Pe,interpolate:Ne,variable:"",imports:{_:n}},n.prototype=r.prototype,n.prototype.constructor=n,i.prototype=yf(r.prototype),i.prototype.constructor=i,b.prototype=yf(r.prototype),b.prototype.constructor=b,nt.prototype.clear=ze,nt.prototype.delete=en,nt.prototype.get=nn,nt.prototype.has=rn,nt.prototype.set=on,an.prototype.clear=un,an.prototype.delete=cn,an.prototype.get=sn,an.prototype.has=ln,an.prototype.set=fn,pn.prototype.clear=hn,pn.prototype.delete=dn,pn.prototype.get=vn,pn.prototype.has=gn,pn.prototype.set=mn,yn.prototype.add=yn.prototype.push=_n,yn.prototype.has=bn,xn.prototype.clear=wn,xn.prototype.delete=Cn,xn.prototype.get=Mn,xn.prototype.has=kn,xn.prototype.set=En;var _f=Yi(nr),bf=Yi(or,!0),xf=Ki(),wf=Ki(!0),Cf=cf?function(t,e){return cf.set(t,e),t}:Ds,Mf=Fl?function(t,e){return Fl(t,"toString",{configurable:!0,enumerable:!1,value:Os(e),writable:!0})}:Ds,kf=oi,Ef=jl||function(t){return ur.clearTimeout(t)},Tf=of&&1/$(new of([,-0]))[1]==Dt?function(t){return new of(t)}:Bs,Sf=cf?function(t){return cf.get(t)}:Bs,Pf=Hl?function(t){return null==t?[]:(t=fl(t),p(Hl(t),function(e){return Il.call(t,e)}))}:Hs,Nf=Hl?function(t){for(var e=[];t;)g(e,Pf(t)),t=Al(t);return e}:Hs,Af=fr;(ef&&Af(new ef(new ArrayBuffer(1)))!=fe||nf&&Af(new nf)!=Zt||rf&&Af(rf.resolve())!=ee||of&&Af(new of)!=ie||af&&Af(new af)!=ce)&&(Af=function(t){var e=fr(t),n=e==te?t.constructor:it,r=n?ia(n):"";if(r)switch(r){case lf:return fe;case ff:return Zt;case pf:return ee;case hf:return ie;case df:return ce}return e});var Of=yl?oc:qs,If=ea(Cf),Df=Wl||function(t,e){return ur.setTimeout(t,e)},Rf=ea(Mf),Lf=Ko(function(t){var e=[];return Ie.test(t)&&e.push(""),t.replace(De,function(t,n,r,i){e.push(r?i.replace(He,"$1"):n||t)}),e}),Uf=oi(function(t,e){return Zu(t)?Yn(t,er(e,1,Zu,!0)):[]}),Ff=oi(function(t,e){var n=ka(e);return Zu(n)&&(n=it),Zu(t)?Yn(t,er(e,1,Zu,!0),ko(n,2)):[]}),jf=oi(function(t,e){var n=ka(e);return Zu(n)&&(n=it),Zu(t)?Yn(t,er(e,1,Zu,!0),it,n):[]}),Bf=oi(function(t){var e=v(t,Mi);return e.length&&e[0]===t[0]?Er(e):[]}),Wf=oi(function(t){var e=ka(t),n=v(t,Mi);return e===ka(n)?e=it:n.pop(),n.length&&n[0]===t[0]?Er(n,ko(e,2)):[]}),Vf=oi(function(t){var e=ka(t),n=v(t,Mi);return e="function"==typeof e?e:it,e&&n.pop(),n.length&&n[0]===t[0]?Er(n,it,e):[]}),zf=oi(Sa),Hf=bo(function(t,e){var n=null==t?0:t.length,r=Fn(t,e);return ei(t,v(e,function(t){return Fo(t,n)?+t:t}).sort(Li)),r}),qf=oi(function(t){return mi(er(t,1,Zu,!0))}),Yf=oi(function(t){var e=ka(t);return Zu(e)&&(e=it),mi(er(t,1,Zu,!0),ko(e,2))}),Kf=oi(function(t){var e=ka(t);return e="function"==typeof e?e:it,mi(er(t,1,Zu,!0),it,e)}),Gf=oi(function(t,e){return Zu(t)?Yn(t,e):[]}),$f=oi(function(t){return wi(p(t,Zu))}),Xf=oi(function(t){var e=ka(t);return Zu(e)&&(e=it),wi(p(t,Zu),ko(e,2))}),Zf=oi(function(t){var e=ka(t);return e="function"==typeof e?e:it,wi(p(t,Zu),it,e)}),Qf=oi(Xa),Jf=oi(function(t){var e=t.length,n=e>1?t[e-1]:it;return n="function"==typeof n?(t.pop(),n):it,Za(t,n)}),tp=bo(function(t){var e=t.length,n=e?t[0]:0,r=this.__wrapped__,o=function(e){return Fn(e,t)};return!(e>1||this.__actions__.length)&&r instanceof b&&Fo(n)?(r=r.slice(n,+n+(e?1:0)),r.__actions__.push({func:nu,args:[o],thisArg:it}),new i(r,this.__chain__).thru(function(t){return e&&!t.length&&t.push(it),t})):this.thru(o)}),ep=Hi(function(t,e,n){bl.call(t,n)?++t[n]:Un(t,n,1)}),np=Ji(va),rp=Ji(ga),ip=Hi(function(t,e,n){bl.call(t,n)?t[n].push(e):Un(t,n,[e])}),op=oi(function(t,e,n){var r=-1,i="function"==typeof e,o=Xu(t)?al(t.length):[];return _f(t,function(t){o[++r]=i?u(e,t,n):Sr(t,e,n)}),o}),ap=Hi(function(t,e,n){Un(t,n,e)}),up=Hi(function(t,e,n){t[n?0:1].push(e)},function(){return[[],[]]}),cp=oi(function(t,e){if(null==t)return[];var n=e.length;return n>1&&jo(t,e[0],e[1])?e=[]:n>2&&jo(e[0],e[1],e[2])&&(e=[e[0]]),Xr(t,er(e,1),[])}),sp=Bl||function(){return ur.Date.now()},lp=oi(function(t,e,n){var r=mt;if(n.length){var i=G(n,Mo(lp));r|=wt}return po(t,r,e,n,i)}),fp=oi(function(t,e,n){var r=mt|yt;if(n.length){var i=G(n,Mo(fp));r|=wt}return po(e,r,t,n,i)}),pp=oi(function(t,e){return qn(t,1,e)}),hp=oi(function(t,e,n){return qn(t,Sc(e)||0,n)});Ru.Cache=pn;var dp=kf(function(t,e){e=1==e.length&&xp(e[0])?v(e[0],R(ko())):v(er(e,1),R(ko()));var n=e.length;return oi(function(r){for(var i=-1,o=Xl(r.length,n);++i<o;)r[i]=e[i].call(this,r[i]);return u(t,this,r)})}),vp=oi(function(t,e){var n=G(e,Mo(vp));return po(t,wt,it,e,n)}),gp=oi(function(t,e){var n=G(e,Mo(gp));return po(t,Ct,it,e,n)}),mp=bo(function(t,e){return po(t,kt,it,it,it,e)}),yp=co(pr),_p=co(function(t,e){return t>=e}),bp=Pr(function(){return arguments}())?Pr:function(t){return sc(t)&&bl.call(t,"callee")&&!Il.call(t,"callee")},xp=al.isArray,wp=hr?R(hr):Nr,Cp=ql||qs,Mp=dr?R(dr):Ar,kp=vr?R(vr):Dr,Ep=gr?R(gr):Ur,Tp=mr?R(mr):Fr,Sp=yr?R(yr):jr,Pp=co(zr),Np=co(function(t,e){return t<=e}),Ap=qi(function(t,e){if(Ho(e)||Xu(e))return void Wi(e,Hc(e),t);for(var n in e)bl.call(e,n)&&On(t,n,e[n])}),Op=qi(function(t,e){Wi(e,qc(e),t)}),Ip=qi(function(t,e,n,r){Wi(e,qc(e),t,r)}),Dp=qi(function(t,e,n,r){Wi(e,Hc(e),t,r)}),Rp=bo(Fn),Lp=oi(function(t){return t.push(it,ho),u(Ip,it,t)}),Up=oi(function(t){return t.push(it,vo),u(Vp,it,t)}),Fp=no(function(t,e,n){t[e]=n},Os(Ds)),jp=no(function(t,e,n){bl.call(t,e)?t[e].push(n):t[e]=[n]},ko),Bp=oi(Sr),Wp=qi(function(t,e,n){Kr(t,e,n)}),Vp=qi(function(t,e,n,r){Kr(t,e,n,r)}),zp=bo(function(t,e){var n={};if(null==t)return n;var r=!1;e=v(e,function(e){return e=Ei(e,t),r||(r=e.length>1),e}),Wi(t,wo(t),n),r&&(n=Bn(n,pt|ht|dt,go));for(var i=e.length;i--;)yi(n,e[i]);return n}),Hp=bo(function(t,e){return null==t?{}:Zr(t,e)}),qp=fo(Hc),Yp=fo(qc),Kp=Xi(function(t,e,n){return e=e.toLowerCase(),t+(n?cs(e):e)}),Gp=Xi(function(t,e,n){return t+(n?"-":"")+e.toLowerCase()}),$p=Xi(function(t,e,n){return t+(n?" ":"")+e.toLowerCase()}),Xp=$i("toLowerCase"),Zp=Xi(function(t,e,n){return t+(n?"_":"")+e.toLowerCase()}),Qp=Xi(function(t,e,n){return t+(n?" ":"")+th(e)}),Jp=Xi(function(t,e,n){return t+(n?" ":"")+e.toUpperCase()}),th=$i("toUpperCase"),eh=oi(function(t,e){try{return u(t,it,e)}catch(t){return rc(t)?t:new cl(t)}}),nh=bo(function(t,e){return s(e,function(e){e=ra(e),Un(t,e,lp(t[e],t))}),t}),rh=to(),ih=to(!0),oh=oi(function(t,e){return function(n){return Sr(n,t,e)}}),ah=oi(function(t,e){return function(n){return Sr(t,n,e)}}),uh=io(v),ch=io(f),sh=io(_),lh=uo(),fh=uo(!0),ph=ro(function(t,e){return t+e},0),hh=lo("ceil"),dh=ro(function(t,e){return t/e},1),vh=lo("floor"),gh=ro(function(t,e){return t*e},1),mh=lo("round"),yh=ro(function(t,e){return t-e},0);return n.after=Su,n.ary=Pu,n.assign=Ap,n.assignIn=Op,n.assignInWith=Ip,n.assignWith=Dp,n.at=Rp,n.before=Nu,n.bind=lp,n.bindAll=nh,n.bindKey=fp,n.castArray=zu,n.chain=tu,n.chunk=ua,n.compact=ca,n.concat=sa,n.cond=Ns,n.conforms=As,n.constant=Os,n.countBy=ep,n.create=Oc,n.curry=Au,n.curryRight=Ou,n.debounce=Iu,n.defaults=Lp,n.defaultsDeep=Up,n.defer=pp,n.delay=hp,n.difference=Uf,n.differenceBy=Ff,n.differenceWith=jf,n.drop=la,n.dropRight=fa,n.dropRightWhile=pa,n.dropWhile=ha,n.fill=da,n.filter=fu,n.flatMap=pu,n.flatMapDeep=hu,n.flatMapDepth=du,n.flatten=ma,n.flattenDeep=ya,n.flattenDepth=_a,n.flip=Du,n.flow=rh,n.flowRight=ih,n.fromPairs=ba,n.functions=jc,n.functionsIn=Bc,n.groupBy=ip,n.initial=Ca,n.intersection=Bf,n.intersectionBy=Wf,n.intersectionWith=Vf,n.invert=Fp,n.invertBy=jp,n.invokeMap=op,n.iteratee=Rs,n.keyBy=ap,n.keys=Hc,n.keysIn=qc,n.map=yu,n.mapKeys=Yc,n.mapValues=Kc,n.matches=Ls,n.matchesProperty=Us,n.memoize=Ru,n.merge=Wp,n.mergeWith=Vp,n.method=oh,n.methodOf=ah,n.mixin=Fs,n.negate=Lu,n.nthArg=Ws,n.omit=zp,n.omitBy=Gc,n.once=Uu,n.orderBy=_u,n.over=uh,n.overArgs=dp,n.overEvery=ch,n.overSome=sh,n.partial=vp,n.partialRight=gp,n.partition=up,n.pick=Hp,n.pickBy=$c,n.property=Vs,n.propertyOf=zs,n.pull=zf,n.pullAll=Sa,n.pullAllBy=Pa,n.pullAllWith=Na,n.pullAt=Hf,n.range=lh,n.rangeRight=fh,n.rearg=mp,n.reject=wu,n.remove=Aa,n.rest=Fu,n.reverse=Oa,n.sampleSize=Mu,n.set=Zc,n.setWith=Qc,n.shuffle=ku,n.slice=Ia,n.sortBy=cp,n.sortedUniq=Ba,n.sortedUniqBy=Wa,n.split=_s,n.spread=ju,n.tail=Va,n.take=za,n.takeRight=Ha,n.takeRightWhile=qa,n.takeWhile=Ya,n.tap=eu,n.throttle=Bu,n.thru=nu,n.toArray=Mc,n.toPairs=qp,n.toPairsIn=Yp,n.toPath=Xs,n.toPlainObject=Pc,n.transform=Jc,n.unary=Wu,n.union=qf,n.unionBy=Yf,n.unionWith=Kf,n.uniq=Ka,n.uniqBy=Ga,n.uniqWith=$a,n.unset=ts,n.unzip=Xa,n.unzipWith=Za,n.update=es,n.updateWith=ns,n.values=rs,n.valuesIn=is,n.without=Gf,n.words=Ps,n.wrap=Vu,n.xor=$f,n.xorBy=Xf,n.xorWith=Zf,n.zip=Qf,n.zipObject=Qa,n.zipObjectDeep=Ja,n.zipWith=Jf,n.entries=qp,n.entriesIn=Yp,n.extend=Op,n.extendWith=Ip,Fs(n,n),n.add=ph,n.attempt=eh,n.camelCase=Kp,n.capitalize=cs,n.ceil=hh,n.clamp=os,n.clone=Hu,n.cloneDeep=Yu,n.cloneDeepWith=Ku,n.cloneWith=qu,n.conformsTo=Gu,n.deburr=ss,n.defaultTo=Is,n.divide=dh,n.endsWith=ls,n.eq=$u,n.escape=fs,n.escapeRegExp=ps,n.every=lu,n.find=np,n.findIndex=va,n.findKey=Ic,n.findLast=rp,n.findLastIndex=ga,n.findLastKey=Dc,n.floor=vh,n.forEach=vu,n.forEachRight=gu,n.forIn=Rc,n.forInRight=Lc,n.forOwn=Uc,n.forOwnRight=Fc,n.get=Wc,n.gt=yp,n.gte=_p,n.has=Vc,n.hasIn=zc,n.head=xa,n.identity=Ds,n.includes=mu,n.indexOf=wa,n.inRange=as,n.invoke=Bp,n.isArguments=bp,n.isArray=xp,n.isArrayBuffer=wp,n.isArrayLike=Xu,n.isArrayLikeObject=Zu,n.isBoolean=Qu,n.isBuffer=Cp,n.isDate=Mp,n.isElement=Ju,n.isEmpty=tc,n.isEqual=ec,n.isEqualWith=nc,n.isError=rc,n.isFinite=ic,n.isFunction=oc,n.isInteger=ac,n.isLength=uc,n.isMap=kp,n.isMatch=lc,n.isMatchWith=fc,n.isNaN=pc,n.isNative=hc,n.isNil=vc,n.isNull=dc,n.isNumber=gc,n.isObject=cc,n.isObjectLike=sc,n.isPlainObject=mc,n.isRegExp=Ep,n.isSafeInteger=yc,n.isSet=Tp,n.isString=_c,n.isSymbol=bc,n.isTypedArray=Sp,n.isUndefined=xc,n.isWeakMap=wc,n.isWeakSet=Cc,n.join=Ma,n.kebabCase=Gp,n.last=ka,n.lastIndexOf=Ea,n.lowerCase=$p,n.lowerFirst=Xp,n.lt=Pp,n.lte=Np,n.max=Qs,n.maxBy=Js,n.mean=tl,n.meanBy=el,n.min=nl,n.minBy=rl,n.stubArray=Hs,n.stubFalse=qs,n.stubObject=Ys,n.stubString=Ks,n.stubTrue=Gs,n.multiply=gh,n.nth=Ta,n.noConflict=js,n.noop=Bs,n.now=sp,n.pad=hs,n.padEnd=ds,n.padStart=vs,n.parseInt=gs,n.random=us,n.reduce=bu,n.reduceRight=xu,n.repeat=ms,n.replace=ys,n.result=Xc,n.round=mh,n.runInContext=t,n.sample=Cu,n.size=Eu,n.snakeCase=Zp,n.some=Tu,n.sortedIndex=Da,n.sortedIndexBy=Ra,n.sortedIndexOf=La,n.sortedLastIndex=Ua,n.sortedLastIndexBy=Fa,n.sortedLastIndexOf=ja,n.startCase=Qp,n.startsWith=bs,n.subtract=yh,n.sum=il,n.sumBy=ol,n.template=xs,n.times=$s,n.toFinite=kc,n.toInteger=Ec,n.toLength=Tc,n.toLower=ws,n.toNumber=Sc,n.toSafeInteger=Nc,n.toString=Ac,n.toUpper=Cs,n.trim=Ms,n.trimEnd=ks,n.trimStart=Es,n.truncate=Ts,n.unescape=Ss,n.uniqueId=Zs,n.upperCase=Jp,n.upperFirst=th,n.each=vu,n.eachRight=gu,n.first=xa,Fs(n,function(){var t={};return nr(n,function(e,r){bl.call(n.prototype,r)||(t[r]=e)}),t}(),{chain:!1}),n.VERSION=ot,s(["bind","bindKey","curry","curryRight","partial","partialRight"],function(t){n[t].placeholder=n}),s(["drop","take"],function(t,e){b.prototype[t]=function(n){n=n===it?1:$l(Ec(n),0);var r=this.__filtered__&&!e?new b(this):this.clone();return r.__filtered__?r.__takeCount__=Xl(n,r.__takeCount__):r.__views__.push({size:Xl(n,Ft),type:t+(r.__dir__<0?"Right":"")}),r},b.prototype[t+"Right"]=function(e){return this.reverse()[t](e).reverse()}}),s(["filter","map","takeWhile"],function(t,e){var n=e+1,r=n==At||n==It;b.prototype[t]=function(t){var e=this.clone();return e.__iteratees__.push({iteratee:ko(t,3),type:n}),e.__filtered__=e.__filtered__||r,e}}),s(["head","last"],function(t,e){var n="take"+(e?"Right":"");b.prototype[t]=function(){return this[n](1).value()[0]}}),s(["initial","tail"],function(t,e){var n="drop"+(e?"":"Right");b.prototype[t]=function(){return this.__filtered__?new b(this):this[n](1)}}),b.prototype.compact=function(){return this.filter(Ds)},b.prototype.find=function(t){return this.filter(t).head()},b.prototype.findLast=function(t){return this.reverse().find(t)},b.prototype.invokeMap=oi(function(t,e){return"function"==typeof t?new b(this):this.map(function(n){return Sr(n,t,e)})}),b.prototype.reject=function(t){return this.filter(Lu(ko(t)))},b.prototype.slice=function(t,e){t=Ec(t);var n=this;return n.__filtered__&&(t>0||e<0)?new b(n):(t<0?n=n.takeRight(-t):t&&(n=n.drop(t)),e!==it&&(e=Ec(e),n=e<0?n.dropRight(-e):n.take(e-t)),n)},b.prototype.takeRightWhile=function(t){return this.reverse().takeWhile(t).reverse()},b.prototype.toArray=function(){return this.take(Ft)},nr(b.prototype,function(t,e){var r=/^(?:filter|find|map|reject)|While$/.test(e),o=/^(?:head|last)$/.test(e),a=n[o?"take"+("last"==e?"Right":""):e],u=o||/^find/.test(e);a&&(n.prototype[e]=function(){var e=this.__wrapped__,c=o?[1]:arguments,s=e instanceof b,l=c[0],f=s||xp(e),p=function(t){var e=a.apply(n,g([t],c));return o&&h?e[0]:e};f&&r&&"function"==typeof l&&1!=l.length&&(s=f=!1);var h=this.__chain__,d=!!this.__actions__.length,v=u&&!h,m=s&&!d;if(!u&&f){e=m?e:new b(this);var y=t.apply(e,c);return y.__actions__.push({func:nu,args:[p],thisArg:it}),new i(y,h)}return v&&m?t.apply(this,c):(y=this.thru(p),v?o?y.value()[0]:y.value():y)})}),s(["pop","push","shift","sort","splice","unshift"],function(t){var e=vl[t],r=/^(?:push|sort|unshift)$/.test(t)?"tap":"thru",i=/^(?:pop|shift)$/.test(t);n.prototype[t]=function(){var t=arguments;if(i&&!this.__chain__){var n=this.value();return e.apply(xp(n)?n:[],t)}return this[r](function(n){return e.apply(xp(n)?n:[],t)})}}),nr(b.prototype,function(t,e){var r=n[e];if(r){var i=r.name+"",o=sf[i]||(sf[i]=[]);o.push({name:e,func:r})}}),sf[eo(it,yt).name]=[{name:"wrapper",func:it}],b.prototype.clone=P,b.prototype.reverse=Z,b.prototype.value=et,n.prototype.at=tp,n.prototype.chain=ru,n.prototype.commit=iu,n.prototype.next=ou,n.prototype.plant=uu,n.prototype.reverse=cu,n.prototype.toJSON=n.prototype.valueOf=n.prototype.value=su,n.prototype.first=n.prototype.head,Ll&&(n.prototype[Ll]=au),n},Mr=Cr();ur._=Mr,i=function(){return Mr}.call(e,n,e,r),!(i!==it&&(r.exports=i))}).call(this)}).call(e,n(99),n(100)(t))},function(t,e,n){"use strict";var r={remove:function(t){t._reactInternalInstance=void 0},get:function(t){return t._reactInternalInstance},has:function(t){return void 0!==t._reactInternalInstance},set:function(t,e){t._reactInternalInstance=e}};t.exports=r},function(t,e,n){"use strict";t.exports=n(26)},function(t,e,n){"use strict";var r=n(61);e.a=function(t){return t=n.i(r.a)(Math.abs(t)),t?t[1]:NaN}},function(t,e,n){"use strict";e.a=function(t,e){return t=+t,e-=t,function(n){return t+e*n}}},function(t,e,n){"use strict";var r=n(228);n.d(e,"a",function(){return r.a})},function(t,e,n){"use strict";function r(t,e){return(e-=t=+t)?function(n){return(n-t)/e}:n.i(h.a)(e)}function i(t){return function(e,n){var r=t(e=+e,n=+n);return function(t){return t<=e?0:t>=n?1:r(t)}}}function o(t){return function(e,n){var r=t(e=+e,n=+n);return function(t){return t<=0?e:t>=1?n:r(t)}}}function a(t,e,n,r){var i=t[0],o=t[1],a=e[0],u=e[1];return o<i?(i=n(o,i),a=r(u,a)):(i=n(i,o),a=r(a,u)),function(t){return a(i(t))}}function u(t,e,r,i){var o=Math.min(t.length,e.length)-1,a=new Array(o),u=new Array(o),c=-1;for(t[o]<t[0]&&(t=t.slice().reverse(),e=e.slice().reverse());++c<o;)a[c]=r(t[c],t[c+1]),u[c]=i(e[c],e[c+1]);return function(e){var r=n.i(l.c)(t,e,1,o)-1;return u[r](a[r](e))}}function c(t,e){return e.domain(t.domain()).range(t.range()).interpolate(t.interpolate()).clamp(t.clamp())}function s(t,e){function n(){return s=Math.min(g.length,m.length)>2?u:a,l=h=null,c}function c(e){return(l||(l=s(g,m,_?i(t):t,y)))(+e)}var s,l,h,g=v,m=v,y=f.b,_=!1;return c.invert=function(t){return(h||(h=s(m,g,r,_?o(e):e)))(+t)},c.domain=function(t){return arguments.length?(g=p.a.call(t,d.a),n()):g.slice()},c.range=function(t){return arguments.length?(m=p.b.call(t),n()):m.slice()},c.rangeRound=function(t){return m=p.b.call(t),y=f.c,n()},c.clamp=function(t){return arguments.length?(_=!!t,n()):_},c.interpolate=function(t){return arguments.length?(y=t,n()):y},n()}var l=n(12),f=n(31),p=n(16),h=n(65),d=n(126);e.b=r,e.c=c,e.a=s;var v=[0,1]},function(t,e,n){"use strict";function r(t,e,n){t._context.bezierCurveTo((2*t._x0+t._x1)/3,(2*t._y0+t._y1)/3,(t._x0+2*t._x1)/3,(t._y0+2*t._y1)/3,(t._x0+4*t._x1+e)/6,(t._y0+4*t._y1+n)/6)}function i(t){this._context=t}e.c=r,e.b=i,i.prototype={
areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN,this._point=0},lineEnd:function(){switch(this._point){case 3:r(this,this._x1,this._y1);case 2:this._context.lineTo(this._x1,this._y1)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3,this._context.lineTo((5*this._x0+this._x1)/6,(5*this._y0+this._y1)/6);default:r(this,t,e)}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e}},e.a=function(t){return new i(t)}},function(t,e,n){"use strict";function r(t,e,n){t._context.bezierCurveTo(t._x1+t._k*(t._x2-t._x0),t._y1+t._k*(t._y2-t._y0),t._x2+t._k*(t._x1-e),t._y2+t._k*(t._y1-n),t._x2,t._y2)}function i(t,e){this._context=t,this._k=(1-e)/6}e.c=r,e.b=i,i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:r(this,this._x1,this._y1)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2,this._x1=t,this._y1=e;break;case 2:this._point=3;default:r(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new i(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t){this._context=t}r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;default:this._context.lineTo(t,e)}}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";e.a=function(){}},function(t,e,n){"use strict";function r(t){return"topMouseUp"===t||"topTouchEnd"===t||"topTouchCancel"===t}function i(t){return"topMouseMove"===t||"topTouchMove"===t}function o(t){return"topMouseDown"===t||"topTouchStart"===t}function a(t,e,n,r){var i=t.type||"unknown-event";t.currentTarget=m.getNodeFromInstance(r),e?v.invokeGuardedCallbackWithCatch(i,n,t):v.invokeGuardedCallback(i,n,t),t.currentTarget=null}function u(t,e){var n=t._dispatchListeners,r=t._dispatchInstances;if(Array.isArray(n))for(var i=0;i<n.length&&!t.isPropagationStopped();i++)a(t,e,n[i],r[i]);else n&&a(t,e,n,r);t._dispatchListeners=null,t._dispatchInstances=null}function c(t){var e=t._dispatchListeners,n=t._dispatchInstances;if(Array.isArray(e)){for(var r=0;r<e.length&&!t.isPropagationStopped();r++)if(e[r](t,n[r]))return n[r]}else if(e&&e(t,n))return n;return null}function s(t){var e=c(t);return t._dispatchInstances=null,t._dispatchListeners=null,e}function l(t){var e=t._dispatchListeners,n=t._dispatchInstances;Array.isArray(e)?d("103"):void 0,t.currentTarget=e?m.getNodeFromInstance(n):null;var r=e?e(t):null;return t.currentTarget=null,t._dispatchListeners=null,t._dispatchInstances=null,r}function f(t){return!!t._dispatchListeners}var p,h,d=n(2),v=n(87),g=(n(0),n(1),{injectComponentTree:function(t){p=t},injectTreeTraversal:function(t){h=t}}),m={isEndish:r,isMoveish:i,isStartish:o,executeDirectDispatch:l,executeDispatchesInOrder:u,executeDispatchesInOrderStopAtTrue:s,hasDispatches:f,getInstanceFromNode:function(t){return p.getInstanceFromNode(t)},getNodeFromInstance:function(t){return p.getNodeFromInstance(t)},isAncestor:function(t,e){return h.isAncestor(t,e)},getLowestCommonAncestor:function(t,e){return h.getLowestCommonAncestor(t,e)},getParentInstance:function(t){return h.getParentInstance(t)},traverseTwoPhase:function(t,e,n){return h.traverseTwoPhase(t,e,n)},traverseEnterLeave:function(t,e,n,r,i){return h.traverseEnterLeave(t,e,n,r,i)},injection:g};t.exports=m},function(t,e,n){"use strict";function r(t){return Object.prototype.hasOwnProperty.call(t,v)||(t[v]=h++,f[t[v]]={}),f[t[v]]}var i,o=n(3),a=n(83),u=n(360),c=n(89),s=n(393),l=n(94),f={},p=!1,h=0,d={topAbort:"abort",topAnimationEnd:s("animationend")||"animationend",topAnimationIteration:s("animationiteration")||"animationiteration",topAnimationStart:s("animationstart")||"animationstart",topBlur:"blur",topCanPlay:"canplay",topCanPlayThrough:"canplaythrough",topChange:"change",topClick:"click",topCompositionEnd:"compositionend",topCompositionStart:"compositionstart",topCompositionUpdate:"compositionupdate",topContextMenu:"contextmenu",topCopy:"copy",topCut:"cut",topDoubleClick:"dblclick",topDrag:"drag",topDragEnd:"dragend",topDragEnter:"dragenter",topDragExit:"dragexit",topDragLeave:"dragleave",topDragOver:"dragover",topDragStart:"dragstart",topDrop:"drop",topDurationChange:"durationchange",topEmptied:"emptied",topEncrypted:"encrypted",topEnded:"ended",topError:"error",topFocus:"focus",topInput:"input",topKeyDown:"keydown",topKeyPress:"keypress",topKeyUp:"keyup",topLoadedData:"loadeddata",topLoadedMetadata:"loadedmetadata",topLoadStart:"loadstart",topMouseDown:"mousedown",topMouseMove:"mousemove",topMouseOut:"mouseout",topMouseOver:"mouseover",topMouseUp:"mouseup",topPaste:"paste",topPause:"pause",topPlay:"play",topPlaying:"playing",topProgress:"progress",topRateChange:"ratechange",topScroll:"scroll",topSeeked:"seeked",topSeeking:"seeking",topSelectionChange:"selectionchange",topStalled:"stalled",topSuspend:"suspend",topTextInput:"textInput",topTimeUpdate:"timeupdate",topTouchCancel:"touchcancel",topTouchEnd:"touchend",topTouchMove:"touchmove",topTouchStart:"touchstart",topTransitionEnd:s("transitionend")||"transitionend",topVolumeChange:"volumechange",topWaiting:"waiting",topWheel:"wheel"},v="_reactListenersID"+String(Math.random()).slice(2),g=o({},u,{ReactEventListener:null,injection:{injectReactEventListener:function(t){t.setHandleTopLevel(g.handleTopLevel),g.ReactEventListener=t}},setEnabled:function(t){g.ReactEventListener&&g.ReactEventListener.setEnabled(t)},isEnabled:function(){return!(!g.ReactEventListener||!g.ReactEventListener.isEnabled())},listenTo:function(t,e){for(var n=e,i=r(n),o=a.registrationNameDependencies[t],u=0;u<o.length;u++){var c=o[u];i.hasOwnProperty(c)&&i[c]||("topWheel"===c?l("wheel")?g.ReactEventListener.trapBubbledEvent("topWheel","wheel",n):l("mousewheel")?g.ReactEventListener.trapBubbledEvent("topWheel","mousewheel",n):g.ReactEventListener.trapBubbledEvent("topWheel","DOMMouseScroll",n):"topScroll"===c?l("scroll",!0)?g.ReactEventListener.trapCapturedEvent("topScroll","scroll",n):g.ReactEventListener.trapBubbledEvent("topScroll","scroll",g.ReactEventListener.WINDOW_HANDLE):"topFocus"===c||"topBlur"===c?(l("focus",!0)?(g.ReactEventListener.trapCapturedEvent("topFocus","focus",n),g.ReactEventListener.trapCapturedEvent("topBlur","blur",n)):l("focusin")&&(g.ReactEventListener.trapBubbledEvent("topFocus","focusin",n),g.ReactEventListener.trapBubbledEvent("topBlur","focusout",n)),i.topBlur=!0,i.topFocus=!0):d.hasOwnProperty(c)&&g.ReactEventListener.trapBubbledEvent(c,d[c],n),i[c]=!0)}},trapBubbledEvent:function(t,e,n){return g.ReactEventListener.trapBubbledEvent(t,e,n)},trapCapturedEvent:function(t,e,n){return g.ReactEventListener.trapCapturedEvent(t,e,n)},supportsEventPageXY:function(){if(!document.createEvent)return!1;var t=document.createEvent("MouseEvent");return null!=t&&"pageX"in t},ensureScrollValueMonitoring:function(){if(void 0===i&&(i=g.supportsEventPageXY()),!i&&!p){var t=c.refreshScrollValues;g.ReactEventListener.monitorScrollValue(t),p=!0}}});t.exports=g},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o=n(89),a=n(92),u={screenX:null,screenY:null,clientX:null,clientY:null,ctrlKey:null,shiftKey:null,altKey:null,metaKey:null,getModifierState:a,button:function(t){var e=t.button;return"which"in t?e:2===e?2:4===e?1:0},buttons:null,relatedTarget:function(t){return t.relatedTarget||(t.fromElement===t.srcElement?t.toElement:t.fromElement)},pageX:function(t){return"pageX"in t?t.pageX:t.clientX+o.currentScrollLeft},pageY:function(t){return"pageY"in t?t.pageY:t.clientY+o.currentScrollTop}};i.augmentClass(r,u),t.exports=r},function(t,e,n){"use strict";var r=n(2),i=(n(0),{}),o={reinitializeTransaction:function(){this.transactionWrappers=this.getTransactionWrappers(),this.wrapperInitData?this.wrapperInitData.length=0:this.wrapperInitData=[],this._isInTransaction=!1},_isInTransaction:!1,getTransactionWrappers:null,isInTransaction:function(){return!!this._isInTransaction},perform:function(t,e,n,i,o,a,u,c){this.isInTransaction()?r("27"):void 0;var s,l;try{this._isInTransaction=!0,s=!0,this.initializeAll(0),l=t.call(e,n,i,o,a,u,c),s=!1}finally{try{if(s)try{this.closeAll(0)}catch(t){}else this.closeAll(0)}finally{this._isInTransaction=!1}}return l},initializeAll:function(t){for(var e=this.transactionWrappers,n=t;n<e.length;n++){var r=e[n];try{this.wrapperInitData[n]=i,this.wrapperInitData[n]=r.initialize?r.initialize.call(this):null}finally{if(this.wrapperInitData[n]===i)try{this.initializeAll(n+1)}catch(t){}}}},closeAll:function(t){this.isInTransaction()?void 0:r("28");for(var e=this.transactionWrappers,n=t;n<e.length;n++){var o,a=e[n],u=this.wrapperInitData[n];try{o=!0,u!==i&&a.close&&a.close.call(this,u),o=!1}finally{if(o)try{this.closeAll(n+1)}catch(t){}}}this.wrapperInitData.length=0}};t.exports=o},function(t,e,n){"use strict";function r(t){var e=""+t,n=o.exec(e);if(!n)return e;var r,i="",a=0,u=0;for(a=n.index;a<e.length;a++){switch(e.charCodeAt(a)){case 34:r="&quot;";break;case 38:r="&amp;";break;case 39:r="&#x27;";break;case 60:r="&lt;";break;case 62:r="&gt;";break;default:continue}u!==a&&(i+=e.substring(u,a)),u=a+1,i+=r}return u!==a?i+e.substring(u,a):i}function i(t){return"boolean"==typeof t||"number"==typeof t?""+t:r(t)}var o=/["'&<>]/;t.exports=i},function(t,e,n){"use strict";var r,i=n(6),o=n(82),a=/^[ \r\n\t\f]/,u=/<(!--|link|noscript|meta|script|style)[ \r\n\t\f\/>]/,c=n(90),s=c(function(t,e){if(t.namespaceURI!==o.svg||"innerHTML"in t)t.innerHTML=e;else{r=r||document.createElement("div"),r.innerHTML="<svg>"+e+"</svg>";for(var n=r.firstChild;n.firstChild;)t.appendChild(n.firstChild)}});if(i.canUseDOM){var l=document.createElement("div");l.innerHTML=" ",""===l.innerHTML&&(s=function(t,e){if(t.parentNode&&t.parentNode.replaceChild(t,t),a.test(e)||"<"===e[0]&&u.test(e)){t.innerHTML=String.fromCharCode(65279)+e;var n=t.firstChild;1===n.data.length?t.removeChild(n):n.deleteData(0,1)}else t.innerHTML=e}),l=null}t.exports=s},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default={colors:{RdBu:["rgb(255, 13, 87)","rgb(30, 136, 229)"],GnPR:["rgb(24, 196, 93)","rgb(124, 82, 255)"],CyPU:["#0099C6","#990099"],PkYg:["#DD4477","#66AA00"],DrDb:["#B82E2E","#316395"],LpLb:["#994499","#22AA99"],YlDp:["#AAAA11","#6633CC"],OrId:["#E67300","#3E0099"]},gray:"#777"}},function(t,e,n){"use strict";var r=n(29);e.a=function(t,e,n){if(null==n&&(n=r.a),i=t.length){if((e=+e)<=0||i<2)return+n(t[0],0,t);if(e>=1)return+n(t[i-1],i-1,t);var i,o=(i-1)*e,a=Math.floor(o),u=+n(t[a],a,t),c=+n(t[a+1],a+1,t);return u+(c-u)*(o-a)}}},function(t,e,n){"use strict";function r(){}function i(t,e){var n=new r;if(t instanceof r)t.each(function(t,e){n.set(e,t)});else if(Array.isArray(t)){var i,o=-1,a=t.length;if(null==e)for(;++o<a;)n.set(o,t[o]);else for(;++o<a;)n.set(e(i=t[o],o,t),i)}else if(t)for(var u in t)n.set(u,t[u]);return n}n.d(e,"b",function(){return o});var o="$";r.prototype=i.prototype={constructor:r,has:function(t){return o+t in this},get:function(t){return this[o+t]},set:function(t,e){return this[o+t]=e,this},remove:function(t){var e=o+t;return e in this&&delete this[e]},clear:function(){for(var t in this)t[0]===o&&delete this[t]},keys:function(){var t=[];for(var e in this)e[0]===o&&t.push(e.slice(1));return t},values:function(){var t=[];for(var e in this)e[0]===o&&t.push(this[e]);return t},entries:function(){var t=[];for(var e in this)e[0]===o&&t.push({key:e.slice(1),value:this[e]});return t},size:function(){var t=0;for(var e in this)e[0]===o&&++t;return t},empty:function(){for(var t in this)if(t[0]===o)return!1;return!0},each:function(t){for(var e in this)e[0]===o&&t(this[e],e.slice(1),this)}},e.a=i},function(t,e,n){"use strict";function r(){}function i(t){var e;return t=(t+"").trim().toLowerCase(),(e=x.exec(t))?(e=parseInt(e[1],16),new s(e>>8&15|e>>4&240,e>>4&15|240&e,(15&e)<<4|15&e,1)):(e=w.exec(t))?o(parseInt(e[1],16)):(e=C.exec(t))?new s(e[1],e[2],e[3],1):(e=M.exec(t))?new s(255*e[1]/100,255*e[2]/100,255*e[3]/100,1):(e=k.exec(t))?a(e[1],e[2],e[3],e[4]):(e=E.exec(t))?a(255*e[1]/100,255*e[2]/100,255*e[3]/100,e[4]):(e=T.exec(t))?l(e[1],e[2]/100,e[3]/100,1):(e=S.exec(t))?l(e[1],e[2]/100,e[3]/100,e[4]):P.hasOwnProperty(t)?o(P[t]):"transparent"===t?new s(NaN,NaN,NaN,0):null}function o(t){return new s(t>>16&255,t>>8&255,255&t,1)}function a(t,e,n,r){return r<=0&&(t=e=n=NaN),new s(t,e,n,r)}function u(t){return t instanceof r||(t=i(t)),t?(t=t.rgb(),new s(t.r,t.g,t.b,t.opacity)):new s}function c(t,e,n,r){return 1===arguments.length?u(t):new s(t,e,n,null==r?1:r)}function s(t,e,n,r){this.r=+t,this.g=+e,this.b=+n,this.opacity=+r}function l(t,e,n,r){return r<=0?t=e=n=NaN:n<=0||n>=1?t=e=NaN:e<=0&&(t=NaN),new h(t,e,n,r)}function f(t){if(t instanceof h)return new h(t.h,t.s,t.l,t.opacity);if(t instanceof r||(t=i(t)),!t)return new h;if(t instanceof h)return t;t=t.rgb();var e=t.r/255,n=t.g/255,o=t.b/255,a=Math.min(e,n,o),u=Math.max(e,n,o),c=NaN,s=u-a,l=(u+a)/2;return s?(c=e===u?(n-o)/s+6*(n<o):n===u?(o-e)/s+2:(e-n)/s+4,s/=l<.5?u+a:2-u-a,c*=60):s=l>0&&l<1?0:c,new h(c,s,l,t.opacity)}function p(t,e,n,r){return 1===arguments.length?f(t):new h(t,e,n,null==r?1:r)}function h(t,e,n,r){this.h=+t,this.s=+e,this.l=+n,this.opacity=+r}function d(t,e,n){return 255*(t<60?e+(n-e)*t/60:t<180?n:t<240?e+(n-e)*(240-t)/60:e)}var v=n(60);e.f=r,n.d(e,"h",function(){return g}),n.d(e,"g",function(){return m}),e.a=i,e.e=u,e.b=c,e.d=s,e.c=p;var g=.7,m=1/g,y="\\s*([+-]?\\d+)\\s*",_="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*",b="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*",x=/^#([0-9a-f]{3})$/,w=/^#([0-9a-f]{6})$/,C=new RegExp("^rgb\\("+[y,y,y]+"\\)$"),M=new RegExp("^rgb\\("+[b,b,b]+"\\)$"),k=new RegExp("^rgba\\("+[y,y,y,_]+"\\)$"),E=new RegExp("^rgba\\("+[b,b,b,_]+"\\)$"),T=new RegExp("^hsl\\("+[_,b,b]+"\\)$"),S=new RegExp("^hsla\\("+[_,b,b,_]+"\\)$"),P={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};n.i(v.a)(r,i,{displayable:function(){return this.rgb().displayable()},toString:function(){return this.rgb()+""}}),n.i(v.a)(s,c,n.i(v.b)(r,{brighter:function(t){return t=null==t?m:Math.pow(m,t),new s(this.r*t,this.g*t,this.b*t,this.opacity)},darker:function(t){return t=null==t?g:Math.pow(g,t),new s(this.r*t,this.g*t,this.b*t,this.opacity)},rgb:function(){return this},displayable:function(){return 0<=this.r&&this.r<=255&&0<=this.g&&this.g<=255&&0<=this.b&&this.b<=255&&0<=this.opacity&&this.opacity<=1},toString:function(){var t=this.opacity;return t=isNaN(t)?1:Math.max(0,Math.min(1,t)),(1===t?"rgb(":"rgba(")+Math.max(0,Math.min(255,Math.round(this.r)||0))+", "+Math.max(0,Math.min(255,Math.round(this.g)||0))+", "+Math.max(0,Math.min(255,Math.round(this.b)||0))+(1===t?")":", "+t+")")}})),n.i(v.a)(h,p,n.i(v.b)(r,{brighter:function(t){return t=null==t?m:Math.pow(m,t),new h(this.h,this.s,this.l*t,this.opacity)},darker:function(t){return t=null==t?g:Math.pow(g,t),new h(this.h,this.s,this.l*t,this.opacity)},rgb:function(){var t=this.h%360+360*(this.h<0),e=isNaN(t)||isNaN(this.s)?0:this.s,n=this.l,r=n+(n<.5?n:1-n)*e,i=2*n-r;return new s(d(t>=240?t-240:t+120,i,r),d(t,i,r),d(t<120?t+240:t-120,i,r),this.opacity)},displayable:function(){return(0<=this.s&&this.s<=1||isNaN(this.s))&&0<=this.l&&this.l<=1&&0<=this.opacity&&this.opacity<=1}}))},function(t,e,n){"use strict";function r(t,e){var n=Object.create(t.prototype);for(var r in e)n[r]=e[r];return n}e.b=r,e.a=function(t,e,n){t.prototype=e.prototype=n,n.constructor=t}},function(t,e,n){"use strict";e.a=function(t,e){if((n=(t=e?t.toExponential(e-1):t.toExponential()).indexOf("e"))<0)return null;var n,r=t.slice(0,n);return[r.length>1?r[0]+r.slice(2):r,+t.slice(n+1)]}},function(t,e,n){"use strict";function r(t,e,n,r,i){var o=t*t,a=o*t;return((1-3*t+3*o-a)*e+(4-6*o+3*a)*n+(1+3*t+3*o-3*a)*r+a*i)/6}e.b=r,e.a=function(t){var e=t.length-1;return function(n){var i=n<=0?n=0:n>=1?(n=1,e-1):Math.floor(n*e),o=t[i],a=t[i+1],u=i>0?t[i-1]:2*o-a,c=i<e-1?t[i+2]:2*a-o;return r((n-i/e)*e,u,o,a,c)}}},function(t,e,n){"use strict";var r=n(10),i=n(123),o=n(118),a=n(121),u=n(43),c=n(122),s=n(124),l=n(120);e.a=function(t,e){var f,p=typeof e;return null==e||"boolean"===p?n.i(l.a)(e):("number"===p?u.a:"string"===p?(f=n.i(r.color)(e))?(e=f,i.a):s.a:e instanceof r.color?i.a:e instanceof Date?a.a:Array.isArray(e)?o.a:isNaN(e)?c.a:u.a)(t,e)}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(229);n.d(e,"scaleBand",function(){return r.a}),n.d(e,"scalePoint",function(){return r.b});var i=n(235);n.d(e,"scaleIdentity",function(){return i.a});var o=n(34);n.d(e,"scaleLinear",function(){return o.a});var a=n(236);n.d(e,"scaleLog",function(){return a.a});var u=n(127);n.d(e,"scaleOrdinal",function(){return u.a}),n.d(e,"scaleImplicit",function(){return u.b});var c=n(237);n.d(e,"scalePow",function(){return c.a}),n.d(e,"scaleSqrt",function(){return c.b});var s=n(238);n.d(e,"scaleQuantile",function(){return s.a});var l=n(239);n.d(e,"scaleQuantize",function(){return l.a});var f=n(242);n.d(e,"scaleThreshold",function(){return f.a});var p=n(128);n.d(e,"scaleTime",function(){return p.a});var h=n(244);n.d(e,"scaleUtc",function(){return h.a});var d=n(230);n.d(e,"schemeCategory10",function(){return d.a});var v=n(232);n.d(e,"schemeCategory20b",function(){return v.a});var g=n(233);n.d(e,"schemeCategory20c",function(){return g.a});var m=n(231);n.d(e,"schemeCategory20",function(){return m.a});var y=n(234);n.d(e,"interpolateCubehelixDefault",function(){return y.a});var _=n(240);n.d(e,"interpolateRainbow",function(){return _.a}),n.d(e,"interpolateWarm",function(){return _.b}),n.d(e,"interpolateCool",function(){return _.c});var b=n(245);n.d(e,"interpolateViridis",function(){return b.a}),n.d(e,"interpolateMagma",function(){return b.b}),n.d(e,"interpolateInferno",function(){return b.c}),n.d(e,"interpolatePlasma",function(){return b.d});var x=n(241);n.d(e,"scaleSequential",function(){return x.a})},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";function r(t){return function(){var e=this.ownerDocument,n=this.namespaceURI;return n===a.b&&e.documentElement.namespaceURI===a.b?e.createElement(t):e.createElementNS(n,t)}}function i(t){return function(){return this.ownerDocument.createElementNS(t.space,t.local)}}var o=n(67),a=n(68);e.a=function(t){var e=n.i(o.a)(t);return(e.local?i:r)(e)}},function(t,e,n){"use strict";var r=n(68);e.a=function(t){var e=t+="",n=e.indexOf(":");return n>=0&&"xmlns"!==(e=t.slice(0,n))&&(t=t.slice(n+1)),r.a.hasOwnProperty(e)?{space:r.a[e],local:t}:t}},function(t,e,n){"use strict";n.d(e,"b",function(){return r});var r="http://www.w3.org/1999/xhtml";e.a={svg:"http://www.w3.org/2000/svg",xhtml:r,xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"}},function(t,e,n){"use strict";e.a=function(t,e){var n=t.ownerSVGElement||t;if(n.createSVGPoint){var r=n.createSVGPoint();return r.x=e.clientX,r.y=e.clientY,r=r.matrixTransform(t.getScreenCTM().inverse()),[r.x,r.y]}var i=t.getBoundingClientRect();return[e.clientX-i.left-t.clientLeft,e.clientY-i.top-t.clientTop]}},function(t,e,n){"use strict";function r(t,e,n){return t=i(t,e,n),function(e){var n=e.relatedTarget;n&&(n===this||8&n.compareDocumentPosition(this))||t.call(this,e)}}function i(t,e,n){return function(r){var i=l;l=r;try{t.call(this,this.__data__,e,n)}finally{l=i}}}function o(t){return t.trim().split(/^|\s+/).map(function(t){var e="",n=t.indexOf(".");return n>=0&&(e=t.slice(n+1),t=t.slice(0,n)),{type:t,name:e}})}function a(t){return function(){var e=this.__on;if(e){for(var n,r=0,i=-1,o=e.length;r<o;++r)n=e[r],t.type&&n.type!==t.type||n.name!==t.name?e[++i]=n:this.removeEventListener(n.type,n.listener,n.capture);++i?e.length=i:delete this.__on}}}function u(t,e,n){var o=s.hasOwnProperty(t.type)?r:i;return function(r,i,a){var u,c=this.__on,s=o(e,i,a);if(c)for(var l=0,f=c.length;l<f;++l)if((u=c[l]).type===t.type&&u.name===t.name)return this.removeEventListener(u.type,u.listener,u.capture),this.addEventListener(u.type,u.listener=s,u.capture=n),void(u.value=e);this.addEventListener(t.type,s,n),u={type:t.type,name:t.name,value:e,listener:s,capture:n},c?c.push(u):this.__on=[u]}}function c(t,e,n,r){var i=l;t.sourceEvent=l,l=t;try{return e.apply(n,r)}finally{l=i}}n.d(e,"a",function(){return l}),e.b=c;var s={},l=null;if("undefined"!=typeof document){var f=document.documentElement;"onmouseenter"in f||(s={mouseenter:"mouseover",mouseleave:"mouseout"})}e.c=function(t,e,n){var r,i,c=o(t+""),s=c.length;{if(!(arguments.length<2)){for(l=e?u:a,null==n&&(n=!1),r=0;r<s;++r)this.each(l(c[r],e,n));return this}var l=this.node().__on;if(l)for(var f,p=0,h=l.length;p<h;++p)for(r=0,f=l[p];r<s;++r)if((i=c[r]).type===f.type&&i.name===f.name)return f.value}}},function(t,e,n){"use strict";function r(){}e.a=function(t){return null==t?r:function(){return this.querySelector(t)}}},function(t,e,n){"use strict";var r=n(70);e.a=function(){for(var t,e=r.a;t=e.sourceEvent;)e=t;return e}},function(t,e,n){"use strict";e.a=function(t){return t.ownerDocument&&t.ownerDocument.defaultView||t.document&&t||t.defaultView}},function(t,e,n){"use strict";function r(t,e,n){var r=t._x1,i=t._y1,a=t._x2,u=t._y2;if(t._l01_a>o.a){var c=2*t._l01_2a+3*t._l01_a*t._l12_a+t._l12_2a,s=3*t._l01_a*(t._l01_a+t._l12_a);r=(r*c-t._x0*t._l12_2a+t._x2*t._l01_2a)/s,i=(i*c-t._y0*t._l12_2a+t._y2*t._l01_2a)/s}if(t._l23_a>o.a){var l=2*t._l23_2a+3*t._l23_a*t._l12_a+t._l12_2a,f=3*t._l23_a*(t._l23_a+t._l12_a);a=(a*l+t._x1*t._l23_2a-e*t._l12_2a)/f,u=(u*l+t._y1*t._l23_2a-n*t._l12_2a)/f}t._context.bezierCurveTo(r,i,a,u,t._x2,t._y2)}function i(t,e){this._context=t,this._alpha=e}var o=n(35),a=n(47);e.b=r,i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:this.point(this._x2,this._y2)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){if(t=+t,e=+e,this._point){var n=this._x2-t,i=this._y2-e;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(n*n+i*i,this._alpha))}switch(this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3;default:r(this,t,e)}this._l01_a=this._l12_a,this._l12_a=this._l23_a,this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a,this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return e?new i(t,e):new a.b(t,0)}return n.alpha=function(e){return t(+e)},n}(.5)},function(t,e,n){"use strict";var r=n(44),i=n(19),o=n(48),a=n(139);e.a=function(){function t(t){var i,o,a,p=t.length,h=!1;for(null==s&&(f=l(a=n.i(r.a)())),i=0;i<=p;++i)!(i<p&&c(o=t[i],i,t))===h&&((h=!h)?f.lineStart():f.lineEnd()),h&&f.point(+e(o,i,t),+u(o,i,t));if(a)return f=null,a+""||null}var e=a.a,u=a.b,c=n.i(i.a)(!0),s=null,l=o.a,f=null;return t.x=function(r){return arguments.length?(e="function"==typeof r?r:n.i(i.a)(+r),t):e},t.y=function(e){return arguments.length?(u="function"==typeof e?e:n.i(i.a)(+e),t):u},t.defined=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(!!e),t):c},t.curve=function(e){return arguments.length?(l=e,null!=s&&(f=l(s)),t):l},t.context=function(e){return arguments.length?(null==e?s=f=null:f=l(s=e),t):s},t}},function(t,e,n){"use strict";function r(t){for(var e,n=0,r=-1,i=t.length;++r<i;)(e=+t[r][1])&&(n+=e);return n}var i=n(37);e.b=r,e.a=function(t){var e=t.map(r);return n.i(i.a)(t).sort(function(t,n){return e[t]-e[n]})}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(78);n.d(e,"timeFormatDefaultLocale",function(){return r.a}),n.d(e,"timeFormat",function(){return r.b}),n.d(e,"timeParse",function(){return r.c}),n.d(e,"utcFormat",function(){return r.d}),n.d(e,"utcParse",function(){return r.e});var i=n(149);n.d(e,"timeFormatLocale",function(){return i.a});var o=n(148);n.d(e,"isoFormat",function(){return o.a});var a=n(303);n.d(e,"isoParse",function(){return a.a})},function(t,e,n){"use strict";function r(t){return o=n.i(i.a)(t),a=o.format,u=o.parse,c=o.utcFormat,s=o.utcParse,o}var i=n(149);n.d(e,"b",function(){return a}),n.d(e,"c",function(){return u}),n.d(e,"d",function(){return c}),n.d(e,"e",function(){return s}),e.a=r;var o,a,u,c,s;r({dateTime:"%x, %X",date:"%-m/%-d/%Y",time:"%-I:%M:%S %p",periods:["AM","PM"],days:["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],shortDays:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],months:["January","February","March","April","May","June","July","August","September","October","November","December"],shortMonths:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]})},function(t,e,n){"use strict";var r=(n(5),n(306));n.d(e,"t",function(){return r.a}),n.d(e,"n",function(){return r.a});var i=n(309);n.d(e,"s",function(){return i.a}),n.d(e,"m",function(){return i.a});var o=n(307);n.d(e,"r",function(){return o.a});var a=n(305);n.d(e,"q",function(){return a.a});var u=n(304);n.d(e,"a",function(){return u.a});var c=n(316);n.d(e,"p",function(){return c.a}),n.d(e,"c",function(){return c.a}),n.d(e,"d",function(){return c.b});var s=n(308);n.d(e,"o",function(){return s.a});var l=n(317);n.d(e,"b",function(){return l.a});var f=n(312);n.d(e,"l",function(){return f.a});var p=n(311);n.d(e,"k",function(){return p.a});var h=n(310);n.d(e,"e",function(){return h.a});var d=n(314);n.d(e,"j",function(){return d.a}),n.d(e,"g",function(){return d.a}),n.d(e,"h",function(){return d.b});var v=n(313);n.d(e,"i",function(){return v.a});var g=n(315);n.d(e,"f",function(){return g.a})},function(t,e,n){"use strict";function r(t,e){return t===e?0!==t||0!==e||1/t===1/e:t!==t&&e!==e}function i(t,e){if(r(t,e))return!0;if("object"!=typeof t||null===t||"object"!=typeof e||null===e)return!1;var n=Object.keys(t),i=Object.keys(e);if(n.length!==i.length)return!1;for(var a=0;a<n.length;a++)if(!o.call(e,n[a])||!r(t[n[a]],e[n[a]]))return!1;return!0}var o=Object.prototype.hasOwnProperty;t.exports=i},function(t,e,n){"use strict";function r(t,e){return Array.isArray(e)&&(e=e[1]),e?e.nextSibling:t.firstChild}function i(t,e,n){l.insertTreeBefore(t,e,n)}function o(t,e,n){Array.isArray(e)?u(t,e[0],e[1],n):v(t,e,n)}function a(t,e){if(Array.isArray(e)){var n=e[1];e=e[0],c(t,e,n),t.removeChild(n)}t.removeChild(e)}function u(t,e,n,r){for(var i=e;;){var o=i.nextSibling;if(v(t,i,r),i===n)break;i=o}}function c(t,e,n){for(;;){var r=e.nextSibling;if(r===n)break;t.removeChild(r)}}function s(t,e,n){var r=t.parentNode,i=t.nextSibling;i===e?n&&v(r,document.createTextNode(n),i):n?(d(i,n),c(r,i,e)):c(r,t,e)}var l=n(20),f=n(336),p=(n(4),n(9),n(90)),h=n(55),d=n(171),v=p(function(t,e,n){t.insertBefore(e,n)}),g=f.dangerouslyReplaceNodeWithMarkup,m={dangerouslyReplaceNodeWithMarkup:g,replaceDelimitedText:s,processUpdates:function(t,e){for(var n=0;n<e.length;n++){var u=e[n];switch(u.type){case"INSERT_MARKUP":i(t,u.content,r(t,u.afterNode));break;case"MOVE_EXISTING":o(t,u.fromNode,r(t,u.afterNode));break;case"SET_MARKUP":h(t,u.content);break;case"TEXT_CONTENT":d(t,u.content);break;case"REMOVE_NODE":a(t,u.fromNode)}}}};t.exports=m},function(t,e,n){"use strict";var r={html:"http://www.w3.org/1999/xhtml",mathml:"http://www.w3.org/1998/Math/MathML",svg:"http://www.w3.org/2000/svg"};t.exports=r},function(t,e,n){"use strict";function r(){if(u)for(var t in c){var e=c[t],n=u.indexOf(t);if(n>-1?void 0:a("96",t),!s.plugins[n]){e.extractEvents?void 0:a("97",t),s.plugins[n]=e;var r=e.eventTypes;for(var o in r)i(r[o],e,o)?void 0:a("98",o,t)}}}function i(t,e,n){s.eventNameDispatchConfigs.hasOwnProperty(n)?a("99",n):void 0,s.eventNameDispatchConfigs[n]=t;var r=t.phasedRegistrationNames;if(r){for(var i in r)if(r.hasOwnProperty(i)){var u=r[i];o(u,e,n)}return!0}return!!t.registrationName&&(o(t.registrationName,e,n),!0)}function o(t,e,n){s.registrationNameModules[t]?a("100",t):void 0,s.registrationNameModules[t]=e,s.registrationNameDependencies[t]=e.eventTypes[n].dependencies}var a=n(2),u=(n(0),null),c={},s={plugins:[],eventNameDispatchConfigs:{},registrationNameModules:{},registrationNameDependencies:{},possibleRegistrationNames:null,injectEventPluginOrder:function(t){
u?a("101"):void 0,u=Array.prototype.slice.call(t),r()},injectEventPluginsByName:function(t){var e=!1;for(var n in t)if(t.hasOwnProperty(n)){var i=t[n];c.hasOwnProperty(n)&&c[n]===i||(c[n]?a("102",n):void 0,c[n]=i,e=!0)}e&&r()},getPluginModuleForEvent:function(t){var e=t.dispatchConfig;if(e.registrationName)return s.registrationNameModules[e.registrationName]||null;if(void 0!==e.phasedRegistrationNames){var n=e.phasedRegistrationNames;for(var r in n)if(n.hasOwnProperty(r)){var i=s.registrationNameModules[n[r]];if(i)return i}}return null},_resetEventPlugins:function(){u=null;for(var t in c)c.hasOwnProperty(t)&&delete c[t];s.plugins.length=0;var e=s.eventNameDispatchConfigs;for(var n in e)e.hasOwnProperty(n)&&delete e[n];var r=s.registrationNameModules;for(var i in r)r.hasOwnProperty(i)&&delete r[i]}};t.exports=s},function(t,e,n){"use strict";function r(t){var e=/[=:]/g,n={"=":"=0",":":"=2"},r=(""+t).replace(e,function(t){return n[t]});return"$"+r}function i(t){var e=/(=0|=2)/g,n={"=0":"=","=2":":"},r="."===t[0]&&"$"===t[1]?t.substring(2):t.substring(1);return(""+r).replace(e,function(t){return n[t]})}var o={escape:r,unescape:i};t.exports=o},function(t,e,n){"use strict";function r(t){null!=t.checkedLink&&null!=t.valueLink?u("87"):void 0}function i(t){r(t),null!=t.value||null!=t.onChange?u("88"):void 0}function o(t){r(t),null!=t.checked||null!=t.onChange?u("89"):void 0}function a(t){if(t){var e=t.getName();if(e)return" Check the render method of `"+e+"`."}return""}var u=n(2),c=n(26),s=n(366),l=(n(0),n(1),{button:!0,checkbox:!0,image:!0,hidden:!0,radio:!0,reset:!0,submit:!0}),f={value:function(t,e,n){return!t[e]||l[t.type]||t.onChange||t.readOnly||t.disabled?null:new Error("You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set either `onChange` or `readOnly`.")},checked:function(t,e,n){return!t[e]||t.onChange||t.readOnly||t.disabled?null:new Error("You provided a `checked` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultChecked`. Otherwise, set either `onChange` or `readOnly`.")},onChange:c.PropTypes.func},p={},h={checkPropTypes:function(t,e,n){for(var r in f){if(f.hasOwnProperty(r))var i=f[r](e,r,t,"prop",null,s);if(i instanceof Error&&!(i.message in p)){p[i.message]=!0;a(n)}}},getValue:function(t){return t.valueLink?(i(t),t.valueLink.value):t.value},getChecked:function(t){return t.checkedLink?(o(t),t.checkedLink.value):t.checked},executeOnChange:function(t,e){return t.valueLink?(i(t),t.valueLink.requestChange(e.target.value)):t.checkedLink?(o(t),t.checkedLink.requestChange(e.target.checked)):t.onChange?t.onChange.call(void 0,e):void 0}};t.exports=h},function(t,e,n){"use strict";var r=n(2),i=(n(0),!1),o={replaceNodeWithMarkup:null,processChildrenUpdates:null,injection:{injectEnvironment:function(t){i?r("104"):void 0,o.replaceNodeWithMarkup=t.replaceNodeWithMarkup,o.processChildrenUpdates=t.processChildrenUpdates,i=!0}}};t.exports=o},function(t,e,n){"use strict";function r(t,e,n){try{e(n)}catch(t){null===i&&(i=t)}}var i=null,o={invokeGuardedCallback:r,invokeGuardedCallbackWithCatch:r,rethrowCaughtError:function(){if(i){var t=i;throw i=null,t}}};t.exports=o},function(t,e,n){"use strict";function r(t){c.enqueueUpdate(t)}function i(t){var e=typeof t;if("object"!==e)return e;var n=t.constructor&&t.constructor.name||e,r=Object.keys(t);return r.length>0&&r.length<20?n+" (keys: "+r.join(", ")+")":n}function o(t,e){var n=u.get(t);if(!n){return null}return n}var a=n(2),u=(n(15),n(40)),c=(n(9),n(11)),s=(n(0),n(1),{isMounted:function(t){var e=u.get(t);return!!e&&!!e._renderedComponent},enqueueCallback:function(t,e,n){s.validateCallback(e,n);var i=o(t);return i?(i._pendingCallbacks?i._pendingCallbacks.push(e):i._pendingCallbacks=[e],void r(i)):null},enqueueCallbackInternal:function(t,e){t._pendingCallbacks?t._pendingCallbacks.push(e):t._pendingCallbacks=[e],r(t)},enqueueForceUpdate:function(t){var e=o(t,"forceUpdate");e&&(e._pendingForceUpdate=!0,r(e))},enqueueReplaceState:function(t,e){var n=o(t,"replaceState");n&&(n._pendingStateQueue=[e],n._pendingReplaceState=!0,r(n))},enqueueSetState:function(t,e){var n=o(t,"setState");if(n){var i=n._pendingStateQueue||(n._pendingStateQueue=[]);i.push(e),r(n)}},enqueueElementInternal:function(t,e,n){t._pendingElement=e,t._context=n,r(t)},validateCallback:function(t,e){t&&"function"!=typeof t?a("122",e,i(t)):void 0}});t.exports=s},function(t,e,n){"use strict";var r={currentScrollLeft:0,currentScrollTop:0,refreshScrollValues:function(t){r.currentScrollLeft=t.x,r.currentScrollTop=t.y}};t.exports=r},function(t,e,n){"use strict";var r=function(t){return"undefined"!=typeof MSApp&&MSApp.execUnsafeLocalFunction?function(e,n,r,i){MSApp.execUnsafeLocalFunction(function(){return t(e,n,r,i)})}:t};t.exports=r},function(t,e,n){"use strict";function r(t){var e,n=t.keyCode;return"charCode"in t?(e=t.charCode,0===e&&13===n&&(e=13)):e=n,e>=32||13===e?e:0}t.exports=r},function(t,e,n){"use strict";function r(t){var e=this,n=e.nativeEvent;if(n.getModifierState)return n.getModifierState(t);var r=o[t];return!!r&&!!n[r]}function i(t){return r}var o={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};t.exports=i},function(t,e,n){"use strict";function r(t){var e=t.target||t.srcElement||window;return e.correspondingUseElement&&(e=e.correspondingUseElement),3===e.nodeType?e.parentNode:e}t.exports=r},function(t,e,n){"use strict";/**
 * Checks if an event is supported in the current execution environment.
 *
 * NOTE: This will not work correctly for non-generic events such as `change`,
 * `reset`, `load`, `error`, and `select`.
 *
 * Borrows from Modernizr.
 *
 * @param {string} eventNameSuffix Event name, e.g. "click".
 * @param {?boolean} capture Check if the capture phase is supported.
 * @return {boolean} True if the event is supported.
 * @internal
 * @license Modernizr 3.0.0pre (Custom Build) | MIT
 */
function r(t,e){if(!o.canUseDOM||e&&!("addEventListener"in document))return!1;var n="on"+t,r=n in document;if(!r){var a=document.createElement("div");a.setAttribute(n,"return;"),r="function"==typeof a[n]}return!r&&i&&"wheel"===t&&(r=document.implementation.hasFeature("Events.wheel","3.0")),r}var i,o=n(6);o.canUseDOM&&(i=document.implementation&&document.implementation.hasFeature&&document.implementation.hasFeature("","")!==!0),t.exports=r},function(t,e,n){"use strict";function r(t,e){var n=null===t||t===!1,r=null===e||e===!1;if(n||r)return n===r;var i=typeof t,o=typeof e;return"string"===i||"number"===i?"string"===o||"number"===o:"object"===o&&t.type===e.type&&t.key===e.key}t.exports=r},function(t,e,n){"use strict";var r=(n(3),n(8)),i=(n(1),r);t.exports=i},function(t,e,n){"use strict";function r(t,e,n){this.props=t,this.context=e,this.refs=a,this.updater=n||o}var i=n(28),o=n(98),a=(n(176),n(38));n(0),n(1);r.prototype.isReactComponent={},r.prototype.setState=function(t,e){"object"!=typeof t&&"function"!=typeof t&&null!=t?i("85"):void 0,this.updater.enqueueSetState(this,t),e&&this.updater.enqueueCallback(this,e,"setState")},r.prototype.forceUpdate=function(t){this.updater.enqueueForceUpdate(this),t&&this.updater.enqueueCallback(this,t,"forceUpdate")};t.exports=r},function(t,e,n){"use strict";function r(t,e){}var i=(n(1),{isMounted:function(t){return!1},enqueueCallback:function(t,e){},enqueueForceUpdate:function(t){r(t,"forceUpdate")},enqueueReplaceState:function(t,e){r(t,"replaceState")},enqueueSetState:function(t,e){r(t,"setState")}});t.exports=i},function(t,e){var n;n=function(){return this}();try{n=n||Function("return this")()||(0,eval)("this")}catch(t){"object"==typeof window&&(n=window)}t.exports=n},function(t,e){t.exports=function(t){return t.webpackPolyfill||(t.deprecate=function(){},t.paths=[],t.children||(t.children=[]),Object.defineProperty(t,"loaded",{enumerable:!0,get:function(){return t.l}}),Object.defineProperty(t,"id",{enumerable:!0,get:function(){return t.i}}),t.webpackPolyfill=1),t}},function(t,e,n){"use strict";n.d(e,"b",function(){return i}),n.d(e,"a",function(){return o});var r=Array.prototype,i=r.slice,o=r.map},function(t,e,n){"use strict";var r=n(18),i=n(103),o=n.i(i.a)(r.a),a=o.right;o.left;e.a=a},function(t,e,n){"use strict";function r(t){return function(e,r){return n.i(i.a)(t(e),r)}}var i=n(18);e.a=function(t){return 1===t.length&&(t=r(t)),{left:function(e,n,r,i){for(null==r&&(r=0),null==i&&(i=e.length);r<i;){var o=r+i>>>1;t(e[o],n)<0?r=o+1:i=o}return r},right:function(e,n,r,i){for(null==r&&(r=0),null==i&&(i=e.length);r<i;){var o=r+i>>>1;t(e[o],n)>0?i=o:r=o+1}return r}}}},function(t,e,n){"use strict";var r=n(111);e.a=function(t,e){var i=n.i(r.a)(t,e);return i?Math.sqrt(i):i}},function(t,e,n){"use strict";e.a=function(t,e){var n,r,i,o=-1,a=t.length;if(null==e){for(;++o<a;)if(null!=(r=t[o])&&r>=r){n=i=r;break}for(;++o<a;)null!=(r=t[o])&&(n>r&&(n=r),i<r&&(i=r))}else{for(;++o<a;)if(null!=(r=e(t[o],o,t))&&r>=r){n=i=r;break}for(;++o<a;)null!=(r=e(t[o],o,t))&&(n>r&&(n=r),i<r&&(i=r))}return[n,i]}},function(t,e,n){"use strict";e.a=function(t,e){var n,r,i=-1,o=t.length;if(null==e){for(;++i<o;)if(null!=(r=t[i])&&r>=r){n=r;break}for(;++i<o;)null!=(r=t[i])&&n>r&&(n=r)}else{for(;++i<o;)if(null!=(r=e(t[i],i,t))&&r>=r){n=r;break}for(;++i<o;)null!=(r=e(t[i],i,t))&&n>r&&(n=r)}return n}},function(t,e,n){"use strict";e.a=function(t,e,n){t=+t,e=+e,n=(i=arguments.length)<2?(e=t,t=0,1):i<3?1:+n;for(var r=-1,i=0|Math.max(0,Math.ceil((e-t)/n)),o=new Array(i);++r<i;)o[r]=t+r*n;return o}},function(t,e,n){"use strict";e.a=function(t){return Math.ceil(Math.log(t.length)/Math.LN2)+1}},function(t,e,n){"use strict";function r(t,e,n){var r=Math.abs(e-t)/Math.max(0,n),i=Math.pow(10,Math.floor(Math.log(r)/Math.LN10)),c=r/i;return c>=o?i*=10:c>=a?i*=5:c>=u&&(i*=2),e<t?-i:i}var i=n(107);e.b=r;var o=Math.sqrt(50),a=Math.sqrt(10),u=Math.sqrt(2);e.a=function(t,e,o){var a=r(t,e,o);return n.i(i.a)(Math.ceil(t/a)*a,Math.floor(e/a)*a+a/2,a)}},function(t,e,n){"use strict";function r(t){return t.length}var i=n(106);e.a=function(t){if(!(u=t.length))return[];for(var e=-1,o=n.i(i.a)(t,r),a=new Array(o);++e<o;)for(var u,c=-1,s=a[e]=new Array(u);++c<u;)s[c]=t[c][e];return a}},function(t,e,n){"use strict";var r=n(29);e.a=function(t,e){var i,o,a=t.length,u=0,c=0,s=-1,l=0;if(null==e)for(;++s<a;)isNaN(i=n.i(r.a)(t[s]))||(o=i-u,u+=o/++l,c+=o*(i-u));else for(;++s<a;)isNaN(i=n.i(r.a)(e(t[s],s,t)))||(o=i-u,u+=o/++l,c+=o*(i-u));if(l>1)return c/(l-1)}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(201);n.d(e,"axisTop",function(){return r.a}),n.d(e,"axisRight",function(){return r.b}),n.d(e,"axisBottom",function(){return r.c}),n.d(e,"axisLeft",function(){return r.d})},function(t,e,n){"use strict";n.d(e,"b",function(){return r}),n.d(e,"a",function(){return i});var r=Math.PI/180,i=180/Math.PI},function(t,e,n){"use strict";var r=n(61);n.d(e,"b",function(){return i});var i;e.a=function(t,e){var o=n.i(r.a)(t,e);if(!o)return t+"";var a=o[0],u=o[1],c=u-(i=3*Math.max(-8,Math.min(8,Math.floor(u/3))))+1,s=a.length;return c===s?a:c>s?a+new Array(c-s+1).join("0"):c>0?a.slice(0,c)+"."+a.slice(c):"0."+new Array(1-c).join("0")+n.i(r.a)(t,Math.max(0,e+c-1))[0]}},function(t,e,n){"use strict";function r(t){if(!(e=o.exec(t)))throw new Error("invalid format: "+t);var e,n=e[1]||" ",r=e[2]||">",a=e[3]||"-",u=e[4]||"",c=!!e[5],s=e[6]&&+e[6],l=!!e[7],f=e[8]&&+e[8].slice(1),p=e[9]||"";"n"===p?(l=!0,p="g"):i.a[p]||(p=""),(c||"0"===n&&"="===r)&&(c=!0,n="0",r="="),this.fill=n,this.align=r,this.sign=a,this.symbol=u,this.zero=c,this.width=s,this.comma=l,this.precision=f,this.type=p}var i=n(116),o=/^(?:(.)?([<>=^]))?([+\-\( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?([a-z%])?$/i;e.a=function(t){return new r(t)},r.prototype.toString=function(){return this.fill+this.align+this.sign+this.symbol+(this.zero?"0":"")+(null==this.width?"":Math.max(1,0|this.width))+(this.comma?",":"")+(null==this.precision?"":"."+Math.max(0,0|this.precision))+this.type}},function(t,e,n){"use strict";var r=n(212),i=n(114),o=n(214);e.a={"":r.a,"%":function(t,e){return(100*t).toFixed(e)},b:function(t){return Math.round(t).toString(2)},c:function(t){return t+""},d:function(t){return Math.round(t).toString(10)},e:function(t,e){return t.toExponential(e)},f:function(t,e){return t.toFixed(e)},g:function(t,e){return t.toPrecision(e)},o:function(t){return Math.round(t).toString(8)},p:function(t,e){return n.i(o.a)(100*t,e)},r:o.a,s:i.a,X:function(t){return Math.round(t).toString(16).toUpperCase()},x:function(t){return Math.round(t).toString(16)}}},function(t,e,n){"use strict";function r(t){return t}var i=n(42),o=n(213),a=n(115),u=n(116),c=n(114),s=["y","z","a","f","p","n","µ","m","","k","M","G","T","P","E","Z","Y"];e.a=function(t){function e(t){function e(t){var e,n,a,u=_,l=b;if("c"===y)l=x(t)+l,t="";else{t=+t;var p=(t<0||1/t<0)&&(t*=-1,!0);if(t=x(t,m),p)for(e=-1,n=t.length,p=!1;++e<n;)if(a=t.charCodeAt(e),48<a&&a<58||"x"===y&&96<a&&a<103||"X"===y&&64<a&&a<71){p=!0;break}if(u=(p?"("===o?o:"-":"-"===o||"("===o?"":o)+u,l=l+("s"===y?s[8+c.b/3]:"")+(p&&"("===o?")":""),w)for(e=-1,n=t.length;++e<n;)if(a=t.charCodeAt(e),48>a||a>57){l=(46===a?h+t.slice(e+1):t.slice(e))+l,t=t.slice(0,e);break}}g&&!d&&(t=f(t,1/0));var C=u.length+t.length+l.length,M=C<v?new Array(v-C+1).join(r):"";switch(g&&d&&(t=f(M+t,M.length?v-l.length:1/0),M=""),i){case"<":return u+t+l+M;case"=":return u+M+t+l;case"^":return M.slice(0,C=M.length>>1)+u+t+l+M.slice(C)}return M+u+t+l}t=n.i(a.a)(t);var r=t.fill,i=t.align,o=t.sign,l=t.symbol,d=t.zero,v=t.width,g=t.comma,m=t.precision,y=t.type,_="$"===l?p[0]:"#"===l&&/[boxX]/.test(y)?"0"+y.toLowerCase():"",b="$"===l?p[1]:/[%p]/.test(y)?"%":"",x=u.a[y],w=!y||/[defgprs%]/.test(y);return m=null==m?y?6:12:/[gprs]/.test(y)?Math.max(1,Math.min(21,m)):Math.max(0,Math.min(20,m)),e.toString=function(){return t+""},e}function l(t,r){var o=e((t=n.i(a.a)(t),t.type="f",t)),u=3*Math.max(-8,Math.min(8,Math.floor(n.i(i.a)(r)/3))),c=Math.pow(10,-u),l=s[8+u/3];return function(t){return o(c*t)+l}}var f=t.grouping&&t.thousands?n.i(o.a)(t.grouping,t.thousands):r,p=t.currency,h=t.decimal;return{format:e,formatPrefix:l}}},function(t,e,n){"use strict";var r=n(63);e.a=function(t,e){var i,o=e?e.length:0,a=t?Math.min(o,t.length):0,u=new Array(o),c=new Array(o);for(i=0;i<a;++i)u[i]=n.i(r.a)(t[i],e[i]);for(;i<o;++i)c[i]=e[i];return function(t){for(i=0;i<a;++i)c[i]=u[i](t);return c}}},function(t,e,n){"use strict";var r=n(62);e.a=function(t){var e=t.length;return function(i){var o=Math.floor(((i%=1)<0?++i:i)*e),a=t[(o+e-1)%e],u=t[o%e],c=t[(o+1)%e],s=t[(o+2)%e];return n.i(r.b)((i-o/e)*e,a,u,c,s)}}},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";e.a=function(t,e){var n=new Date;return t=+t,e-=t,function(r){return n.setTime(t+e*r),n}}},function(t,e,n){"use strict";var r=n(63);e.a=function(t,e){var i,o={},a={};null!==t&&"object"==typeof t||(t={}),null!==e&&"object"==typeof e||(e={});for(i in e)i in t?o[i]=n.i(r.a)(t[i],e[i]):a[i]=e[i];return function(t){for(i in o)a[i]=o[i](t);return a}}},function(t,e,n){"use strict";function r(t){return function(e){var r,o,a=e.length,u=new Array(a),c=new Array(a),s=new Array(a);for(r=0;r<a;++r)o=n.i(i.rgb)(e[r]),u[r]=o.r||0,c[r]=o.g||0,s[r]=o.b||0;return u=t(u),c=t(c),s=t(s),o.opacity=1,function(t){return o.r=u(t),o.g=c(t),o.b=s(t),o+""}}}var i=n(10),o=n(62),a=n(119),u=n(32);e.a=function t(e){function r(t,e){var r=o((t=n.i(i.rgb)(t)).r,(e=n.i(i.rgb)(e)).r),a=o(t.g,e.g),c=o(t.b,e.b),s=n.i(u.a)(t.opacity,e.opacity);return function(e){return t.r=r(e),t.g=a(e),t.b=c(e),t.opacity=s(e),t+""}}var o=n.i(u.c)(e);return r.gamma=t,r}(1);r(o.a),r(a.a)},function(t,e,n){"use strict";function r(t){return function(){return t}}function i(t){return function(e){return t(e)+""}}var o=n(43),a=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,u=new RegExp(a.source,"g");e.a=function(t,e){var c,s,l,f=a.lastIndex=u.lastIndex=0,p=-1,h=[],d=[];for(t+="",e+="";(c=a.exec(t))&&(s=u.exec(e));)(l=s.index)>f&&(l=e.slice(f,l),h[p]?h[p]+=l:h[++p]=l),(c=c[0])===(s=s[0])?h[p]?h[p]+=s:h[++p]=s:(h[++p]=null,d.push({i:p,x:n.i(o.a)(c,s)})),f=u.lastIndex;return f<e.length&&(l=e.slice(f),h[p]?h[p]+=l:h[++p]=l),h.length<2?d[0]?i(d[0].x):r(e):(e=d.length,function(t){for(var n,r=0;r<e;++r)h[(n=d[r]).i]=n.x(t);return h.join("")})}},function(t,e,n){"use strict";e.a=function(t,e){t=t.slice();var n,r=0,i=t.length-1,o=t[r],a=t[i];return a<o&&(n=r,r=i,i=n,n=o,o=a,a=n),t[r]=e.floor(o),t[i]=e.ceil(a),t}},function(t,e,n){"use strict";e.a=function(t){return+t}},function(t,e,n){"use strict";function r(t){function e(e){var n=e+"",r=u.get(n);if(!r){if(s!==a)return s;u.set(n,r=c.push(e))}return t[(r-1)%t.length]}var u=n.i(i.a)(),c=[],s=a;return t=null==t?[]:o.b.call(t),e.domain=function(t){if(!arguments.length)return c.slice();c=[],u=n.i(i.a)();for(var r,o,a=-1,s=t.length;++a<s;)u.has(o=(r=t[a])+"")||u.set(o,c.push(r));return e},e.range=function(n){return arguments.length?(t=o.b.call(n),e):t.slice()},e.unknown=function(t){return arguments.length?(s=t,e):s},e.copy=function(){return r().domain(c).range(t).unknown(s)},e}var i=n(203),o=n(16);n.d(e,"b",function(){return a}),e.a=r;var a={name:"implicit"}},function(t,e,n){"use strict";function r(t){return new Date(t)}function i(t){return t instanceof Date?+t:+new Date(+t)}function o(t,e,c,s,b,x,w,C,M){function k(n){return(w(n)<n?N:x(n)<n?A:b(n)<n?O:s(n)<n?I:e(n)<n?c(n)<n?D:R:t(n)<n?L:U)(n)}function E(e,r,i,o){if(null==e&&(e=10),"number"==typeof e){var u=Math.abs(i-r)/e,c=n.i(a.d)(function(t){return t[2]}).right(F,u);c===F.length?(o=n.i(a.b)(r/_,i/_,e),e=t):c?(c=F[u/F[c-1][2]<F[c][2]/u?c-1:c],o=c[1],e=c[0]):(o=n.i(a.b)(r,i,e),e=C)}return null==o?e:e.every(o)}var T=n.i(f.a)(f.b,u.a),S=T.invert,P=T.domain,N=M(".%L"),A=M(":%S"),O=M("%I:%M"),I=M("%I %p"),D=M("%a %d"),R=M("%b %d"),L=M("%B"),U=M("%Y"),F=[[w,1,h],[w,5,5*h],[w,15,15*h],[w,30,30*h],[x,1,d],[x,5,5*d],[x,15,15*d],[x,30,30*d],[b,1,v],[b,3,3*v],[b,6,6*v],[b,12,12*v],[s,1,g],[s,2,2*g],[c,1,m],[e,1,y],[e,3,3*y],[t,1,_]];return T.invert=function(t){return new Date(S(t))},T.domain=function(t){return arguments.length?P(l.a.call(t,i)):P().map(r)},T.ticks=function(t,e){var n,r=P(),i=r[0],o=r[r.length-1],a=o<i;return a&&(n=i,i=o,o=n),n=E(t,i,o,e),n=n?n.range(i,o+1):[],a?n.reverse():n},T.tickFormat=function(t,e){return null==e?k:M(e)},T.nice=function(t,e){var r=P();return(t=E(t,r[0],r[r.length-1],e))?P(n.i(p.a)(r,t)):T},T.copy=function(){return n.i(f.c)(T,o(t,e,c,s,b,x,w,C,M))},T}var a=n(12),u=n(31),c=n(79),s=n(77),l=n(16),f=n(45),p=n(125);e.b=o;var h=1e3,d=60*h,v=60*d,g=24*v,m=7*g,y=30*g,_=365*g;e.a=function(){return o(c.b,c.o,c.p,c.a,c.q,c.r,c.s,c.t,s.timeFormat).domain([new Date(2e3,0,1),new Date(2e3,0,2)])}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(66);n.d(e,"creator",function(){return r.a});var i=n(247);n.d(e,"local",function(){return i.a});var o=n(130);n.d(e,"matcher",function(){return o.a});var a=n(248);n.d(e,"mouse",function(){return a.a});var u=n(67);n.d(e,"namespace",function(){return u.a});var c=n(68);n.d(e,"namespaces",function(){return c.a});var s=n(249);n.d(e,"select",function(){return s.a});var l=n(250);n.d(e,"selectAll",function(){return l.a});var f=n(7);n.d(e,"selection",function(){return f.a});var p=n(71);n.d(e,"selector",function(){return p.a});var h=n(133);n.d(e,"selectorAll",function(){return h.a});var d=n(278);n.d(e,"touch",function(){return d.a});var v=n(279);n.d(e,"touches",function(){return v.a});var g=n(73);n.d(e,"window",function(){return g.a});var m=n(70);n.d(e,"event",function(){return m.a}),n.d(e,"customEvent",function(){return m.b})},function(t,e,n){"use strict";var r=function(t){return function(){return this.matches(t)}};if("undefined"!=typeof document){var i=document.documentElement;if(!i.matches){var o=i.webkitMatchesSelector||i.msMatchesSelector||i.mozMatchesSelector||i.oMatchesSelector;r=function(t){return function(){return o.call(this,t)}}}}e.a=r},function(t,e,n){"use strict";function r(t,e){this.ownerDocument=t.ownerDocument,this.namespaceURI=t.namespaceURI,this._next=null,this._parent=t,this.__data__=e}var i=n(132),o=n(7);e.b=r,e.a=function(){return new o.b(this._enter||this._groups.map(i.a),this._parents)},r.prototype={constructor:r,appendChild:function(t){return this._parent.insertBefore(t,this._next)},insertBefore:function(t,e){return this._parent.insertBefore(t,e)},querySelector:function(t){return this._parent.querySelector(t)},querySelectorAll:function(t){return this._parent.querySelectorAll(t)}}},function(t,e,n){"use strict";e.a=function(t){return new Array(t.length)}},function(t,e,n){"use strict";function r(){return[]}e.a=function(t){return null==t?r:function(){return this.querySelectorAll(t)}}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(280);n.d(e,"arc",function(){return r.a});var i=n(135);n.d(e,"area",function(){return i.a});var o=n(75);n.d(e,"line",function(){return o.a});var a=n(299);n.d(e,"pie",function(){return a.a});var u=n(300);n.d(e,"radialArea",function(){return u.a});var c=n(140);n.d(e,"radialLine",function(){return c.a});var s=n(302);n.d(e,"symbol",function(){return s.a}),n.d(e,"symbols",function(){return s.b});var l=n(141);n.d(e,"symbolCircle",function(){return l.a});var f=n(142);n.d(e,"symbolCross",function(){return f.a});var p=n(143);n.d(e,"symbolDiamond",function(){return p.a});var h=n(144);n.d(e,"symbolSquare",function(){return h.a});var d=n(145);n.d(e,"symbolStar",function(){return d.a});var v=n(146);n.d(e,"symbolTriangle",function(){return v.a});var g=n(147);n.d(e,"symbolWye",function(){return g.a});var m=n(282);n.d(e,"curveBasisClosed",function(){return m.a});var y=n(283);n.d(e,"curveBasisOpen",function(){return y.a});var _=n(46);n.d(e,"curveBasis",function(){return _.a});var b=n(284);n.d(e,"curveBundle",function(){return b.a});var x=n(136);n.d(e,"curveCardinalClosed",function(){return x.a});var w=n(137);n.d(e,"curveCardinalOpen",function(){return w.a});var C=n(47);n.d(e,"curveCardinal",function(){return C.a});var M=n(285);n.d(e,"curveCatmullRomClosed",function(){return M.a});var k=n(286);n.d(e,"curveCatmullRomOpen",function(){return k.a});var E=n(74);n.d(e,"curveCatmullRom",function(){return E.a});var T=n(287);n.d(e,"curveLinearClosed",function(){return T.a});var S=n(48);n.d(e,"curveLinear",function(){return S.a});var P=n(288);n.d(e,"curveMonotoneX",function(){return P.a}),n.d(e,"curveMonotoneY",function(){return P.b});var N=n(289);n.d(e,"curveNatural",function(){return N.a});var A=n(290);n.d(e,"curveStep",function(){return A.a}),n.d(e,"curveStepAfter",function(){return A.b}),n.d(e,"curveStepBefore",function(){return A.c});var O=n(301);n.d(e,"stack",function(){return O.a});var I=n(293);n.d(e,"stackOffsetExpand",function(){return I.a});var D=n(36);n.d(e,"stackOffsetNone",function(){return D.a});var R=n(294);n.d(e,"stackOffsetSilhouette",function(){return R.a});var L=n(295);n.d(e,"stackOffsetWiggle",function(){return L.a});var U=n(76);n.d(e,"stackOrderAscending",function(){return U.a});var F=n(296);n.d(e,"stackOrderDescending",function(){return F.a});var j=n(297);n.d(e,"stackOrderInsideOut",function(){return j.a});var B=n(37);n.d(e,"stackOrderNone",function(){return B.a});var W=n(298);n.d(e,"stackOrderReverse",function(){return W.a})},function(t,e,n){"use strict";var r=n(44),i=n(19),o=n(48),a=n(75),u=n(139);e.a=function(){function t(t){var e,i,o,a,u,g=t.length,m=!1,y=new Array(g),_=new Array(g);for(null==h&&(v=d(u=n.i(r.a)())),e=0;e<=g;++e){if(!(e<g&&p(a=t[e],e,t))===m)if(m=!m)i=e,v.areaStart(),v.lineStart();else{for(v.lineEnd(),v.lineStart(),o=e-1;o>=i;--o)v.point(y[o],_[o]);v.lineEnd(),v.areaEnd()}m&&(y[e]=+c(a,e,t),_[e]=+l(a,e,t),v.point(s?+s(a,e,t):y[e],f?+f(a,e,t):_[e]))}if(u)return v=null,u+""||null}function e(){return n.i(a.a)().defined(p).curve(d).context(h)}var c=u.a,s=null,l=n.i(i.a)(0),f=u.b,p=n.i(i.a)(!0),h=null,d=o.a,v=null;return t.x=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(+e),s=null,t):c},t.x0=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(+e),t):c},t.x1=function(e){return arguments.length?(s=null==e?null:"function"==typeof e?e:n.i(i.a)(+e),t):s},t.y=function(e){return arguments.length?(l="function"==typeof e?e:n.i(i.a)(+e),f=null,t):l},t.y0=function(e){return arguments.length?(l="function"==typeof e?e:n.i(i.a)(+e),t):l},t.y1=function(e){return arguments.length?(f=null==e?null:"function"==typeof e?e:n.i(i.a)(+e),t):f},t.lineX0=t.lineY0=function(){return e().x(c).y(l)},t.lineY1=function(){return e().x(c).y(f)},t.lineX1=function(){return e().x(s).y(l)},t.defined=function(e){return arguments.length?(p="function"==typeof e?e:n.i(i.a)(!!e),t):p},t.curve=function(e){return arguments.length?(d=e,null!=h&&(v=d(h)),t):d},t.context=function(e){return arguments.length?(null==e?h=v=null:v=d(h=e),t):h},t}},function(t,e,n){"use strict";function r(t,e){this._context=t,this._k=(1-e)/6}var i=n(49),o=n(47);e.b=r,r.prototype={areaStart:i.a,areaEnd:i.a,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN,this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x3,this._y3),this._context.closePath();break;case 2:this._context.lineTo(this._x3,this._y3),this._context.closePath();break;case 3:this.point(this._x3,this._y3),this.point(this._x4,this._y4),this.point(this._x5,this._y5)}},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._x3=t,this._y3=e;break;case 1:this._point=2,this._context.moveTo(this._x4=t,this._y4=e);break;case 2:this._point=3,this._x5=t,this._y5=e;break;default:n.i(o.c)(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new r(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t,e){this._context=t,this._k=(1-e)/6}var i=n(47);e.b=r,r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3,this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;default:n.i(i.c)(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new r(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t){this._curve=t}function i(t){function e(e){return new r(t(e))}return e._curve=t,e}var o=n(48);n.d(e,"b",function(){return a}),e.a=i;var a=i(o.a);r.prototype={areaStart:function(){this._curve.areaStart()},areaEnd:function(){this._curve.areaEnd()},lineStart:function(){this._curve.lineStart()},lineEnd:function(){this._curve.lineEnd()},point:function(t,e){this._curve.point(e*Math.sin(t),e*-Math.cos(t))}}},function(t,e,n){"use strict";function r(t){return t[0]}function i(t){return t[1]}e.a=r,e.b=i},function(t,e,n){"use strict";function r(t){var e=t.curve;return t.angle=t.x,delete t.x,t.radius=t.y,delete t.y,t.curve=function(t){return arguments.length?e(n.i(i.a)(t)):e()._curve},t}var i=n(138),o=n(75);e.b=r,e.a=function(){return r(n.i(o.a)().curve(i.b))}},function(t,e,n){"use strict";var r=n(35);e.a={draw:function(t,e){var n=Math.sqrt(e/r.b);t.moveTo(n,0),t.arc(0,0,n,0,r.c)}}},function(t,e,n){"use strict";e.a={draw:function(t,e){var n=Math.sqrt(e/5)/2;t.moveTo(-3*n,-n),t.lineTo(-n,-n),t.lineTo(-n,-3*n),t.lineTo(n,-3*n),t.lineTo(n,-n),t.lineTo(3*n,-n),t.lineTo(3*n,n),t.lineTo(n,n),t.lineTo(n,3*n),t.lineTo(-n,3*n),t.lineTo(-n,n),t.lineTo(-3*n,n),t.closePath()}}},function(t,e,n){"use strict";var r=Math.sqrt(1/3),i=2*r;e.a={draw:function(t,e){var n=Math.sqrt(e/i),o=n*r;t.moveTo(0,-n),t.lineTo(o,0),t.lineTo(0,n),t.lineTo(-o,0),t.closePath()}}},function(t,e,n){"use strict";e.a={draw:function(t,e){var n=Math.sqrt(e),r=-n/2;t.rect(r,r,n,n)}}},function(t,e,n){"use strict";var r=n(35),i=.8908130915292852,o=Math.sin(r.b/10)/Math.sin(7*r.b/10),a=Math.sin(r.c/10)*o,u=-Math.cos(r.c/10)*o;e.a={draw:function(t,e){var n=Math.sqrt(e*i),o=a*n,c=u*n;t.moveTo(0,-n),t.lineTo(o,c);for(var s=1;s<5;++s){var l=r.c*s/5,f=Math.cos(l),p=Math.sin(l);t.lineTo(p*n,-f*n),t.lineTo(f*o-p*c,p*o+f*c)}t.closePath()}}},function(t,e,n){"use strict";var r=Math.sqrt(3);e.a={draw:function(t,e){var n=-Math.sqrt(e/(3*r));t.moveTo(0,2*n),t.lineTo(-r*n,-n),t.lineTo(r*n,-n),t.closePath()}}},function(t,e,n){"use strict";var r=-.5,i=Math.sqrt(3)/2,o=1/Math.sqrt(12),a=3*(o/2+1);e.a={draw:function(t,e){var n=Math.sqrt(e/a),u=n/2,c=n*o,s=u,l=n*o+n,f=-s,p=l;t.moveTo(u,c),t.lineTo(s,l),t.lineTo(f,p),t.lineTo(r*u-i*c,i*u+r*c),t.lineTo(r*s-i*l,i*s+r*l),t.lineTo(r*f-i*p,i*f+r*p),t.lineTo(r*u+i*c,r*c-i*u),t.lineTo(r*s+i*l,r*l-i*s),t.lineTo(r*f+i*p,r*p-i*f),t.closePath()}}},function(t,e,n){"use strict";function r(t){return t.toISOString()}var i=n(78);n.d(e,"b",function(){return o});var o="%Y-%m-%dT%H:%M:%S.%LZ",a=Date.prototype.toISOString?r:n.i(i.d)(o);e.a=a},function(t,e,n){"use strict";function r(t){if(0<=t.y&&t.y<100){var e=new Date(-1,t.m,t.d,t.H,t.M,t.S,t.L);return e.setFullYear(t.y),e}return new Date(t.y,t.m,t.d,t.H,t.M,t.S,t.L)}function i(t){if(0<=t.y&&t.y<100){var e=new Date(Date.UTC(-1,t.m,t.d,t.H,t.M,t.S,t.L));return e.setUTCFullYear(t.y),e}return new Date(Date.UTC(t.y,t.m,t.d,t.H,t.M,t.S,t.L))}function o(t){return{y:t,m:0,d:1,H:0,M:0,S:0,L:0}}function a(t){function e(t,e){return function(n){var r,i,o,a=[],u=-1,c=0,s=t.length;for(n instanceof Date||(n=new Date(+n));++u<s;)37===t.charCodeAt(u)&&(a.push(t.slice(c,u)),null!=(i=et[r=t.charAt(++u)])?r=t.charAt(++u):i="e"===r?" ":"0",(o=e[r])&&(r=o(n,i)),a.push(r),c=u+1);return a.push(t.slice(c,u)),a.join("")}}function n(t,e){return function(n){var r=o(1900),u=a(r,t,n+="",0);if(u!=n.length)return null;if("p"in r&&(r.H=r.H%12+12*r.p),"W"in r||"U"in r){"w"in r||(r.w="W"in r?1:0);var c="Z"in r?i(o(r.y)).getUTCDay():e(o(r.y)).getDay();r.m=0,r.d="W"in r?(r.w+6)%7+7*r.W-(c+5)%7:r.w+7*r.U-(c+6)%7}return"Z"in r?(r.H+=r.Z/100|0,r.M+=r.Z%100,i(r)):e(r)}}function a(t,e,n,r){for(var i,o,a=0,u=e.length,c=n.length;a<u;){if(r>=c)return-1;if(i=e.charCodeAt(a++),37===i){if(i=e.charAt(a++),o=Ut[i in et?e.charAt(a++):i],!o||(r=o(t,n,r))<0)return-1}else if(i!=n.charCodeAt(r++))return-1}return r}function u(t,e,n){var r=kt.exec(e.slice(n));return r?(t.p=Et[r[0].toLowerCase()],n+r[0].length):-1}function c(t,e,n){var r=Pt.exec(e.slice(n));return r?(t.w=Nt[r[0].toLowerCase()],n+r[0].length):-1}function tt(t,e,n){var r=Tt.exec(e.slice(n));return r?(t.w=St[r[0].toLowerCase()],n+r[0].length):-1}function nt(t,e,n){var r=It.exec(e.slice(n));return r?(t.m=Dt[r[0].toLowerCase()],n+r[0].length):-1}function rt(t,e,n){var r=At.exec(e.slice(n));return r?(t.m=Ot[r[0].toLowerCase()],n+r[0].length):-1}function it(t,e,n){return a(t,mt,e,n)}function ot(t,e,n){return a(t,yt,e,n)}function at(t,e,n){return a(t,_t,e,n)}function ut(t){return wt[t.getDay()]}function ct(t){return xt[t.getDay()]}function st(t){return Mt[t.getMonth()]}function lt(t){return Ct[t.getMonth()]}function ft(t){return bt[+(t.getHours()>=12)]}function pt(t){return wt[t.getUTCDay()]}function ht(t){return xt[t.getUTCDay()]}function dt(t){return Mt[t.getUTCMonth()]}function vt(t){return Ct[t.getUTCMonth()]}function gt(t){return bt[+(t.getUTCHours()>=12)]}var mt=t.dateTime,yt=t.date,_t=t.time,bt=t.periods,xt=t.days,wt=t.shortDays,Ct=t.months,Mt=t.shortMonths,kt=s(bt),Et=l(bt),Tt=s(xt),St=l(xt),Pt=s(wt),Nt=l(wt),At=s(Ct),Ot=l(Ct),It=s(Mt),Dt=l(Mt),Rt={a:ut,A:ct,b:st,B:lt,c:null,d:k,e:k,H:E,I:T,j:S,L:P,m:N,M:A,p:ft,S:O,U:I,w:D,W:R,x:null,X:null,y:L,Y:U,Z:F,"%":J},Lt={a:pt,A:ht,b:dt,B:vt,c:null,d:j,e:j,H:B,I:W,j:V,L:z,m:H,M:q,p:gt,S:Y,U:K,w:G,W:$,x:null,X:null,y:X,Y:Z,Z:Q,"%":J},Ut={a:c,A:tt,b:nt,B:rt,c:it,d:y,e:y,H:b,I:b,j:_,L:C,m:m,M:x,p:u,S:w,U:p,w:f,W:h,x:ot,X:at,y:v,Y:d,Z:g,"%":M};return Rt.x=e(yt,Rt),Rt.X=e(_t,Rt),Rt.c=e(mt,Rt),Lt.x=e(yt,Lt),Lt.X=e(_t,Lt),Lt.c=e(mt,Lt),{format:function(t){var n=e(t+="",Rt);return n.toString=function(){return t},n},parse:function(t){var e=n(t+="",r);return e.toString=function(){return t},e},utcFormat:function(t){var n=e(t+="",Lt);return n.toString=function(){return t},n},utcParse:function(t){var e=n(t,i);return e.toString=function(){return t},e}}}function u(t,e,n){var r=t<0?"-":"",i=(r?-t:t)+"",o=i.length;return r+(o<n?new Array(n-o+1).join(e)+i:i)}function c(t){return t.replace(it,"\\$&")}function s(t){return new RegExp("^(?:"+t.map(c).join("|")+")","i")}function l(t){for(var e={},n=-1,r=t.length;++n<r;)e[t[n].toLowerCase()]=n;return e}function f(t,e,n){var r=nt.exec(e.slice(n,n+1));return r?(t.w=+r[0],n+r[0].length):-1}function p(t,e,n){var r=nt.exec(e.slice(n));return r?(t.U=+r[0],n+r[0].length):-1}function h(t,e,n){var r=nt.exec(e.slice(n));return r?(t.W=+r[0],n+r[0].length):-1}function d(t,e,n){var r=nt.exec(e.slice(n,n+4));return r?(t.y=+r[0],n+r[0].length):-1}function v(t,e,n){var r=nt.exec(e.slice(n,n+2));return r?(t.y=+r[0]+(+r[0]>68?1900:2e3),n+r[0].length):-1}function g(t,e,n){var r=/^(Z)|([+-]\d\d)(?:\:?(\d\d))?/.exec(e.slice(n,n+6));return r?(t.Z=r[1]?0:-(r[2]+(r[3]||"00")),n+r[0].length):-1}function m(t,e,n){var r=nt.exec(e.slice(n,n+2));return r?(t.m=r[0]-1,n+r[0].length):-1}function y(t,e,n){var r=nt.exec(e.slice(n,n+2));return r?(t.d=+r[0],n+r[0].length):-1}function _(t,e,n){var r=nt.exec(e.slice(n,n+3));return r?(t.m=0,t.d=+r[0],n+r[0].length):-1}function b(t,e,n){var r=nt.exec(e.slice(n,n+2));return r?(t.H=+r[0],n+r[0].length):-1}function x(t,e,n){var r=nt.exec(e.slice(n,n+2));return r?(t.M=+r[0],n+r[0].length):-1}function w(t,e,n){var r=nt.exec(e.slice(n,n+2));return r?(t.S=+r[0],n+r[0].length):-1}function C(t,e,n){var r=nt.exec(e.slice(n,n+3));return r?(t.L=+r[0],n+r[0].length):-1}function M(t,e,n){var r=rt.exec(e.slice(n,n+1));return r?n+r[0].length:-1}function k(t,e){return u(t.getDate(),e,2)}function E(t,e){return u(t.getHours(),e,2)}function T(t,e){return u(t.getHours()%12||12,e,2)}function S(t,e){return u(1+tt.a.count(n.i(tt.b)(t),t),e,3)}function P(t,e){return u(t.getMilliseconds(),e,3)}function N(t,e){return u(t.getMonth()+1,e,2)}function A(t,e){return u(t.getMinutes(),e,2)}function O(t,e){return u(t.getSeconds(),e,2)}function I(t,e){return u(tt.c.count(n.i(tt.b)(t),t),e,2)}function D(t){return t.getDay()}function R(t,e){return u(tt.d.count(n.i(tt.b)(t),t),e,2)}function L(t,e){return u(t.getFullYear()%100,e,2)}function U(t,e){return u(t.getFullYear()%1e4,e,4)}function F(t){var e=t.getTimezoneOffset();return(e>0?"-":(e*=-1,"+"))+u(e/60|0,"0",2)+u(e%60,"0",2)}function j(t,e){return u(t.getUTCDate(),e,2)}function B(t,e){return u(t.getUTCHours(),e,2)}function W(t,e){return u(t.getUTCHours()%12||12,e,2)}function V(t,e){return u(1+tt.e.count(n.i(tt.f)(t),t),e,3)}function z(t,e){return u(t.getUTCMilliseconds(),e,3)}function H(t,e){return u(t.getUTCMonth()+1,e,2)}function q(t,e){return u(t.getUTCMinutes(),e,2)}function Y(t,e){return u(t.getUTCSeconds(),e,2)}function K(t,e){return u(tt.g.count(n.i(tt.f)(t),t),e,2)}function G(t){return t.getUTCDay()}function $(t,e){return u(tt.h.count(n.i(tt.f)(t),t),e,2)}function X(t,e){return u(t.getUTCFullYear()%100,e,2)}function Z(t,e){return u(t.getUTCFullYear()%1e4,e,4)}function Q(){return"+0000"}function J(){return"%"}var tt=n(79);e.a=a;var et={"-":"",_:" ",0:"0"},nt=/^\s*\d+/,rt=/^%/,it=/[\\\^\$\*\+\?\|\[\]\(\)\.\{\}]/g},function(t,e,n){"use strict";var r=n(8),i={listen:function(t,e,n){return t.addEventListener?(t.addEventListener(e,n,!1),{remove:function(){t.removeEventListener(e,n,!1)}}):t.attachEvent?(t.attachEvent("on"+e,n),{remove:function(){t.detachEvent("on"+e,n)}}):void 0},capture:function(t,e,n){return t.addEventListener?(t.addEventListener(e,n,!0),{remove:function(){t.removeEventListener(e,n,!0)}}):{remove:r}},registerDefault:function(){}};t.exports=i},function(t,e,n){"use strict";function r(t){try{t.focus()}catch(t){}}t.exports=r},function(t,e,n){"use strict";function r(){if("undefined"==typeof document)return null;try{return document.activeElement||document.body}catch(t){return document.body}}t.exports=r},function(t,e){function n(){throw new Error("setTimeout has not been defined")}function r(){throw new Error("clearTimeout has not been defined")}function i(t){if(l===setTimeout)return setTimeout(t,0);if((l===n||!l)&&setTimeout)return l=setTimeout,setTimeout(t,0);try{return l(t,0)}catch(e){try{return l.call(null,t,0)}catch(e){return l.call(this,t,0)}}}function o(t){if(f===clearTimeout)return clearTimeout(t);if((f===r||!f)&&clearTimeout)return f=clearTimeout,clearTimeout(t);try{return f(t)}catch(e){try{return f.call(null,t)}catch(e){return f.call(this,t)}}}function a(){v&&h&&(v=!1,h.length?d=h.concat(d):g=-1,d.length&&u())}function u(){if(!v){var t=i(a);v=!0;for(var e=d.length;e;){for(h=d,d=[];++g<e;)h&&h[g].run();g=-1,e=d.length}h=null,v=!1,o(t)}}function c(t,e){this.fun=t,this.array=e}function s(){}var l,f,p=t.exports={};!function(){try{l="function"==typeof setTimeout?setTimeout:n}catch(t){l=n}try{f="function"==typeof clearTimeout?clearTimeout:r}catch(t){f=r}}();var h,d=[],v=!1,g=-1;p.nextTick=function(t){var e=new Array(arguments.length-1);if(arguments.length>1)for(var n=1;n<arguments.length;n++)e[n-1]=arguments[n];d.push(new c(t,e)),1!==d.length||v||i(u)},c.prototype.run=function(){this.fun.apply(null,this.array)},p.title="browser",p.browser=!0,p.env={},p.argv=[],p.version="",p.versions={},p.on=s,p.addListener=s,p.once=s,p.off=s,p.removeListener=s,p.removeAllListeners=s,p.emit=s,p.binding=function(t){throw new Error("process.binding is not supported")},p.cwd=function(){return"/"},p.chdir=function(t){throw new Error("process.chdir is not supported")},p.umask=function(){
return 0}},function(t,e,n){"use strict";function r(t,e){return t+e.charAt(0).toUpperCase()+e.substring(1)}var i={animationIterationCount:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridRow:!0,gridColumn:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},o=["Webkit","ms","Moz","O"];Object.keys(i).forEach(function(t){o.forEach(function(e){i[r(e,t)]=i[t]})});var a={background:{backgroundAttachment:!0,backgroundColor:!0,backgroundImage:!0,backgroundPositionX:!0,backgroundPositionY:!0,backgroundRepeat:!0},backgroundPosition:{backgroundPositionX:!0,backgroundPositionY:!0},border:{borderWidth:!0,borderStyle:!0,borderColor:!0},borderBottom:{borderBottomWidth:!0,borderBottomStyle:!0,borderBottomColor:!0},borderLeft:{borderLeftWidth:!0,borderLeftStyle:!0,borderLeftColor:!0},borderRight:{borderRightWidth:!0,borderRightStyle:!0,borderRightColor:!0},borderTop:{borderTopWidth:!0,borderTopStyle:!0,borderTopColor:!0},font:{fontStyle:!0,fontVariant:!0,fontWeight:!0,fontSize:!0,lineHeight:!0,fontFamily:!0},outline:{outlineWidth:!0,outlineStyle:!0,outlineColor:!0}},u={isUnitlessNumber:i,shorthandPropertyExpansions:a};t.exports=u},function(t,e,n){"use strict";function r(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}var i=n(2),o=n(17),a=(n(0),function(){function t(e){r(this,t),this._callbacks=null,this._contexts=null,this._arg=e}return t.prototype.enqueue=function(t,e){this._callbacks=this._callbacks||[],this._callbacks.push(t),this._contexts=this._contexts||[],this._contexts.push(e)},t.prototype.notifyAll=function(){var t=this._callbacks,e=this._contexts,n=this._arg;if(t&&e){t.length!==e.length?i("24"):void 0,this._callbacks=null,this._contexts=null;for(var r=0;r<t.length;r++)t[r].call(e[r],n);t.length=0,e.length=0}},t.prototype.checkpoint=function(){return this._callbacks?this._callbacks.length:0},t.prototype.rollback=function(t){this._callbacks&&this._contexts&&(this._callbacks.length=t,this._contexts.length=t)},t.prototype.reset=function(){this._callbacks=null,this._contexts=null},t.prototype.destructor=function(){this.reset()},t}());t.exports=o.addPoolingTo(a)},function(t,e,n){"use strict";function r(t){return!!s.hasOwnProperty(t)||!c.hasOwnProperty(t)&&(u.test(t)?(s[t]=!0,!0):(c[t]=!0,!1))}function i(t,e){return null==e||t.hasBooleanValue&&!e||t.hasNumericValue&&isNaN(e)||t.hasPositiveNumericValue&&e<1||t.hasOverloadedBooleanValue&&e===!1}var o=n(21),a=(n(4),n(9),n(394)),u=(n(1),new RegExp("^["+o.ATTRIBUTE_NAME_START_CHAR+"]["+o.ATTRIBUTE_NAME_CHAR+"]*$")),c={},s={},l={createMarkupForID:function(t){return o.ID_ATTRIBUTE_NAME+"="+a(t)},setAttributeForID:function(t,e){t.setAttribute(o.ID_ATTRIBUTE_NAME,e)},createMarkupForRoot:function(){return o.ROOT_ATTRIBUTE_NAME+'=""'},setAttributeForRoot:function(t){t.setAttribute(o.ROOT_ATTRIBUTE_NAME,"")},createMarkupForProperty:function(t,e){var n=o.properties.hasOwnProperty(t)?o.properties[t]:null;if(n){if(i(n,e))return"";var r=n.attributeName;return n.hasBooleanValue||n.hasOverloadedBooleanValue&&e===!0?r+'=""':r+"="+a(e)}return o.isCustomAttribute(t)?null==e?"":t+"="+a(e):null},createMarkupForCustomAttribute:function(t,e){return r(t)&&null!=e?t+"="+a(e):""},setValueForProperty:function(t,e,n){var r=o.properties.hasOwnProperty(e)?o.properties[e]:null;if(r){var a=r.mutationMethod;if(a)a(t,n);else{if(i(r,n))return void this.deleteValueForProperty(t,e);if(r.mustUseProperty)t[r.propertyName]=n;else{var u=r.attributeName,c=r.attributeNamespace;c?t.setAttributeNS(c,u,""+n):r.hasBooleanValue||r.hasOverloadedBooleanValue&&n===!0?t.setAttribute(u,""):t.setAttribute(u,""+n)}}}else if(o.isCustomAttribute(e))return void l.setValueForAttribute(t,e,n)},setValueForAttribute:function(t,e,n){if(r(e)){null==n?t.removeAttribute(e):t.setAttribute(e,""+n)}},deleteValueForAttribute:function(t,e){t.removeAttribute(e)},deleteValueForProperty:function(t,e){var n=o.properties.hasOwnProperty(e)?o.properties[e]:null;if(n){var r=n.mutationMethod;if(r)r(t,void 0);else if(n.mustUseProperty){var i=n.propertyName;n.hasBooleanValue?t[i]=!1:t[i]=""}else t.removeAttribute(n.attributeName)}else o.isCustomAttribute(e)&&t.removeAttribute(e)}};t.exports=l},function(t,e,n){"use strict";var r={hasCachedChildNodes:1};t.exports=r},function(t,e,n){"use strict";function r(){if(this._rootNodeID&&this._wrapperState.pendingUpdate){this._wrapperState.pendingUpdate=!1;var t=this._currentElement.props,e=u.getValue(t);null!=e&&i(this,Boolean(t.multiple),e)}}function i(t,e,n){var r,i,o=c.getNodeFromInstance(t).options;if(e){for(r={},i=0;i<n.length;i++)r[""+n[i]]=!0;for(i=0;i<o.length;i++){var a=r.hasOwnProperty(o[i].value);o[i].selected!==a&&(o[i].selected=a)}}else{for(r=""+n,i=0;i<o.length;i++)if(o[i].value===r)return void(o[i].selected=!0);o.length&&(o[0].selected=!0)}}function o(t){var e=this._currentElement.props,n=u.executeOnChange(e,t);return this._rootNodeID&&(this._wrapperState.pendingUpdate=!0),s.asap(r,this),n}var a=n(3),u=n(85),c=n(4),s=n(11),l=(n(1),!1),f={getHostProps:function(t,e){return a({},e,{onChange:t._wrapperState.onChange,value:void 0})},mountWrapper:function(t,e){var n=u.getValue(e);t._wrapperState={pendingUpdate:!1,initialValue:null!=n?n:e.defaultValue,listeners:null,onChange:o.bind(t),wasMultiple:Boolean(e.multiple)},void 0===e.value||void 0===e.defaultValue||l||(l=!0)},getSelectValueContext:function(t){return t._wrapperState.initialValue},postUpdateWrapper:function(t){var e=t._currentElement.props;t._wrapperState.initialValue=void 0;var n=t._wrapperState.wasMultiple;t._wrapperState.wasMultiple=Boolean(e.multiple);var r=u.getValue(e);null!=r?(t._wrapperState.pendingUpdate=!1,i(t,Boolean(e.multiple),r)):n!==Boolean(e.multiple)&&(null!=e.defaultValue?i(t,Boolean(e.multiple),e.defaultValue):i(t,Boolean(e.multiple),e.multiple?[]:""))}};t.exports=f},function(t,e,n){"use strict";var r,i={injectEmptyComponentFactory:function(t){r=t}},o={create:function(t){return r(t)}};o.injection=i,t.exports=o},function(t,e,n){"use strict";var r={logTopLevelRenders:!1};t.exports=r},function(t,e,n){"use strict";function r(t){return u?void 0:a("111",t.type),new u(t)}function i(t){return new c(t)}function o(t){return t instanceof c}var a=n(2),u=(n(0),null),c=null,s={injectGenericComponentClass:function(t){u=t},injectTextComponentClass:function(t){c=t}},l={createInternalComponent:r,createInstanceForText:i,isTextComponent:o,injection:s};t.exports=l},function(t,e,n){"use strict";function r(t){return o(document.documentElement,t)}var i=n(353),o=n(320),a=n(151),u=n(152),c={hasSelectionCapabilities:function(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e&&("input"===e&&"text"===t.type||"textarea"===e||"true"===t.contentEditable)},getSelectionInformation:function(){var t=u();return{focusedElem:t,selectionRange:c.hasSelectionCapabilities(t)?c.getSelection(t):null}},restoreSelection:function(t){var e=u(),n=t.focusedElem,i=t.selectionRange;e!==n&&r(n)&&(c.hasSelectionCapabilities(n)&&c.setSelection(n,i),a(n))},getSelection:function(t){var e;if("selectionStart"in t)e={start:t.selectionStart,end:t.selectionEnd};else if(document.selection&&t.nodeName&&"input"===t.nodeName.toLowerCase()){var n=document.selection.createRange();n.parentElement()===t&&(e={start:-n.moveStart("character",-t.value.length),end:-n.moveEnd("character",-t.value.length)})}else e=i.getOffsets(t);return e||{start:0,end:0}},setSelection:function(t,e){var n=e.start,r=e.end;if(void 0===r&&(r=n),"selectionStart"in t)t.selectionStart=n,t.selectionEnd=Math.min(r,t.value.length);else if(document.selection&&t.nodeName&&"input"===t.nodeName.toLowerCase()){var o=t.createTextRange();o.collapse(!0),o.moveStart("character",n),o.moveEnd("character",r-n),o.select()}else i.setOffsets(t,e)}};t.exports=c},function(t,e,n){"use strict";function r(t,e){for(var n=Math.min(t.length,e.length),r=0;r<n;r++)if(t.charAt(r)!==e.charAt(r))return r;return t.length===e.length?-1:n}function i(t){return t?t.nodeType===D?t.documentElement:t.firstChild:null}function o(t){return t.getAttribute&&t.getAttribute(A)||""}function a(t,e,n,r,i){var o;if(x.logTopLevelRenders){var a=t._currentElement.props.child,u=a.type;o="React mount: "+("string"==typeof u?u:u.displayName||u.name),console.time(o)}var c=M.mountComponent(t,n,null,_(t,e),i,0);o&&console.timeEnd(o),t._renderedComponent._topLevelWrapper=t,j._mountImageIntoNode(c,e,t,r,n)}function u(t,e,n,r){var i=E.ReactReconcileTransaction.getPooled(!n&&b.useCreateElement);i.perform(a,null,t,e,i,n,r),E.ReactReconcileTransaction.release(i)}function c(t,e,n){for(M.unmountComponent(t,n),e.nodeType===D&&(e=e.documentElement);e.lastChild;)e.removeChild(e.lastChild)}function s(t){var e=i(t);if(e){var n=y.getInstanceFromNode(e);return!(!n||!n._hostParent)}}function l(t){return!(!t||t.nodeType!==I&&t.nodeType!==D&&t.nodeType!==R)}function f(t){var e=i(t),n=e&&y.getInstanceFromNode(e);return n&&!n._hostParent?n:null}function p(t){var e=f(t);return e?e._hostContainerInfo._topLevelWrapper:null}var h=n(2),d=n(20),v=n(21),g=n(26),m=n(51),y=(n(15),n(4)),_=n(347),b=n(349),x=n(160),w=n(40),C=(n(9),n(363)),M=n(24),k=n(88),E=n(11),T=n(38),S=n(169),P=(n(0),n(55)),N=n(95),A=(n(1),v.ID_ATTRIBUTE_NAME),O=v.ROOT_ATTRIBUTE_NAME,I=1,D=9,R=11,L={},U=1,F=function(){this.rootID=U++};F.prototype.isReactComponent={},F.prototype.render=function(){return this.props.child},F.isReactTopLevelWrapper=!0;var j={TopLevelWrapper:F,_instancesByReactRootID:L,scrollMonitor:function(t,e){e()},_updateRootComponent:function(t,e,n,r,i){return j.scrollMonitor(r,function(){k.enqueueElementInternal(t,e,n),i&&k.enqueueCallbackInternal(t,i)}),t},_renderNewRootComponent:function(t,e,n,r){l(e)?void 0:h("37"),m.ensureScrollValueMonitoring();var i=S(t,!1);E.batchedUpdates(u,i,e,n,r);var o=i._instance.rootID;return L[o]=i,i},renderSubtreeIntoContainer:function(t,e,n,r){return null!=t&&w.has(t)?void 0:h("38"),j._renderSubtreeIntoContainer(t,e,n,r)},_renderSubtreeIntoContainer:function(t,e,n,r){k.validateCallback(r,"ReactDOM.render"),g.isValidElement(e)?void 0:h("39","string"==typeof e?" Instead of passing a string like 'div', pass React.createElement('div') or <div />.":"function"==typeof e?" Instead of passing a class like Foo, pass React.createElement(Foo) or <Foo />.":null!=e&&void 0!==e.props?" This may be caused by unintentionally loading two independent copies of React.":"");var a,u=g.createElement(F,{child:e});if(t){var c=w.get(t);a=c._processChildContext(c._context)}else a=T;var l=p(n);if(l){var f=l._currentElement,d=f.props.child;if(N(d,e)){var v=l._renderedComponent.getPublicInstance(),m=r&&function(){r.call(v)};return j._updateRootComponent(l,u,a,n,m),v}j.unmountComponentAtNode(n)}var y=i(n),_=y&&!!o(y),b=s(n),x=_&&!l&&!b,C=j._renderNewRootComponent(u,n,x,a)._renderedComponent.getPublicInstance();return r&&r.call(C),C},render:function(t,e,n){return j._renderSubtreeIntoContainer(null,t,e,n)},unmountComponentAtNode:function(t){l(t)?void 0:h("40");var e=p(t);if(!e){s(t),1===t.nodeType&&t.hasAttribute(O);return!1}return delete L[e._instance.rootID],E.batchedUpdates(c,e,t,!1),!0},_mountImageIntoNode:function(t,e,n,o,a){if(l(e)?void 0:h("41"),o){var u=i(e);if(C.canReuseMarkup(t,u))return void y.precacheNode(n,u);var c=u.getAttribute(C.CHECKSUM_ATTR_NAME);u.removeAttribute(C.CHECKSUM_ATTR_NAME);var s=u.outerHTML;u.setAttribute(C.CHECKSUM_ATTR_NAME,c);var f=t,p=r(f,s),v=" (client) "+f.substring(p-20,p+20)+"\n (server) "+s.substring(p-20,p+20);e.nodeType===D?h("42",v):void 0}if(e.nodeType===D?h("43"):void 0,a.useCreateElement){for(;e.lastChild;)e.removeChild(e.lastChild);d.insertTreeBefore(e,t,null)}else P(e,t),y.precacheNode(n,e.firstChild)}};t.exports=j},function(t,e,n){"use strict";var r=n(2),i=n(26),o=(n(0),{HOST:0,COMPOSITE:1,EMPTY:2,getType:function(t){return null===t||t===!1?o.EMPTY:i.isValidElement(t)?"function"==typeof t.type?o.COMPOSITE:o.HOST:void r("26",t)}});t.exports=o},function(t,e,n){"use strict";function r(t,e){return null==e?i("30"):void 0,null==t?e:Array.isArray(t)?Array.isArray(e)?(t.push.apply(t,e),t):(t.push(e),t):Array.isArray(e)?[t].concat(e):[t,e]}var i=n(2);n(0);t.exports=r},function(t,e,n){"use strict";function r(t,e,n){Array.isArray(t)?t.forEach(e,n):t&&e.call(n,t)}t.exports=r},function(t,e,n){"use strict";function r(t){for(var e;(e=t._renderedNodeType)===i.COMPOSITE;)t=t._renderedComponent;return e===i.HOST?t._renderedComponent:e===i.EMPTY?null:void 0}var i=n(164);t.exports=r},function(t,e,n){"use strict";function r(){return!o&&i.canUseDOM&&(o="textContent"in document.documentElement?"textContent":"innerText"),o}var i=n(6),o=null;t.exports=r},function(t,e,n){"use strict";function r(t){if(t){var e=t.getName();if(e)return" Check the render method of `"+e+"`."}return""}function i(t){return"function"==typeof t&&"undefined"!=typeof t.prototype&&"function"==typeof t.prototype.mountComponent&&"function"==typeof t.prototype.receiveComponent}function o(t,e){var n;if(null===t||t===!1)n=s.create(o);else if("object"==typeof t){var u=t,c=u.type;if("function"!=typeof c&&"string"!=typeof c){var p="";p+=r(u._owner),a("130",null==c?c:typeof c,p)}"string"==typeof u.type?n=l.createInternalComponent(u):i(u.type)?(n=new u.type(u),n.getHostNode||(n.getHostNode=n.getNativeNode)):n=new f(u)}else"string"==typeof t||"number"==typeof t?n=l.createInstanceForText(t):a("131",typeof t);return n._mountIndex=0,n._mountImage=null,n}var a=n(2),u=n(3),c=n(344),s=n(159),l=n(161),f=(n(391),n(0),n(1),function(t){this.construct(t)});u(f.prototype,c,{_instantiateReactComponent:o}),t.exports=o},function(t,e,n){"use strict";function r(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return"input"===e?!!i[t.type]:"textarea"===e}var i={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};t.exports=r},function(t,e,n){"use strict";var r=n(6),i=n(54),o=n(55),a=function(t,e){if(e){var n=t.firstChild;if(n&&n===t.lastChild&&3===n.nodeType)return void(n.nodeValue=e)}t.textContent=e};r.canUseDOM&&("textContent"in document.documentElement||(a=function(t,e){return 3===t.nodeType?void(t.nodeValue=e):void o(t,i(e))})),t.exports=a},function(t,e,n){"use strict";function r(t,e){return t&&"object"==typeof t&&null!=t.key?s.escape(t.key):e.toString(36)}function i(t,e,n,o){var p=typeof t;if("undefined"!==p&&"boolean"!==p||(t=null),null===t||"string"===p||"number"===p||"object"===p&&t.$$typeof===u)return n(o,t,""===e?l+r(t,0):e),1;var h,d,v=0,g=""===e?l:e+f;if(Array.isArray(t))for(var m=0;m<t.length;m++)h=t[m],d=g+r(h,m),v+=i(h,d,n,o);else{var y=c(t);if(y){var _,b=y.call(t);if(y!==t.entries)for(var x=0;!(_=b.next()).done;)h=_.value,d=g+r(h,x++),v+=i(h,d,n,o);else for(;!(_=b.next()).done;){var w=_.value;w&&(h=w[1],d=g+s.escape(w[0])+f+r(h,0),v+=i(h,d,n,o))}}else if("object"===p){var C="",M=String(t);a("31","[object Object]"===M?"object with keys {"+Object.keys(t).join(", ")+"}":M,C)}}return v}function o(t,e,n){return null==t?0:i(t,"",e,n)}var a=n(2),u=(n(15),n(359)),c=n(390),s=(n(0),n(84)),l=(n(1),"."),f=":";t.exports=o},function(t,e,n){"use strict";function r(t){var e=Function.prototype.toString,n=Object.prototype.hasOwnProperty,r=RegExp("^"+e.call(n).replace(/[\\^$.*+?()[\]{}|]/g,"\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$");try{var i=e.call(t);return r.test(i)}catch(t){return!1}}function i(t){var e=s(t);if(e){var n=e.childIDs;l(t),n.forEach(i)}}function o(t,e,n){return"\n    in "+(t||"Unknown")+(e?" (at "+e.fileName.replace(/^.*[\\\/]/,"")+":"+e.lineNumber+")":n?" (created by "+n+")":"")}function a(t){return null==t?"#empty":"string"==typeof t||"number"==typeof t?"#text":"string"==typeof t.type?t.type:t.type.displayName||t.type.name||"Unknown"}function u(t){var e,n=k.getDisplayName(t),r=k.getElement(t),i=k.getOwnerID(t);return i&&(e=k.getDisplayName(i)),o(n,r&&r._source,e)}var c,s,l,f,p,h,d,v=n(28),g=n(15),m=(n(0),n(1),"function"==typeof Array.from&&"function"==typeof Map&&r(Map)&&null!=Map.prototype&&"function"==typeof Map.prototype.keys&&r(Map.prototype.keys)&&"function"==typeof Set&&r(Set)&&null!=Set.prototype&&"function"==typeof Set.prototype.keys&&r(Set.prototype.keys));if(m){var y=new Map,_=new Set;c=function(t,e){y.set(t,e)},s=function(t){return y.get(t)},l=function(t){y.delete(t)},f=function(){return Array.from(y.keys())},p=function(t){_.add(t)},h=function(t){_.delete(t)},d=function(){return Array.from(_.keys())}}else{var b={},x={},w=function(t){return"."+t},C=function(t){return parseInt(t.substr(1),10)};c=function(t,e){var n=w(t);b[n]=e},s=function(t){var e=w(t);return b[e]},l=function(t){var e=w(t);delete b[e]},f=function(){return Object.keys(b).map(C)},p=function(t){var e=w(t);x[e]=!0},h=function(t){var e=w(t);delete x[e]},d=function(){return Object.keys(x).map(C)}}var M=[],k={onSetChildren:function(t,e){var n=s(t);n?void 0:v("144"),n.childIDs=e;for(var r=0;r<e.length;r++){var i=e[r],o=s(i);o?void 0:v("140"),null==o.childIDs&&"object"==typeof o.element&&null!=o.element?v("141"):void 0,o.isMounted?void 0:v("71"),null==o.parentID&&(o.parentID=t),o.parentID!==t?v("142",i,o.parentID,t):void 0}},onBeforeMountComponent:function(t,e,n){var r={element:e,parentID:n,text:null,childIDs:[],isMounted:!1,updateCount:0};c(t,r)},onBeforeUpdateComponent:function(t,e){var n=s(t);n&&n.isMounted&&(n.element=e)},onMountComponent:function(t){var e=s(t);e?void 0:v("144"),e.isMounted=!0;var n=0===e.parentID;n&&p(t)},onUpdateComponent:function(t){var e=s(t);e&&e.isMounted&&e.updateCount++},onUnmountComponent:function(t){var e=s(t);if(e){e.isMounted=!1;var n=0===e.parentID;n&&h(t)}M.push(t)},purgeUnmountedComponents:function(){if(!k._preventPurging){for(var t=0;t<M.length;t++){var e=M[t];i(e)}M.length=0}},isMounted:function(t){var e=s(t);return!!e&&e.isMounted},getCurrentStackAddendum:function(t){var e="";if(t){var n=a(t),r=t._owner;e+=o(n,t._source,r&&r.getName())}var i=g.current,u=i&&i._debugID;return e+=k.getStackAddendumByID(u)},getStackAddendumByID:function(t){for(var e="";t;)e+=u(t),t=k.getParentID(t);return e},getChildIDs:function(t){var e=s(t);return e?e.childIDs:[]},getDisplayName:function(t){var e=k.getElement(t);return e?a(e):null},getElement:function(t){var e=s(t);return e?e.element:null},getOwnerID:function(t){var e=k.getElement(t);return e&&e._owner?e._owner._debugID:null},getParentID:function(t){var e=s(t);return e?e.parentID:null},getSource:function(t){var e=s(t),n=e?e.element:null,r=null!=n?n._source:null;return r},getText:function(t){var e=k.getElement(t);return"string"==typeof e?e:"number"==typeof e?""+e:null},getUpdateCount:function(t){var e=s(t);return e?e.updateCount:0},getRootIDs:d,getRegisteredIDs:f};t.exports=k},function(t,e,n){"use strict";var r="function"==typeof Symbol&&Symbol.for&&Symbol.for("react.element")||60103;t.exports=r},function(t,e,n){"use strict";var r={};t.exports=r},function(t,e,n){"use strict";var r=!1;t.exports=r},function(t,e,n){"use strict";function r(t){var e=t&&(i&&t[i]||t[o]);if("function"==typeof e)return e}var i="function"==typeof Symbol&&Symbol.iterator,o="@@iterator";t.exports=r},,function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}function i(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function o(t,e){if(!t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!e||"object"!=typeof e&&"function"!=typeof e?t:e}function a(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function, not "+typeof e);t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,enumerable:!1,writable:!0,configurable:!0}}),e&&(Object.setPrototypeOf?Object.setPrototypeOf(t,e):t.__proto__=e)}Object.defineProperty(e,"__esModule",{value:!0});var u="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(t){return typeof t}:function(t){return t&&"function"==typeof Symbol&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t},c=function(){function t(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}return function(e,n,r){return n&&t(e.prototype,n),r&&t(e,r),e}}(),s=n(41),l=r(s),f=n(129),p=n(64),h=n(30),d=n(77),v=n(112),g=n(134),m=n(10),y=n(39),_=n(56),b=r(_),x=function(t){function e(){i(this,e);var t=o(this,(e.__proto__||Object.getPrototypeOf(e)).call(this));return window.lastAdditiveForceArrayVisualizer=t,t.topOffset=28,t.leftOffset=80,t.height=350,t.effectFormat=(0,h.format)(".2"),t.redraw=(0,y.debounce)(function(){return t.draw()},200),t}return a(e,t),c(e,[{key:"componentDidMount",value:function(){var t=this;this.mainGroup=this.svg.append("g"),this.onTopGroup=this.svg.append("g"),this.xaxisElement=this.onTopGroup.append("g").attr("transform","translate(0,35)").attr("class","force-bar-array-xaxis"),this.yaxisElement=this.onTopGroup.append("g").attr("transform","translate(0,35)").attr("class","force-bar-array-yaxis"),this.hoverGroup1=this.svg.append("g"),this.hoverGroup2=this.svg.append("g"),this.baseValueTitle=this.svg.append("text"),this.hoverLine=this.svg.append("line"),this.hoverxOutline=this.svg.append("text").attr("text-anchor","middle").attr("font-weight","bold").attr("fill","#fff").attr("stroke","#fff").attr("stroke-width","6").attr("font-size","12px"),this.hoverx=this.svg.append("text").attr("text-anchor","middle").attr("font-weight","bold").attr("fill","#000").attr("font-size","12px"),this.hoverxTitle=this.svg.append("text").attr("text-anchor","middle").attr("opacity",.6).attr("font-size","12px"),this.hoveryOutline=this.svg.append("text").attr("text-anchor","end").attr("font-weight","bold").attr("fill","#fff").attr("stroke","#fff").attr("stroke-width","6").attr("font-size","12px"),this.hovery=this.svg.append("text").attr("text-anchor","end").attr("font-weight","bold").attr("fill","#000").attr("font-size","12px"),this.xlabel=this.wrapper.select(".additive-force-array-xlabel"),this.ylabel=this.wrapper.select(".additive-force-array-ylabel");var e=void 0;"string"==typeof this.props.plot_cmap?this.props.plot_cmap in b.default.colors?e=b.default.colors[this.props.plot_cmap]:(console.log("Invalid color map name, reverting to default."),e=b.default.colors.RdBu):Array.isArray(this.props.plot_cmap)&&(e=this.props.plot_cmap),this.colors=e.map(function(t){return(0,m.hsl)(t)}),this.brighterColors=[1.45,1.6].map(function(e,n){return t.colors[n].brighter(e)});var n=(0,h.format)(",.4");if(null!=this.props.ordering_keys&&null!=this.props.ordering_keys_time_format){var r=function(t){return"object"==("undefined"==typeof t?"undefined":u(t))?this.formatTime(t):n(t)};this.parseTime=(0,d.timeParse)(this.props.ordering_keys_time_format),this.formatTime=(0,d.timeFormat)(this.props.ordering_keys_time_format),this.xtickFormat=r}else this.parseTime=null,this.formatTime=null,this.xtickFormat=n;this.xscale=(0,p.scaleLinear)(),this.xaxis=(0,v.axisBottom)().scale(this.xscale).tickSizeInner(4).tickSizeOuter(0).tickFormat(function(e){return t.xtickFormat(e)}).tickPadding(-18),this.ytickFormat=n,this.yscale=(0,p.scaleLinear)(),this.yaxis=(0,v.axisLeft)().scale(this.yscale).tickSizeInner(4).tickSizeOuter(0).tickFormat(function(e){return t.ytickFormat(t.invLinkFunction(e))}).tickPadding(2),this.xlabel.node().onchange=function(){return t.internalDraw()},this.ylabel.node().onchange=function(){return t.internalDraw()},this.svg.on("mousemove",function(e){return t.mouseMoved(e)}),this.svg.on("click",function(){return alert("This original index of the sample you clicked is "+t.nearestExpIndex)}),this.svg.on("mouseout",function(e){return t.mouseOut(e)}),window.addEventListener("resize",this.redraw),window.setTimeout(this.redraw,50)}},{key:"componentDidUpdate",value:function(){this.draw()}},{key:"mouseOut",value:function(){this.hoverLine.attr("display","none"),this.hoverx.attr("display","none"),this.hoverxOutline.attr("display","none"),this.hoverxTitle.attr("display","none"),this.hovery.attr("display","none"),this.hoveryOutline.attr("display","none"),this.hoverGroup1.attr("display","none"),this.hoverGroup2.attr("display","none")}},{key:"mouseMoved",value:function(){var t=this,e=void 0,n=void 0;this.hoverLine.attr("display",""),this.hoverx.attr("display",""),this.hoverxOutline.attr("display",""),this.hoverxTitle.attr("display",""),this.hovery.attr("display",""),this.hoveryOutline.attr("display",""),this.hoverGroup1.attr("display",""),this.hoverGroup2.attr("display","");var r=(0,f.mouse)(this.svg.node())[0];if(this.props.explanations){for(e=0;e<this.currExplanations.length;++e)(!n||Math.abs(n.xmapScaled-r)>Math.abs(this.currExplanations[e].xmapScaled-r))&&(n=this.currExplanations[e]);this.nearestExpIndex=n.origInd,this.hoverLine.attr("x1",n.xmapScaled).attr("x2",n.xmapScaled).attr("y1",0+this.topOffset).attr("y2",this.height),this.hoverx.attr("x",n.xmapScaled).attr("y",this.topOffset-5).text(this.xtickFormat(n.xmap)),this.hoverxOutline.attr("x",n.xmapScaled).attr("y",this.topOffset-5).text(this.xtickFormat(n.xmap)),this.hoverxTitle.attr("x",n.xmapScaled).attr("y",this.topOffset-18).text(n.count>1?n.count+" averaged samples":""),this.hovery.attr("x",this.leftOffset-6).attr("y",n.joinPointy).text(this.ytickFormat(this.invLinkFunction(n.joinPoint))),this.hoveryOutline.attr("x",this.leftOffset-6).attr("y",n.joinPointy).text(this.ytickFormat(this.invLinkFunction(n.joinPoint)));for(var i=[],o=void 0,a=void 0,u=this.currPosOrderedFeatures.length-1;u>=0;--u){var c=this.currPosOrderedFeatures[u],s=n.features[c];a=5+(s.posyTop+s.posyBottom)/2,(!o||a-o>=15)&&s.posyTop-s.posyBottom>=6&&(i.push(s),o=a)}var l=[];o=void 0;var p=!0,h=!1,d=void 0;try{for(var v,g=this.currNegOrderedFeatures[Symbol.iterator]();!(p=(v=g.next()).done);p=!0){var m=v.value,y=n.features[m];a=5+(y.negyTop+y.negyBottom)/2,(!o||o-a>=15)&&y.negyTop-y.negyBottom>=6&&(l.push(y),o=a)}}catch(t){h=!0,d=t}finally{try{!p&&g.return&&g.return()}finally{if(h)throw d}}var _=function(e){var r="";return null!==e.value&&void 0!==e.value&&(r=" = "+(isNaN(e.value)?e.value:t.ytickFormat(e.value))),n.count>1?"mean("+t.props.featureNames[e.ind]+")"+r:t.props.featureNames[e.ind]+r},b=this.hoverGroup1.selectAll(".pos-values").data(i);b.enter().append("text").attr("class","pos-values").merge(b).attr("x",n.xmapScaled+5).attr("y",function(t){return 4+(t.posyTop+t.posyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("stroke","#fff").attr("fill","#fff").attr("stroke-width","4").attr("stroke-linejoin","round").attr("opacity",1).text(_),b.exit().remove();var x=this.hoverGroup2.selectAll(".pos-values").data(i);x.enter().append("text").attr("class","pos-values").merge(x).attr("x",n.xmapScaled+5).attr("y",function(t){return 4+(t.posyTop+t.posyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("fill",this.colors[0]).text(_),x.exit().remove();var w=this.hoverGroup1.selectAll(".neg-values").data(l);w.enter().append("text").attr("class","neg-values").merge(w).attr("x",n.xmapScaled+5).attr("y",function(t){return 4+(t.negyTop+t.negyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("stroke","#fff").attr("fill","#fff").attr("stroke-width","4").attr("stroke-linejoin","round").attr("opacity",1).text(_),w.exit().remove();var C=this.hoverGroup2.selectAll(".neg-values").data(l);C.enter().append("text").attr("class","neg-values").merge(C).attr("x",n.xmapScaled+5).attr("y",function(t){return 4+(t.negyTop+t.negyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("fill",this.colors[1]).text(_),C.exit().remove()}}},{key:"draw",value:function(){var t=this;if(this.props.explanations&&0!==this.props.explanations.length){(0,y.each)(this.props.explanations,function(t,e){return t.origInd=e});var e={},n={},r={},i=!0,o=!1,a=void 0;try{for(var u,c=this.props.explanations[Symbol.iterator]();!(i=(u=c.next()).done);i=!0){var s=u.value;for(var l in s.features)void 0===e[l]&&(e[l]=0,n[l]=0,r[l]=0),s.features[l].effect>0?e[l]+=s.features[l].effect:n[l]-=s.features[l].effect,null!==s.features[l].value&&void 0!==s.features[l].value&&(r[l]+=1)}}catch(t){o=!0,a=t}finally{try{!i&&c.return&&c.return()}finally{if(o)throw a}}this.usedFeatures=(0,y.sortBy)((0,y.keys)(e),function(t){return-(e[t]+n[t])}),console.log("found ",this.usedFeatures.length," used features"),this.posOrderedFeatures=(0,y.sortBy)(this.usedFeatures,function(t){return e[t]}),this.negOrderedFeatures=(0,y.sortBy)(this.usedFeatures,function(t){return-n[t]}),this.singleValueFeatures=(0,y.filter)(this.usedFeatures,function(t){return r[t]>0});var f=["sample order by similarity","sample order by output value","original sample ordering"].concat(this.singleValueFeatures.map(function(e){return t.props.featureNames[e]}));null!=this.props.ordering_keys&&f.unshift("sample order by key");var p=this.xlabel.selectAll("option").data(f);p.enter().append("option").merge(p).attr("value",function(t){return t}).text(function(t){return t}),p.exit().remove();var h=this.props.outNames[0]?this.props.outNames[0]:"model output value";f=(0,y.map)(this.usedFeatures,function(e){return[t.props.featureNames[e],t.props.featureNames[e]+" effects"]}),f.unshift(["model output value",h]);var d=this.ylabel.selectAll("option").data(f);d.enter().append("option").merge(d).attr("value",function(t){return t[0]}).text(function(t){return t[1]}),d.exit().remove(),this.ylabel.style("top",(this.height-10-this.topOffset)/2+this.topOffset+"px").style("left",10-this.ylabel.node().offsetWidth/2+"px"),this.internalDraw()}}},{key:"internalDraw",value:function(){var t=this,e=!0,n=!1,r=void 0;try{for(var i,o=this.props.explanations[Symbol.iterator]();!(e=(i=o.next()).done);e=!0){var a=i.value,c=!0,s=!1,l=void 0;try{for(var f,h=this.usedFeatures[Symbol.iterator]();!(c=(f=h.next()).done);c=!0){var d=f.value;a.features.hasOwnProperty(d)||(a.features[d]={effect:0,value:0}),a.features[d].ind=d}}catch(t){s=!0,l=t}finally{try{!c&&h.return&&h.return()}finally{if(s)throw l}}}}catch(t){n=!0,r=t}finally{try{!e&&o.return&&o.return()}finally{if(n)throw r}}var v=void 0,m=this.xlabel.node().value,_="sample order by key"===m&&null!=this.props.ordering_keys_time_format;if(_?this.xscale=(0,p.scaleTime)():this.xscale=(0,p.scaleLinear)(),this.xaxis.scale(this.xscale),"sample order by similarity"===m)v=(0,y.sortBy)(this.props.explanations,function(t){return t.simIndex}),(0,y.each)(v,function(t,e){return t.xmap=e});else if("sample order by output value"===m)v=(0,y.sortBy)(this.props.explanations,function(t){return-t.outValue}),(0,y.each)(v,function(t,e){return t.xmap=e});else if("original sample ordering"===m)v=(0,y.sortBy)(this.props.explanations,function(t){return t.origInd}),(0,y.each)(v,function(t,e){return t.xmap=e});else if("sample order by key"===m)v=this.props.explanations,_?(0,y.each)(v,function(e,n){return e.xmap=t.parseTime(t.props.ordering_keys[n])}):(0,y.each)(v,function(e,n){return e.xmap=t.props.ordering_keys[n]}),v=(0,y.sortBy)(v,function(t){return t.xmap});else{var b=function(){var e=(0,y.findKey)(t.props.featureNames,function(t){return t===m});(0,y.each)(t.props.explanations,function(t,n){return t.xmap=t.features[e].value});var n=(0,y.sortBy)(t.props.explanations,function(t){return t.xmap}),r=(0,y.map)(n,function(t){return t.xmap});if("string"==typeof r[0])return alert("Ordering by category names is not yet supported."),{v:void 0};var i=(0,y.min)(r),o=(0,y.max)(r),a=(o-i)/100;v=[];for(var u=void 0,c=void 0,s=0;s<n.length;++s){var l=n[s];if(u&&!c&&l.xmap-u.xmap<=a||c&&l.xmap-c.xmap<=a){c||(c=(0,y.cloneDeep)(u),c.count=1);var f=!0,p=!1,h=void 0;try{for(var d,g=t.usedFeatures[Symbol.iterator]();!(f=(d=g.next()).done);f=!0){var _=d.value;c.features[_].effect+=l.features[_].effect,c.features[_].value+=l.features[_].value;
}}catch(t){p=!0,h=t}finally{try{!f&&g.return&&g.return()}finally{if(p)throw h}}c.count+=1}else if(u)if(c){var b=!0,x=!1,w=void 0;try{for(var C,M=t.usedFeatures[Symbol.iterator]();!(b=(C=M.next()).done);b=!0){var k=C.value;c.features[k].effect/=c.count,c.features[k].value/=c.count}}catch(t){x=!0,w=t}finally{try{!b&&M.return&&M.return()}finally{if(x)throw w}}v.push(c),c=void 0}else v.push(u);u=l}u.xmap-v[v.length-1].xmap>a&&v.push(u)}();if("object"===("undefined"==typeof b?"undefined":u(b)))return b.v}this.currUsedFeatures=this.usedFeatures,this.currPosOrderedFeatures=this.posOrderedFeatures,this.currNegOrderedFeatures=this.negOrderedFeatures;var x=this.ylabel.node().value;if("model output value"!==x){var w=v;v=(0,y.cloneDeep)(v);for(var C=(0,y.findKey)(this.props.featureNames,function(t){return t===x}),M=0;M<v.length;++M){var k=v[M].features[C];v[M].features={},v[M].features[C]=k,w[M].remapped_version=v[M]}this.currUsedFeatures=[C],this.currPosOrderedFeatures=[C],this.currNegOrderedFeatures=[C]}this.currExplanations=v,"identity"===this.props.link?this.invLinkFunction=function(e){return t.props.baseValue+e}:"logit"===this.props.link?this.invLinkFunction=function(e){return 1/(1+Math.exp(-(t.props.baseValue+e)))}:console.log("ERROR: Unrecognized link function: ",this.props.link),this.predValues=(0,y.map)(v,function(t){return(0,y.sum)((0,y.map)(t.features,function(t){return t.effect}))});var E=this.wrapper.node().offsetWidth;if(0==E)return setTimeout(function(){return t.draw(v)},500);this.svg.style("height",this.height+"px"),this.svg.style("width",E+"px");var T=(0,y.map)(v,function(t){return t.xmap});this.xscale.domain([(0,y.min)(T),(0,y.max)(T)]).range([this.leftOffset,E]).clamp(!0),this.xaxisElement.attr("transform","translate(0,"+this.topOffset+")").call(this.xaxis);for(var S=0;S<this.currExplanations.length;++S)this.currExplanations[S].xmapScaled=this.xscale(this.currExplanations[S].xmap);for(var P=v.length,N=0,A=0;A<P;++A){var O=v[A].features,I=(0,y.sum)((0,y.map)((0,y.filter)(O,function(t){return t.effect>0}),function(t){return t.effect}))||0,D=(0,y.sum)((0,y.map)((0,y.filter)(O,function(t){return t.effect<0}),function(t){return-t.effect}))||0;N=Math.max(N,2.2*Math.max(I,D))}this.yscale.domain([-N/2,N/2]).range([this.height-10,this.topOffset]),this.yaxisElement.attr("transform","translate("+this.leftOffset+",0)").call(this.yaxis);for(var R=0;R<P;++R){var L=v[R].features,U=(0,y.sum)((0,y.map)((0,y.filter)(L,function(t){return t.effect<0}),function(t){return-t.effect}))||0,F=-U,j=void 0,B=!0,W=!1,V=void 0;try{for(var z,H=this.currPosOrderedFeatures[Symbol.iterator]();!(B=(z=H.next()).done);B=!0)j=z.value,L[j].posyTop=this.yscale(F),L[j].effect>0&&(F+=L[j].effect),L[j].posyBottom=this.yscale(F),L[j].ind=j}catch(t){W=!0,V=t}finally{try{!B&&H.return&&H.return()}finally{if(W)throw V}}var q=F,Y=!0,K=!1,G=void 0;try{for(var $,X=this.currNegOrderedFeatures[Symbol.iterator]();!(Y=($=X.next()).done);Y=!0)j=$.value,L[j].negyTop=this.yscale(F),L[j].effect<0&&(F-=L[j].effect),L[j].negyBottom=this.yscale(F)}catch(t){K=!0,G=t}finally{try{!Y&&X.return&&X.return()}finally{if(K)throw G}}v[R].joinPoint=q,v[R].joinPointy=this.yscale(q)}var Z=(0,g.line)().x(function(t){return t[0]}).y(function(t){return t[1]}),Q=this.mainGroup.selectAll(".force-bar-array-area-pos").data(this.currUsedFeatures);Q.enter().append("path").attr("class","force-bar-array-area-pos").merge(Q).attr("d",function(t){var e=(0,y.map)((0,y.range)(P),function(e){return[v[e].xmapScaled,v[e].features[t].posyTop]}),n=(0,y.map)((0,y.rangeRight)(P),function(e){return[v[e].xmapScaled,v[e].features[t].posyBottom]});return Z(e.concat(n))}).attr("fill",this.colors[0]),Q.exit().remove();var J=this.mainGroup.selectAll(".force-bar-array-area-neg").data(this.currUsedFeatures);J.enter().append("path").attr("class","force-bar-array-area-neg").merge(J).attr("d",function(t){var e=(0,y.map)((0,y.range)(P),function(e){return[v[e].xmapScaled,v[e].features[t].negyTop]}),n=(0,y.map)((0,y.rangeRight)(P),function(e){return[v[e].xmapScaled,v[e].features[t].negyBottom]});return Z(e.concat(n))}).attr("fill",this.colors[1]),J.exit().remove();var tt=this.mainGroup.selectAll(".force-bar-array-divider-pos").data(this.currUsedFeatures);tt.enter().append("path").attr("class","force-bar-array-divider-pos").merge(tt).attr("d",function(t){var e=(0,y.map)((0,y.range)(P),function(e){return[v[e].xmapScaled,v[e].features[t].posyBottom]});return Z(e)}).attr("fill","none").attr("stroke-width",1).attr("stroke",function(){return t.colors[0].brighter(1.2)}),tt.exit().remove();var et=this.mainGroup.selectAll(".force-bar-array-divider-neg").data(this.currUsedFeatures);et.enter().append("path").attr("class","force-bar-array-divider-neg").merge(et).attr("d",function(t){var e=(0,y.map)((0,y.range)(P),function(e){return[v[e].xmapScaled,v[e].features[t].negyTop]});return Z(e)}).attr("fill","none").attr("stroke-width",1).attr("stroke",function(){return t.colors[1].brighter(1.5)}),et.exit().remove();for(var nt=function(t,e,n,r,i){var o=void 0,a=void 0;"pos"===i?(o=t[n].features[e].posyBottom,a=t[n].features[e].posyTop):(o=t[n].features[e].negyBottom,a=t[n].features[e].negyTop);for(var u=void 0,c=void 0,s=n+1;s<=r;++s)"pos"===i?(u=t[s].features[e].posyBottom,c=t[s].features[e].posyTop):(u=t[s].features[e].negyBottom,c=t[s].features[e].negyTop),u>o&&(o=u),c<a&&(a=c);return{top:o,bottom:a}},rt=100,it=20,ot=100,at=[],ut=["pos","neg"],ct=0;ct<ut.length;ct++){var st=ut[ct],lt=!0,ft=!1,pt=void 0;try{for(var ht,dt=this.currUsedFeatures[Symbol.iterator]();!(lt=(ht=dt.next()).done);lt=!0)for(var vt=ht.value,gt=0,mt=0,yt=0,_t={top:0,bottom:0},bt=void 0;mt<P-1;){for(;yt<rt&&mt<P-1;)++mt,yt=v[mt].xmapScaled-v[gt].xmapScaled;for(_t=nt(v,vt,gt,mt,st);_t.bottom-_t.top<it&&gt<mt;)++gt,_t=nt(v,vt,gt,mt,st);if(yt=v[mt].xmapScaled-v[gt].xmapScaled,_t.bottom-_t.top>=it&&yt>=rt){for(;mt<P-1;){if(++mt,bt=nt(v,vt,gt,mt,st),!(bt.bottom-bt.top>it)){--mt;break}_t=bt}yt=v[mt].xmapScaled-v[gt].xmapScaled,at.push([(v[mt].xmapScaled+v[gt].xmapScaled)/2,(_t.top+_t.bottom)/2,this.props.featureNames[vt]]);var xt=v[mt].xmapScaled;for(gt=mt;xt+ot>v[gt].xmapScaled&&gt<P-1;)++gt;mt=gt}}}catch(t){ft=!0,pt=t}finally{try{!lt&&dt.return&&dt.return()}finally{if(ft)throw pt}}}var wt=this.onTopGroup.selectAll(".force-bar-array-flabels").data(at);wt.enter().append("text").attr("class","force-bar-array-flabels").merge(wt).attr("x",function(t){return t[0]}).attr("y",function(t){return t[1]+4}).text(function(t){return t[2]}),wt.exit().remove()}},{key:"componentWillUnmount",value:function(){window.removeEventListener("resize",this.redraw)}},{key:"render",value:function(){var t=this;return l.default.createElement("div",{ref:function(e){return t.wrapper=(0,f.select)(e)},style:{textAlign:"center"}},l.default.createElement("style",{dangerouslySetInnerHTML:{__html:"\n          .force-bar-array-wrapper {\n            text-align: center;\n          }\n          .force-bar-array-xaxis path {\n            fill: none;\n            opacity: 0.4;\n          }\n          .force-bar-array-xaxis .domain {\n            opacity: 0;\n          }\n          .force-bar-array-xaxis paths {\n            display: none;\n          }\n          .force-bar-array-yaxis path {\n            fill: none;\n            opacity: 0.4;\n          }\n          .force-bar-array-yaxis paths {\n            display: none;\n          }\n          .tick line {\n            stroke: #000;\n            stroke-width: 1px;\n            opacity: 0.4;\n          }\n          .tick text {\n            fill: #000;\n            opacity: 0.5;\n            font-size: 12px;\n            padding: 0px;\n          }\n          .force-bar-array-flabels {\n            font-size: 12px;\n            fill: #fff;\n            text-anchor: middle;\n          }\n          .additive-force-array-xlabel {\n            background: none;\n            border: 1px solid #ccc;\n            opacity: 0.5;\n            margin-bottom: 0px;\n            font-size: 12px;\n            font-family: arial;\n            margin-left: 80px;\n            max-width: 300px;\n          }\n          .additive-force-array-xlabel:focus {\n            outline: none;\n          }\n          .additive-force-array-ylabel {\n            position: relative;\n            top: 0px;\n            left: 0px;\n            transform: rotate(-90deg);\n            background: none;\n            border: 1px solid #ccc;\n            opacity: 0.5;\n            margin-bottom: 0px;\n            font-size: 12px;\n            font-family: arial;\n            max-width: 150px;\n          }\n          .additive-force-array-ylabel:focus {\n            outline: none;\n          }\n          .additive-force-array-hoverLine {\n            stroke-width: 1px;\n            stroke: #fff;\n            opacity: 1;\n          }"}}),l.default.createElement("select",{className:"additive-force-array-xlabel"}),l.default.createElement("div",{style:{height:"0px",textAlign:"left"}},l.default.createElement("select",{className:"additive-force-array-ylabel"})),l.default.createElement("svg",{ref:function(e){return t.svg=(0,f.select)(e)},style:{userSelect:"none",display:"block",fontFamily:"arial",sansSerif:!0}}))}}]),e}(l.default.Component);x.defaultProps={plot_cmap:"RdBu",ordering_keys:null,ordering_keys_time_format:null},e.default=x},function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}function i(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function o(t,e){if(!t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!e||"object"!=typeof e&&"function"!=typeof e?t:e}function a(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function, not "+typeof e);t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,enumerable:!1,writable:!0,configurable:!0}}),e&&(Object.setPrototypeOf?Object.setPrototypeOf(t,e):t.__proto__=e)}Object.defineProperty(e,"__esModule",{value:!0});var u=function(){function t(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}return function(e,n,r){return n&&t(e.prototype,n),r&&t(e,r),e}}(),c=n(41),s=r(c),l=n(129),f=n(64),p=n(30),h=n(112),d=n(134),v=n(10),g=n(39),m=n(56),y=r(m),b=function(t){function e(){i(this,e);var t=o(this,(e.__proto__||Object.getPrototypeOf(e)).call(this));return window.lastAdditiveForceVisualizer=t,t.effectFormat=(0,p.format)(".2"),t.redraw=(0,g.debounce)(function(){return t.draw()},200),t}return a(e,t),u(e,[{key:"componentDidMount",value:function(){var t=this;this.mainGroup=this.svg.append("g"),this.axisElement=this.mainGroup.append("g").attr("transform","translate(0,35)").attr("class","force-bar-axis"),this.onTopGroup=this.svg.append("g"),this.baseValueTitle=this.svg.append("text"),this.joinPointLine=this.svg.append("line"),this.joinPointLabelOutline=this.svg.append("text"),this.joinPointLabel=this.svg.append("text"),this.joinPointTitleLeft=this.svg.append("text"),this.joinPointTitleLeftArrow=this.svg.append("text"),this.joinPointTitle=this.svg.append("text"),this.joinPointTitleRightArrow=this.svg.append("text"),this.joinPointTitleRight=this.svg.append("text"),this.hoverLabelBacking=this.svg.append("text").attr("x",10).attr("y",20).attr("text-anchor","middle").attr("font-size",12).attr("stroke","#fff").attr("fill","#fff").attr("stroke-width","4").attr("stroke-linejoin","round").text("").on("mouseover",function(){t.hoverLabel.attr("opacity",1),t.hoverLabelBacking.attr("opacity",1)}).on("mouseout",function(){t.hoverLabel.attr("opacity",0),t.hoverLabelBacking.attr("opacity",0)}),this.hoverLabel=this.svg.append("text").attr("x",10).attr("y",20).attr("text-anchor","middle").attr("font-size",12).attr("fill","#0f0").text("").on("mouseover",function(){t.hoverLabel.attr("opacity",1),t.hoverLabelBacking.attr("opacity",1)}).on("mouseout",function(){t.hoverLabel.attr("opacity",0),t.hoverLabelBacking.attr("opacity",0)});var e=void 0;"string"==typeof this.props.plot_cmap?this.props.plot_cmap in y.default.colors?e=y.default.colors[this.props.plot_cmap]:(console.log("Invalid color map name, reverting to default."),e=y.default.colors.RdBu):Array.isArray(this.props.plot_cmap)&&(e=this.props.plot_cmap),this.colors=e.map(function(t){return(0,v.hsl)(t)}),this.brighterColors=[1.45,1.6].map(function(e,n){return t.colors[n].brighter(e)}),this.colors.map(function(e,n){var r=t.svg.append("linearGradient").attr("id","linear-grad-"+n).attr("x1","0%").attr("y1","0%").attr("x2","0%").attr("y2","100%");r.append("stop").attr("offset","0%").attr("stop-color",e).attr("stop-opacity",.6),r.append("stop").attr("offset","100%").attr("stop-color",e).attr("stop-opacity",0);var i=t.svg.append("linearGradient").attr("id","linear-backgrad-"+n).attr("x1","0%").attr("y1","0%").attr("x2","0%").attr("y2","100%");i.append("stop").attr("offset","0%").attr("stop-color",e).attr("stop-opacity",.5),i.append("stop").attr("offset","100%").attr("stop-color",e).attr("stop-opacity",0)}),this.tickFormat=(0,p.format)(",.4"),this.scaleCentered=(0,f.scaleLinear)(),this.axis=(0,h.axisBottom)().scale(this.scaleCentered).tickSizeInner(4).tickSizeOuter(0).tickFormat(function(e){return t.tickFormat(t.invLinkFunction(e))}).tickPadding(-18),window.addEventListener("resize",this.redraw),window.setTimeout(this.redraw,50)}},{key:"componentDidUpdate",value:function(){this.draw()}},{key:"draw",value:function(){var t=this;(0,g.each)(this.props.featureNames,function(e,n){t.props.features[n]&&(t.props.features[n].name=e)}),"identity"===this.props.link?this.invLinkFunction=function(e){return t.props.baseValue+e}:"logit"===this.props.link?this.invLinkFunction=function(e){return 1/(1+Math.exp(-(t.props.baseValue+e)))}:console.log("ERROR: Unrecognized link function: ",this.props.link);var e=this.svg.node().parentNode.offsetWidth;if(0==e)return setTimeout(function(){return t.draw(t.props)},500);this.svg.style("height","150px"),this.svg.style("width",e+"px");var n=50,r=(0,g.sortBy)(this.props.features,function(t){return-1/(t.effect+1e-10)}),i=(0,g.sum)((0,g.map)(r,function(t){return Math.abs(t.effect)})),o=(0,g.sum)((0,g.map)((0,g.filter)(r,function(t){return t.effect>0}),function(t){return t.effect}))||0,a=(0,g.sum)((0,g.map)((0,g.filter)(r,function(t){return t.effect<0}),function(t){return-t.effect}))||0;this.domainSize=3*Math.max(o,a);var u=(0,f.scaleLinear)().domain([0,this.domainSize]).range([0,e]),c=e/2-u(a);this.scaleCentered.domain([-this.domainSize/2,this.domainSize/2]).range([0,e]).clamp(!0),this.axisElement.attr("transform","translate(0,"+n+")").call(this.axis);var s=0,l=void 0,h=void 0,v=void 0;for(l=0;l<r.length;++l)r[l].x=s,r[l].effect<0&&void 0===h&&(h=s,v=l),s+=Math.abs(r[l].effect);void 0===h&&(h=s,v=l);var m=(0,d.line)().x(function(t){return t[0]}).y(function(t){return t[1]}),y=function(e){return void 0!==e.value&&null!==e.value&&""!==e.value?e.name+" = "+(isNaN(e.value)?e.value:t.tickFormat(e.value)):e.name};r=this.props.hideBars?[]:r;var b=this.mainGroup.selectAll(".force-bar-blocks").data(r);b.enter().append("path").attr("class","force-bar-blocks").merge(b).attr("d",function(t,e){var r=u(t.x)+c,i=u(Math.abs(t.effect)),o=t.effect<0?-4:4,a=o;return e===v&&(o=0),e===v-1&&(a=0),m([[r,6+n],[r+i,6+n],[r+i+a,14.5+n],[r+i,23+n],[r,23+n],[r+o,14.5+n]])}).attr("fill",function(e){return e.effect>0?t.colors[0]:t.colors[1]}).on("mouseover",function(e){if(u(Math.abs(e.effect))<u(i)/50||u(Math.abs(e.effect))<10){var r=u(e.x)+c,o=u(Math.abs(e.effect));t.hoverLabel.attr("opacity",1).attr("x",r+o/2).attr("y",n+.5).attr("fill",e.effect>0?t.colors[0]:t.colors[1]).text(y(e)),t.hoverLabelBacking.attr("opacity",1).attr("x",r+o/2).attr("y",n+.5).text(y(e))}}).on("mouseout",function(){t.hoverLabel.attr("opacity",0),t.hoverLabelBacking.attr("opacity",0)}),b.exit().remove();var x=_.filter(r,function(t){return u(Math.abs(t.effect))>u(i)/50&&u(Math.abs(t.effect))>10}),w=this.onTopGroup.selectAll(".force-bar-labels").data(x);if(w.exit().remove(),w=w.enter().append("text").attr("class","force-bar-labels").attr("font-size","12px").attr("y",48+n).merge(w).text(function(e){return void 0!==e.value&&null!==e.value&&""!==e.value?e.name+" = "+(isNaN(e.value)?e.value:t.tickFormat(e.value)):e.name}).attr("fill",function(e){return e.effect>0?t.colors[0]:t.colors[1]}).attr("stroke",function(t){return t.textWidth=Math.max(this.getComputedTextLength(),u(Math.abs(t.effect))-10),t.innerTextWidth=this.getComputedTextLength(),"none"}),this.filteredData=x,r.length>0){s=h+u.invert(5);for(var C=v;C<r.length;++C)r[C].textx=s,s+=u.invert(r[C].textWidth+10);s=h-u.invert(5);for(var M=v-1;M>=0;--M)r[M].textx=s,s-=u.invert(r[M].textWidth+10)}w.attr("x",function(t){return u(t.textx)+c+(t.effect>0?-t.textWidth/2:t.textWidth/2)}).attr("text-anchor","middle"),x=(0,g.filter)(x,function(n){return u(n.textx)+c>t.props.labelMargin&&u(n.textx)+c<e-t.props.labelMargin}),this.filteredData2=x;var k=x.slice(),E=(0,g.findIndex)(r,x[0])-1;E>=0&&k.unshift(r[E]);var T=this.mainGroup.selectAll(".force-bar-labelBacking").data(x);T.enter().append("path").attr("class","force-bar-labelBacking").attr("stroke","none").attr("opacity",.2).merge(T).attr("d",function(t){return m([[u(t.x)+u(Math.abs(t.effect))+c,23+n],[(t.effect>0?u(t.textx):u(t.textx)+t.textWidth)+c+5,33+n],[(t.effect>0?u(t.textx):u(t.textx)+t.textWidth)+c+5,54+n],[(t.effect>0?u(t.textx)-t.textWidth:u(t.textx))+c-5,54+n],[(t.effect>0?u(t.textx)-t.textWidth:u(t.textx))+c-5,33+n],[u(t.x)+c,23+n]])}).attr("fill",function(t){return"url(#linear-backgrad-"+(t.effect>0?0:1)+")"}),T.exit().remove();var S=this.mainGroup.selectAll(".force-bar-labelDividers").data(x.slice(0,-1));S.enter().append("rect").attr("class","force-bar-labelDividers").attr("height","21px").attr("width","1px").attr("y",33+n).merge(S).attr("x",function(t){return(t.effect>0?u(t.textx):u(t.textx)+t.textWidth)+c+4.5}).attr("fill",function(t){return"url(#linear-grad-"+(t.effect>0?0:1)+")"}),S.exit().remove();var P=this.mainGroup.selectAll(".force-bar-labelLinks").data(x.slice(0,-1));P.enter().append("line").attr("class","force-bar-labelLinks").attr("y1",23+n).attr("y2",33+n).attr("stroke-opacity",.5).attr("stroke-width",1).merge(P).attr("x1",function(t){return u(t.x)+u(Math.abs(t.effect))+c}).attr("x2",function(t){return(t.effect>0?u(t.textx):u(t.textx)+t.textWidth)+c+5}).attr("stroke",function(e){return e.effect>0?t.colors[0]:t.colors[1]}),P.exit().remove();var N=this.mainGroup.selectAll(".force-bar-blockDividers").data(r.slice(0,-1));N.enter().append("path").attr("class","force-bar-blockDividers").attr("stroke-width",2).attr("fill","none").merge(N).attr("d",function(t){var e=u(t.x)+u(Math.abs(t.effect))+c;return m([[e,6+n],[e+(t.effect<0?-4:4),14.5+n],[e,23+n]])}).attr("stroke",function(e,n){return v===n+1||Math.abs(e.effect)<1e-8?"#rgba(0,0,0,0)":e.effect>0?t.brighterColors[0]:t.brighterColors[1]}),N.exit().remove(),this.joinPointLine.attr("x1",u(h)+c).attr("x2",u(h)+c).attr("y1",0+n).attr("y2",6+n).attr("stroke","#F2F2F2").attr("stroke-width",1).attr("opacity",1),this.joinPointLabelOutline.attr("x",u(h)+c).attr("y",-5+n).attr("color","#fff").attr("text-anchor","middle").attr("font-weight","bold").attr("stroke","#fff").attr("stroke-width",6).text((0,p.format)(",.2f")(this.invLinkFunction(h-a))).attr("opacity",1),console.log("joinPoint",h,c,n,a),this.joinPointLabel.attr("x",u(h)+c).attr("y",-5+n).attr("text-anchor","middle").attr("font-weight","bold").attr("fill","#000").text((0,p.format)(",.2f")(this.invLinkFunction(h-a))).attr("opacity",1),this.joinPointTitle.attr("x",u(h)+c).attr("y",-22+n).attr("text-anchor","middle").attr("font-size","12").attr("fill","#000").text(this.props.outNames[0]).attr("opacity",.5),this.props.hideBars||(this.joinPointTitleLeft.attr("x",u(h)+c-16).attr("y",-38+n).attr("text-anchor","end").attr("font-size","13").attr("fill",this.colors[0]).text("higher").attr("opacity",1),this.joinPointTitleRight.attr("x",u(h)+c+16).attr("y",-38+n).attr("text-anchor","start").attr("font-size","13").attr("fill",this.colors[1]).text("lower").attr("opacity",1),this.joinPointTitleLeftArrow.attr("x",u(h)+c+7).attr("y",-42+n).attr("text-anchor","end").attr("font-size","13").attr("fill",this.colors[0]).text("→").attr("opacity",1),this.joinPointTitleRightArrow.attr("x",u(h)+c-7).attr("y",-36+n).attr("text-anchor","start").attr("font-size","13").attr("fill",this.colors[1]).text("←").attr("opacity",1)),this.props.hideBaseValueLabel||this.baseValueTitle.attr("x",this.scaleCentered(0)).attr("y",-22+n).attr("text-anchor","middle").attr("font-size","12").attr("fill","#000").text("base value").attr("opacity",.5)}},{key:"componentWillUnmount",value:function(){window.removeEventListener("resize",this.redraw)}},{key:"render",value:function(){var t=this;return s.default.createElement("svg",{ref:function(e){return t.svg=(0,l.select)(e)},style:{userSelect:"none",display:"block",fontFamily:"arial",sansSerif:!0}},s.default.createElement("style",{dangerouslySetInnerHTML:{__html:"\n          .force-bar-axis path {\n            fill: none;\n            opacity: 0.4;\n          }\n          .force-bar-axis paths {\n            display: none;\n          }\n          .tick line {\n            stroke: #000;\n            stroke-width: 1px;\n            opacity: 0.4;\n          }\n          .tick text {\n            fill: #000;\n            opacity: 0.5;\n            font-size: 12px;\n            padding: 0px;\n          }"}}))}}]),e}(s.default.Component);b.defaultProps={plot_cmap:"RdBu"},e.default=b},function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}function i(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function o(t,e){if(!t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!e||"object"!=typeof e&&"function"!=typeof e?t:e}function a(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function, not "+typeof e);t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,enumerable:!1,writable:!0,configurable:!0}}),e&&(Object.setPrototypeOf?Object.setPrototypeOf(t,e):t.__proto__=e)}Object.defineProperty(e,"__esModule",{value:!0});var u=function(){function t(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}return function(e,n,r){return n&&t(e.prototype,n),r&&t(e,r),e}}(),c=n(41),s=r(c),l=n(64),f=n(30),p=n(39),h=n(56),d=r(h),v=function(t){function e(){i(this,e);var t=o(this,(e.__proto__||Object.getPrototypeOf(e)).call(this));return t.width=100,window.lastSimpleListInstance=t,t.effectFormat=(0,f.format)(".2"),t}return a(e,t),u(e,[{key:"render",value:function(){var t=this,e=void 0;"string"==typeof this.props.plot_cmap?this.props.plot_cmap in d.default.colors?e=d.default.colors[this.props.plot_cmap]:(console.log("Invalid color map name, reverting to default."),e=d.default.colors.RdBu):Array.isArray(this.props.plot_cmap)&&(e=this.props.plot_cmap),console.log(this.props.features,this.props.features),this.scale=(0,l.scaleLinear)().domain([0,(0,p.max)((0,p.map)(this.props.features,function(t){return Math.abs(t.effect)}))]).range([0,this.width]);var n=(0,p.reverse)((0,p.sortBy)(Object.keys(this.props.features),function(e){return Math.abs(t.props.features[e].effect)})),r=n.map(function(n){var r=t.props.features[n],i=t.props.featureNames[n],o={width:t.scale(Math.abs(r.effect)),height:"20px",background:r.effect<0?e[0]:e[1],display:"inline-block"},a=void 0,u=void 0,c={lineHeight:"20px",display:"inline-block",width:t.width+40,verticalAlign:"top",marginRight:"5px",textAlign:"right"},l={lineHeight:"20px",display:"inline-block",width:t.width+40,verticalAlign:"top",marginLeft:"5px"};return r.effect<0?(u=s.default.createElement("span",{style:l},i),c.width=40+t.width-t.scale(Math.abs(r.effect)),c.textAlign="right",c.color="#999",c.fontSize="13px",a=s.default.createElement("span",{style:c},t.effectFormat(r.effect))):(c.textAlign="right",a=s.default.createElement("span",{style:c},i),l.width=40,l.textAlign="left",l.color="#999",l.fontSize="13px",u=s.default.createElement("span",{style:l},t.effectFormat(r.effect))),s.default.createElement("div",{key:n,style:{marginTop:"2px"}},a,s.default.createElement("div",{style:o}),u)});return s.default.createElement("span",null,r)}}]),e}(s.default.Component);v.defaultProps={plot_cmap:"RdBu"},e.default=v},function(t,e,n){"use strict";t.exports=n(345)},function(t,e,n){var r=(n(0),n(398)),i=!1;t.exports=function(t){t=t||{};var e=t.shouldRejectClick||r;i=!0,n(22).injection.injectEventPluginsByName({TapEventPlugin:n(396)(e)})}},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict"},function(t,e,n){"use strict";n(101),n(102),n(184),n(105),n(187),n(109),n(108)},function(t,e,n){"use strict";e.a=function(t){return t}},function(t,e,n){"use strict"},function(t,e,n){"use strict";n(29)},function(t,e,n){"use strict";n(18),n(29),n(57)},function(t,e,n){"use strict"},function(t,e,n){"use strict"},function(t,e,n){"use strict"},function(t,e,n){"use strict";n(18)},function(t,e,n){"use strict"},function(t,e,n){"use strict"},function(t,e,n){"use strict";n(101),n(18),n(29),n(57)},function(t,e,n){"use strict";n(104)},function(t,e,n){"use strict";n(110)},function(t,e,n){"use strict";n.d(e,"a",function(){return r});var r=Array.prototype.slice},function(t,e,n){"use strict";function r(t,e,n){var r=t(n);return"translate("+(isFinite(r)?r:e(n))+",0)"}function i(t,e,n){var r=t(n);return"translate(0,"+(isFinite(r)?r:e(n))+")"}function o(t){var e=t.bandwidth()/2;return t.round()&&(e=Math.round(e)),function(n){return t(n)+e}}function a(){return!this.__axis}function u(t,e){function n(n){var p,b=null==c?e.ticks?e.ticks.apply(e,u):e.domain():c,x=null==s?e.tickFormat?e.tickFormat.apply(e,u):h.a:s,w=Math.max(l,0)+_,C=t===d||t===g?r:i,M=e.range(),k=M[0]+.5,E=M[M.length-1]+.5,T=(e.bandwidth?o:h.a)(e.copy()),S=n.selection?n.selection():n,P=S.selectAll(".domain").data([null]),N=S.selectAll(".tick").data(b,e).order(),A=N.exit(),O=N.enter().append("g").attr("class","tick"),I=N.select("line"),D=N.select("text"),R=t===d||t===m?-1:1,L=t===m||t===v?(p="x","y"):(p="y","x");P=P.merge(P.enter().insert("path",".tick").attr("class","domain").attr("stroke","#000")),N=N.merge(O),I=I.merge(O.append("line").attr("stroke","#000").attr(p+"2",R*l).attr(L+"1",.5).attr(L+"2",.5)),D=D.merge(O.append("text").attr("fill","#000").attr(p,R*w).attr(L,.5).attr("dy",t===d?"0em":t===g?"0.71em":"0.32em")),n!==S&&(P=P.transition(n),N=N.transition(n),I=I.transition(n),D=D.transition(n),A=A.transition(n).attr("opacity",y).attr("transform",function(t){return C(T,this.parentNode.__axis||T,t)}),O.attr("opacity",y).attr("transform",function(t){return C(this.parentNode.__axis||T,T,t)})),A.remove(),P.attr("d",t===m||t==v?"M"+R*f+","+k+"H0.5V"+E+"H"+R*f:"M"+k+","+R*f+"V0.5H"+E+"V"+R*f),N.attr("opacity",1).attr("transform",function(t){return C(T,T,t)}),I.attr(p+"2",R*l),D.attr(p,R*w).text(x),S.filter(a).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",t===v?"start":t===m?"end":"middle"),S.each(function(){this.__axis=T})}var u=[],c=null,s=null,l=6,f=6,_=3;return n.scale=function(t){return arguments.length?(e=t,n):e},n.ticks=function(){return u=p.a.call(arguments),n},n.tickArguments=function(t){return arguments.length?(u=null==t?[]:p.a.call(t),n):u.slice()},n.tickValues=function(t){return arguments.length?(c=null==t?null:p.a.call(t),n):c&&c.slice()},n.tickFormat=function(t){return arguments.length?(s=t,n):s},n.tickSize=function(t){return arguments.length?(l=f=+t,n):l},n.tickSizeInner=function(t){return arguments.length?(l=+t,n):l},n.tickSizeOuter=function(t){return arguments.length?(f=+t,n):f},n.tickPadding=function(t){return arguments.length?(_=+t,n):_},n}function c(t){return u(d,t)}function s(t){return u(v,t)}function l(t){return u(g,t)}function f(t){return u(m,t)}var p=n(200),h=n(202);e.a=c,e.b=s,e.c=l,e.d=f;var d=1,v=2,g=3,m=4,y=1e-6},function(t,e,n){"use strict";e.a=function(t){return t}},function(t,e,n){"use strict";var r=(n(206),n(207),n(58));n.d(e,"a",function(){return r.a});n(205),n(208),n(204)},function(t,e,n){"use strict"},function(t,e,n){"use strict"},function(t,e,n){"use strict";n(58)},function(t,e,n){"use strict";function r(){}function i(t,e){var n=new r;if(t instanceof r)t.each(function(t){n.add(t)});else if(t){var i=-1,o=t.length;if(null==e)for(;++i<o;)n.add(t[i]);else for(;++i<o;)n.add(e(t[i],i,t))}return n}var o=n(58),a=o.a.prototype;r.prototype=i.prototype={constructor:r,has:a.has,add:function(t){return t+="",this[o.b+t]=t,this},remove:a.remove,clear:a.clear,values:a.keys,size:a.size,empty:a.empty,each:a.each}},function(t,e,n){"use strict"},function(t,e,n){"use strict";function r(t){if(t instanceof o)return new o(t.h,t.s,t.l,t.opacity);t instanceof u.d||(t=n.i(u.e)(t));var e=t.r/255,r=t.g/255,i=t.b/255,a=(g*i+d*e-v*r)/(g+d-v),s=i-a,l=(h*(r-a)-f*s)/p,m=Math.sqrt(l*l+s*s)/(h*a*(1-a)),y=m?Math.atan2(l,s)*c.a-120:NaN;return new o(y<0?y+360:y,m,a,t.opacity)}function i(t,e,n,i){return 1===arguments.length?r(t):new o(t,e,n,null==i?1:i)}function o(t,e,n,r){this.h=+t,this.s=+e,this.l=+n,this.opacity=+r}var a=n(60),u=n(59),c=n(113);e.a=i;var s=-.14861,l=1.78277,f=-.29227,p=-.90649,h=1.97294,d=h*p,v=h*l,g=l*f-p*s;n.i(a.a)(o,i,n.i(a.b)(u.f,{brighter:function(t){return t=null==t?u.g:Math.pow(u.g,t),new o(this.h,this.s,this.l*t,this.opacity)},darker:function(t){return t=null==t?u.h:Math.pow(u.h,t),new o(this.h,this.s,this.l*t,this.opacity)},rgb:function(){var t=isNaN(this.h)?0:(this.h+120)*c.b,e=+this.l,n=isNaN(this.s)?0:this.s*e*(1-e),r=Math.cos(t),i=Math.sin(t);return new u.d(255*(e+n*(s*r+l*i)),255*(e+n*(f*r+p*i)),255*(e+n*(h*r)),this.opacity)}}))},function(t,e,n){"use strict";function r(t){if(t instanceof o)return new o(t.l,t.a,t.b,t.opacity);if(t instanceof p){var e=t.h*v.b;return new o(t.l,Math.cos(e)*t.c,Math.sin(e)*t.c,t.opacity)}t instanceof d.d||(t=n.i(d.e)(t));var r=s(t.r),i=s(t.g),u=s(t.b),c=a((.4124564*r+.3575761*i+.1804375*u)/m),l=a((.2126729*r+.7151522*i+.072175*u)/y),f=a((.0193339*r+.119192*i+.9503041*u)/_);return new o(116*l-16,500*(c-l),200*(l-f),t.opacity)}function i(t,e,n,i){return 1===arguments.length?r(t):new o(t,e,n,null==i?1:i)}function o(t,e,n,r){this.l=+t,this.a=+e,this.b=+n,this.opacity=+r}function a(t){return t>C?Math.pow(t,1/3):t/w+b}function u(t){return t>x?t*t*t:w*(t-b)}function c(t){return 255*(t<=.0031308?12.92*t:1.055*Math.pow(t,1/2.4)-.055)}function s(t){return(t/=255)<=.04045?t/12.92:Math.pow((t+.055)/1.055,2.4)}function l(t){if(t instanceof p)return new p(t.h,t.c,t.l,t.opacity);t instanceof o||(t=r(t));var e=Math.atan2(t.b,t.a)*v.a;return new p(e<0?e+360:e,Math.sqrt(t.a*t.a+t.b*t.b),t.l,t.opacity)}function f(t,e,n,r){return 1===arguments.length?l(t):new p(t,e,n,null==r?1:r)}function p(t,e,n,r){this.h=+t,this.c=+e,this.l=+n,this.opacity=+r}var h=n(60),d=n(59),v=n(113);e.a=i,e.b=f;var g=18,m=.95047,y=1,_=1.08883,b=4/29,x=6/29,w=3*x*x,C=x*x*x;n.i(h.a)(o,i,n.i(h.b)(d.f,{brighter:function(t){return new o(this.l+g*(null==t?1:t),this.a,this.b,this.opacity)},darker:function(t){return new o(this.l-g*(null==t?1:t),this.a,this.b,this.opacity)},rgb:function(){var t=(this.l+16)/116,e=isNaN(this.a)?t:t+this.a/500,n=isNaN(this.b)?t:t-this.b/200;return t=y*u(t),e=m*u(e),n=_*u(n),new d.d(c(3.2404542*e-1.5371385*t-.4985314*n),c(-.969266*e+1.8760108*t+.041556*n),c(.0556434*e-.2040259*t+1.0572252*n),this.opacity)}})),n.i(h.a)(p,f,n.i(h.b)(d.f,{brighter:function(t){return new p(this.h,this.c,this.l+g*(null==t?1:t),this.opacity)},darker:function(t){return new p(this.h,this.c,this.l-g*(null==t?1:t),this.opacity)},rgb:function(){return r(this).rgb()}}))},function(t,e,n){"use strict";function r(t){return o=n.i(i.a)(t),a=o.format,u=o.formatPrefix,o}var i=n(117);n.d(e,"b",function(){return a}),n.d(e,"c",function(){
return u}),e.a=r;var o,a,u;r({decimal:".",thousands:",",grouping:[3],currency:["$",""]})},function(t,e,n){"use strict";e.a=function(t,e){t=t.toPrecision(e);t:for(var n,r=t.length,i=1,o=-1;i<r;++i)switch(t[i]){case".":o=n=i;break;case"0":0===o&&(o=i),n=i;break;case"e":break t;default:o>0&&(o=0)}return o>0?t.slice(0,o)+t.slice(n+1):t}},function(t,e,n){"use strict";e.a=function(t,e){return function(n,r){for(var i=n.length,o=[],a=0,u=t[0],c=0;i>0&&u>0&&(c+u+1>r&&(u=Math.max(1,r-c)),o.push(n.substring(i-=u,i+u)),!((c+=u+1)>r));)u=t[a=(a+1)%t.length];return o.reverse().join(e)}}},function(t,e,n){"use strict";var r=n(61);e.a=function(t,e){var i=n.i(r.a)(t,e);if(!i)return t+"";var o=i[0],a=i[1];return a<0?"0."+new Array(-a).join("0")+o:o.length>a+1?o.slice(0,a+1)+"."+o.slice(a+1):o+new Array(a-o.length+2).join("0")}},function(t,e,n){"use strict";var r=n(42);e.a=function(t){return Math.max(0,-n.i(r.a)(Math.abs(t)))}},function(t,e,n){"use strict";var r=n(42);e.a=function(t,e){return Math.max(0,3*Math.max(-8,Math.min(8,Math.floor(n.i(r.a)(e)/3)))-n.i(r.a)(Math.abs(t)))}},function(t,e,n){"use strict";var r=n(42);e.a=function(t,e){return t=Math.abs(t),e=Math.abs(e)-t,Math.max(0,n.i(r.a)(e)-n.i(r.a)(t))+1}},function(t,e,n){"use strict";function r(t){return function e(r){function a(e,a){var u=t((e=n.i(i.cubehelix)(e)).h,(a=n.i(i.cubehelix)(a)).h),c=n.i(o.a)(e.s,a.s),s=n.i(o.a)(e.l,a.l),l=n.i(o.a)(e.opacity,a.opacity);return function(t){return e.h=u(t),e.s=c(t),e.l=s(Math.pow(t,r)),e.opacity=l(t),e+""}}return r=+r,a.gamma=e,a}(1)}var i=n(10),o=n(32);n.d(e,"a",function(){return a});var a=(r(o.b),r(o.a))},function(t,e,n){"use strict";function r(t){return function(e,r){var a=t((e=n.i(i.hcl)(e)).h,(r=n.i(i.hcl)(r)).h),u=n.i(o.a)(e.c,r.c),c=n.i(o.a)(e.l,r.l),s=n.i(o.a)(e.opacity,r.opacity);return function(t){return e.h=a(t),e.c=u(t),e.l=c(t),e.opacity=s(t),e+""}}}var i=n(10),o=n(32);r(o.b),r(o.a)},function(t,e,n){"use strict";function r(t){return function(e,r){var a=t((e=n.i(i.hsl)(e)).h,(r=n.i(i.hsl)(r)).h),u=n.i(o.a)(e.s,r.s),c=n.i(o.a)(e.l,r.l),s=n.i(o.a)(e.opacity,r.opacity);return function(t){return e.h=a(t),e.s=u(t),e.l=c(t),e.opacity=s(t),e+""}}}var i=n(10),o=n(32);r(o.b),r(o.a)},function(t,e,n){"use strict";n(10),n(32)},function(t,e,n){"use strict"},function(t,e,n){"use strict";e.a=function(t,e){return t=+t,e-=t,function(n){return Math.round(t+e*n)}}},function(t,e,n){"use strict";n.d(e,"a",function(){return i});var r=180/Math.PI,i={translateX:0,translateY:0,rotate:0,skewX:0,scaleX:1,scaleY:1};e.b=function(t,e,n,i,o,a){var u,c,s;return(u=Math.sqrt(t*t+e*e))&&(t/=u,e/=u),(s=t*n+e*i)&&(n-=t*s,i-=e*s),(c=Math.sqrt(n*n+i*i))&&(n/=c,i/=c,s/=c),t*i<e*n&&(t=-t,e=-e,s=-s,u=-u),{translateX:o,translateY:a,rotate:Math.atan2(e,t)*r,skewX:Math.atan(s)*r,scaleX:u,scaleY:c}}},function(t,e,n){"use strict";function r(t,e,r,o){function a(t){return t.length?t.pop()+" ":""}function u(t,o,a,u,c,s){if(t!==a||o!==u){var l=c.push("translate(",null,e,null,r);s.push({i:l-4,x:n.i(i.a)(t,a)},{i:l-2,x:n.i(i.a)(o,u)})}else(a||u)&&c.push("translate("+a+e+u+r)}function c(t,e,r,u){t!==e?(t-e>180?e+=360:e-t>180&&(t+=360),u.push({i:r.push(a(r)+"rotate(",null,o)-2,x:n.i(i.a)(t,e)})):e&&r.push(a(r)+"rotate("+e+o)}function s(t,e,r,u){t!==e?u.push({i:r.push(a(r)+"skewX(",null,o)-2,x:n.i(i.a)(t,e)}):e&&r.push(a(r)+"skewX("+e+o)}function l(t,e,r,o,u,c){if(t!==r||e!==o){var s=u.push(a(u)+"scale(",null,",",null,")");c.push({i:s-4,x:n.i(i.a)(t,r)},{i:s-2,x:n.i(i.a)(e,o)})}else 1===r&&1===o||u.push(a(u)+"scale("+r+","+o+")")}return function(e,n){var r=[],i=[];return e=t(e),n=t(n),u(e.translateX,e.translateY,n.translateX,n.translateY,r,i),c(e.rotate,n.rotate,r,i),s(e.skewX,n.skewX,r,i),l(e.scaleX,e.scaleY,n.scaleX,n.scaleY,r,i),e=n=null,function(t){for(var e,n=-1,o=i.length;++n<o;)r[(e=i[n]).i]=e.x(t);return r.join("")}}}var i=n(43),o=n(226);r(o.a,"px, ","px)","deg)"),r(o.b,", ",")",")")},function(t,e,n){"use strict";function r(t){return"none"===t?o.a:(a||(a=document.createElement("DIV"),u=document.documentElement,c=document.defaultView),a.style.transform=t,t=c.getComputedStyle(u.appendChild(a),null).getPropertyValue("transform"),u.removeChild(a),t=t.slice(7,-1).split(","),n.i(o.b)(+t[0],+t[1],+t[2],+t[3],+t[4],+t[5]))}function i(t){return null==t?o.a:(s||(s=document.createElementNS("http://www.w3.org/2000/svg","g")),s.setAttribute("transform",t),(t=s.transform.baseVal.consolidate())?(t=t.matrix,n.i(o.b)(t.a,t.b,t.c,t.d,t.e,t.f)):o.a)}var o=n(224);e.a=r,e.b=i;var a,u,c,s},function(t,e,n){"use strict";Math.SQRT2},function(t,e,n){"use strict";function r(){this._x0=this._y0=this._x1=this._y1=null,this._=""}function i(){return new r}var o=Math.PI,a=2*o,u=1e-6,c=a-u;r.prototype=i.prototype={constructor:r,moveTo:function(t,e){this._+="M"+(this._x0=this._x1=+t)+","+(this._y0=this._y1=+e)},closePath:function(){null!==this._x1&&(this._x1=this._x0,this._y1=this._y0,this._+="Z")},lineTo:function(t,e){this._+="L"+(this._x1=+t)+","+(this._y1=+e)},quadraticCurveTo:function(t,e,n,r){this._+="Q"+ +t+","+ +e+","+(this._x1=+n)+","+(this._y1=+r)},bezierCurveTo:function(t,e,n,r,i,o){this._+="C"+ +t+","+ +e+","+ +n+","+ +r+","+(this._x1=+i)+","+(this._y1=+o)},arcTo:function(t,e,n,r,i){t=+t,e=+e,n=+n,r=+r,i=+i;var a=this._x1,c=this._y1,s=n-t,l=r-e,f=a-t,p=c-e,h=f*f+p*p;if(i<0)throw new Error("negative radius: "+i);if(null===this._x1)this._+="M"+(this._x1=t)+","+(this._y1=e);else if(h>u)if(Math.abs(p*s-l*f)>u&&i){var d=n-a,v=r-c,g=s*s+l*l,m=d*d+v*v,y=Math.sqrt(g),_=Math.sqrt(h),b=i*Math.tan((o-Math.acos((g+h-m)/(2*y*_)))/2),x=b/_,w=b/y;Math.abs(x-1)>u&&(this._+="L"+(t+x*f)+","+(e+x*p)),this._+="A"+i+","+i+",0,0,"+ +(p*d>f*v)+","+(this._x1=t+w*s)+","+(this._y1=e+w*l)}else this._+="L"+(this._x1=t)+","+(this._y1=e);else;},arc:function(t,e,n,r,i,s){t=+t,e=+e,n=+n;var l=n*Math.cos(r),f=n*Math.sin(r),p=t+l,h=e+f,d=1^s,v=s?r-i:i-r;if(n<0)throw new Error("negative radius: "+n);null===this._x1?this._+="M"+p+","+h:(Math.abs(this._x1-p)>u||Math.abs(this._y1-h)>u)&&(this._+="L"+p+","+h),n&&(v>c?this._+="A"+n+","+n+",0,1,"+d+","+(t-l)+","+(e-f)+"A"+n+","+n+",0,1,"+d+","+(this._x1=p)+","+(this._y1=h):(v<0&&(v=v%a+a),this._+="A"+n+","+n+",0,"+ +(v>=o)+","+d+","+(this._x1=t+n*Math.cos(i))+","+(this._y1=e+n*Math.sin(i))))},rect:function(t,e,n,r){this._+="M"+(this._x0=this._x1=+t)+","+(this._y0=this._y1=+e)+"h"+ +n+"v"+ +r+"h"+-n+"Z"},toString:function(){return this._}},e.a=i},function(t,e,n){"use strict";function r(){function t(){var t=c().length,r=l[1]<l[0],o=l[r-0],u=l[1-r];e=(u-o)/Math.max(1,t-p+2*h),f&&(e=Math.floor(e)),o+=(u-o-e*(t-p))*d,i=e*(1-p),f&&(o=Math.round(o),i=Math.round(i));var v=n.i(a.g)(t).map(function(t){return o+e*t});return s(r?v.reverse():v)}var e,i,o=n.i(u.a)().unknown(void 0),c=o.domain,s=o.range,l=[0,1],f=!1,p=0,h=0,d=.5;return delete o.unknown,o.domain=function(e){return arguments.length?(c(e),t()):c()},o.range=function(e){return arguments.length?(l=[+e[0],+e[1]],t()):l.slice()},o.rangeRound=function(e){return l=[+e[0],+e[1]],f=!0,t()},o.bandwidth=function(){return i},o.step=function(){return e},o.round=function(e){return arguments.length?(f=!!e,t()):f},o.padding=function(e){return arguments.length?(p=h=Math.max(0,Math.min(1,e)),t()):p},o.paddingInner=function(e){return arguments.length?(p=Math.max(0,Math.min(1,e)),t()):p},o.paddingOuter=function(e){return arguments.length?(h=Math.max(0,Math.min(1,e)),t()):h},o.align=function(e){return arguments.length?(d=Math.max(0,Math.min(1,e)),t()):d},o.copy=function(){return r().domain(c()).range(l).round(f).paddingInner(p).paddingOuter(h).align(d)},t()}function i(t){var e=t.copy;return t.padding=t.paddingOuter,delete t.paddingInner,delete t.paddingOuter,t.copy=function(){return i(e())},t}function o(){return i(r().paddingInner(1))}var a=n(12),u=n(127);e.a=r,e.b=o},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf")},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("1f77b4aec7e8ff7f0effbb782ca02c98df8ad62728ff98969467bdc5b0d58c564bc49c94e377c2f7b6d27f7f7fc7c7c7bcbd22dbdb8d17becf9edae5")},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("393b795254a36b6ecf9c9ede6379398ca252b5cf6bcedb9c8c6d31bd9e39e7ba52e7cb94843c39ad494ad6616be7969c7b4173a55194ce6dbdde9ed6")},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("3182bd6baed69ecae1c6dbefe6550dfd8d3cfdae6bfdd0a231a35474c476a1d99bc7e9c0756bb19e9ac8bcbddcdadaeb636363969696bdbdbdd9d9d9")},function(t,e,n){"use strict";var r=n(10),i=n(31);e.a=n.i(i.d)(n.i(r.cubehelix)(300,.5,0),n.i(r.cubehelix)(-240,.5,1))},function(t,e,n){"use strict";function r(){function t(t){return+t}var e=[0,1];return t.invert=t,t.domain=t.range=function(n){return arguments.length?(e=i.a.call(n,a.a),t):e.slice()},t.copy=function(){return r().domain(e)},n.i(o.b)(t)}var i=n(16),o=n(34),a=n(126);e.a=r},function(t,e,n){"use strict";function r(t,e){return(e=Math.log(e/t))?function(n){return Math.log(n/t)/e}:n.i(p.a)(e)}function i(t,e){return t<0?function(n){return-Math.pow(-e,n)*Math.pow(-t,1-n)}:function(n){return Math.pow(e,n)*Math.pow(t,1-n)}}function o(t){return isFinite(t)?+("1e"+t):t<0?0:t}function a(t){return 10===t?o:t===Math.E?Math.exp:function(e){return Math.pow(t,e)}}function u(t){return t===Math.E?Math.log:10===t&&Math.log10||2===t&&Math.log2||(t=Math.log(t),function(e){return Math.log(e)/t})}function c(t){return function(e){return-t(-e)}}function s(){function t(){return v=u(p),g=a(p),o()[0]<0&&(v=c(v),g=c(g)),e}var e=n.i(d.a)(r,i).domain([1,10]),o=e.domain,p=10,v=u(10),g=a(10);return e.base=function(e){return arguments.length?(p=+e,t()):p},e.domain=function(e){return arguments.length?(o(e),t()):o()},e.ticks=function(t){var e,r=o(),i=r[0],a=r[r.length-1];(e=a<i)&&(f=i,i=a,a=f);var u,c,s,f=v(i),h=v(a),d=null==t?10:+t,m=[];if(!(p%1)&&h-f<d){if(f=Math.round(f)-1,h=Math.round(h)+1,i>0){for(;f<h;++f)for(c=1,u=g(f);c<p;++c)if(s=u*c,!(s<i)){if(s>a)break;m.push(s)}}else for(;f<h;++f)for(c=p-1,u=g(f);c>=1;--c)if(s=u*c,!(s<i)){if(s>a)break;m.push(s)}}else m=n.i(l.a)(f,h,Math.min(h-f,d)).map(g);return e?m.reverse():m},e.tickFormat=function(t,r){if(null==r&&(r=10===p?".0e":","),"function"!=typeof r&&(r=n.i(f.format)(r)),t===1/0)return r;null==t&&(t=10);var i=Math.max(1,p*t/e.ticks().length);return function(t){var e=t/g(Math.round(v(t)));return e*p<p-.5&&(e*=p),e<=i?r(t):""}},e.nice=function(){return o(n.i(h.a)(o(),{floor:function(t){return g(Math.floor(v(t)))},ceil:function(t){return g(Math.ceil(v(t)))}}))},e.copy=function(){return n.i(d.c)(e,s().base(p))},e}var l=n(12),f=n(30),p=n(65),h=n(125),d=n(45);e.a=s},function(t,e,n){"use strict";function r(t,e){return t<0?-Math.pow(-t,e):Math.pow(t,e)}function i(){function t(t,e){return(e=r(e,o)-(t=r(t,o)))?function(n){return(r(n,o)-t)/e}:n.i(a.a)(e)}function e(t,e){return e=r(e,o)-(t=r(t,o)),function(n){return r(t+e*n,1/o)}}var o=1,s=n.i(c.a)(t,e),l=s.domain;return s.exponent=function(t){return arguments.length?(o=+t,l(l())):o},s.copy=function(){return n.i(c.c)(s,i().exponent(o))},n.i(u.b)(s)}function o(){return i().exponent(.5)}var a=n(65),u=n(34),c=n(45);e.a=i,e.b=o},function(t,e,n){"use strict";function r(){function t(){var t=0,r=Math.max(1,u.length);for(c=new Array(r-1);++t<r;)c[t-1]=n.i(i.e)(a,t/r);return e}function e(t){if(!isNaN(t=+t))return u[n.i(i.c)(c,t)]}var a=[],u=[],c=[];return e.invertExtent=function(t){var e=u.indexOf(t);return e<0?[NaN,NaN]:[e>0?c[e-1]:a[0],e<c.length?c[e]:a[a.length-1]]},e.domain=function(e){if(!arguments.length)return a.slice();a=[];for(var n,r=0,o=e.length;r<o;++r)n=e[r],null==n||isNaN(n=+n)||a.push(n);return a.sort(i.f),t()},e.range=function(e){return arguments.length?(u=o.b.call(e),t()):u.slice()},e.quantiles=function(){return c.slice()},e.copy=function(){return r().domain(a).range(u)},e}var i=n(12),o=n(16);e.a=r},function(t,e,n){"use strict";function r(){function t(t){if(t<=t)return f[n.i(i.c)(l,t,0,s)]}function e(){var e=-1;for(l=new Array(s);++e<s;)l[e]=((e+1)*c-(e-s)*u)/(s+1);return t}var u=0,c=1,s=1,l=[.5],f=[0,1];return t.domain=function(t){return arguments.length?(u=+t[0],c=+t[1],e()):[u,c]},t.range=function(t){return arguments.length?(s=(f=o.b.call(t)).length-1,e()):f.slice()},t.invertExtent=function(t){var e=f.indexOf(t);return e<0?[NaN,NaN]:e<1?[u,l[0]]:e>=s?[l[s-1],c]:[l[e-1],l[e]]},t.copy=function(){return r().domain([u,c]).range(f)},n.i(a.b)(t)}var i=n(12),o=n(16),a=n(34);e.a=r},function(t,e,n){"use strict";var r=n(10),i=n(31);n.d(e,"b",function(){return o}),n.d(e,"c",function(){return a});var o=n.i(i.d)(n.i(r.cubehelix)(-100,.75,.35),n.i(r.cubehelix)(80,1.5,.8)),a=n.i(i.d)(n.i(r.cubehelix)(260,.75,.35),n.i(r.cubehelix)(80,1.5,.8)),u=n.i(r.cubehelix)();e.a=function(t){(t<0||t>1)&&(t-=Math.floor(t));var e=Math.abs(t-.5);return u.h=360*t-100,u.s=1.5-1.5*e,u.l=.8-.9*e,u+""}},function(t,e,n){"use strict";function r(t){function e(e){var n=(e-o)/(a-o);return t(u?Math.max(0,Math.min(1,n)):n)}var o=0,a=1,u=!1;return e.domain=function(t){return arguments.length?(o=+t[0],a=+t[1],e):[o,a]},e.clamp=function(t){return arguments.length?(u=!!t,e):u},e.interpolator=function(n){return arguments.length?(t=n,e):t},e.copy=function(){return r(t).domain([o,a]).clamp(u)},n.i(i.b)(e)}var i=n(34);e.a=r},function(t,e,n){"use strict";function r(){function t(t){if(t<=t)return a[n.i(i.c)(e,t,0,u)]}var e=[.5],a=[0,1],u=1;return t.domain=function(n){return arguments.length?(e=o.b.call(n),u=Math.min(e.length,a.length-1),t):e.slice()},t.range=function(n){return arguments.length?(a=o.b.call(n),u=Math.min(e.length,a.length-1),t):a.slice()},t.invertExtent=function(t){var n=a.indexOf(t);return[e[n-1],e[n]]},t.copy=function(){return r().domain(e).range(a)},t}var i=n(12),o=n(16);e.a=r},function(t,e,n){"use strict";var r=n(12),i=n(30);e.a=function(t,e,o){var a,u=t[0],c=t[t.length-1],s=n.i(r.b)(u,c,null==e?10:e);switch(o=n.i(i.formatSpecifier)(null==o?",f":o),o.type){case"s":var l=Math.max(Math.abs(u),Math.abs(c));return null!=o.precision||isNaN(a=n.i(i.precisionPrefix)(s,l))||(o.precision=a),n.i(i.formatPrefix)(o,l);case"":case"e":case"g":case"p":case"r":null!=o.precision||isNaN(a=n.i(i.precisionRound)(s,Math.max(Math.abs(u),Math.abs(c))))||(o.precision=a-("e"===o.type));break;case"f":case"%":null!=o.precision||isNaN(a=n.i(i.precisionFixed)(s))||(o.precision=a-2*("%"===o.type))}return n.i(i.format)(o)}},function(t,e,n){"use strict";var r=n(128),i=n(77),o=n(79);e.a=function(){return n.i(r.b)(o.f,o.i,o.j,o.e,o.k,o.l,o.m,o.n,i.utcFormat).domain([Date.UTC(2e3,0,1),Date.UTC(2e3,0,2)])}},function(t,e,n){"use strict";function r(t){var e=t.length;return function(n){return t[Math.max(0,Math.min(e-1,Math.floor(n*e)))]}}var i=n(33);n.d(e,"b",function(){return o}),n.d(e,"c",function(){return a}),n.d(e,"d",function(){return u}),e.a=r(n.i(i.a)("44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725"));var o=r(n.i(i.a)("00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf")),a=r(n.i(i.a)("00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4")),u=r(n.i(i.a)("0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f059631059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a14c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a76600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a87e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a29511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586ada5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5de66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4ff0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921"))},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";function r(){return new i}function i(){this._="@"+(++o).toString(36)}e.a=r;var o=0;i.prototype=r.prototype={constructor:i,get:function(t){for(var e=this._;!(e in t);)if(!(t=t.parentNode))return;return t[e]},set:function(t,e){return t[this._]=e},remove:function(t){return this._ in t&&delete t[this._]},toString:function(){return this._}}},function(t,e,n){"use strict";var r=n(72),i=n(69);e.a=function(t){var e=n.i(r.a)();return e.changedTouches&&(e=e.changedTouches[0]),n.i(i.a)(t,e)}},function(t,e,n){"use strict";var r=n(7);e.a=function(t){return"string"==typeof t?new r.b([[document.querySelector(t)]],[document.documentElement]):new r.b([[t]],r.c)}},function(t,e,n){"use strict";var r=n(7);e.a=function(t){return"string"==typeof t?new r.b([document.querySelectorAll(t)],[document.documentElement]):new r.b([null==t?[]:t],r.c)}},function(t,e,n){"use strict";var r=n(66);e.a=function(t){var e="function"==typeof t?t:n.i(r.a)(t);return this.select(function(){return this.appendChild(e.apply(this,arguments))})}},function(t,e,n){"use strict";function r(t){return function(){this.removeAttribute(t)}}function i(t){return function(){this.removeAttributeNS(t.space,t.local)}}function o(t,e){return function(){this.setAttribute(t,e)}}function a(t,e){return function(){this.setAttributeNS(t.space,t.local,e)}}function u(t,e){return function(){var n=e.apply(this,arguments);null==n?this.removeAttribute(t):this.setAttribute(t,n)}}function c(t,e){return function(){var n=e.apply(this,arguments);null==n?this.removeAttributeNS(t.space,t.local):this.setAttributeNS(t.space,t.local,n)}}var s=n(67);e.a=function(t,e){var l=n.i(s.a)(t);if(arguments.length<2){var f=this.node();return l.local?f.getAttributeNS(l.space,l.local):f.getAttribute(l)}return this.each((null==e?l.local?i:r:"function"==typeof e?l.local?c:u:l.local?a:o)(l,e))}},function(t,e,n){"use strict";e.a=function(){var t=arguments[0];return arguments[0]=this,t.apply(null,arguments),this}},function(t,e,n){"use strict";function r(t){return t.trim().split(/^|\s+/)}function i(t){return t.classList||new o(t)}function o(t){this._node=t,this._names=r(t.getAttribute("class")||"")}function a(t,e){for(var n=i(t),r=-1,o=e.length;++r<o;)n.add(e[r])}function u(t,e){for(var n=i(t),r=-1,o=e.length;++r<o;)n.remove(e[r])}function c(t){return function(){a(this,t)}}function s(t){return function(){u(this,t)}}function l(t,e){return function(){(e.apply(this,arguments)?a:u)(this,t)}}o.prototype={add:function(t){var e=this._names.indexOf(t);e<0&&(this._names.push(t),this._node.setAttribute("class",this._names.join(" ")))},remove:function(t){var e=this._names.indexOf(t);e>=0&&(this._names.splice(e,1),this._node.setAttribute("class",this._names.join(" ")))},contains:function(t){return this._names.indexOf(t)>=0}},e.a=function(t,e){var n=r(t+"");if(arguments.length<2){for(var o=i(this.node()),a=-1,u=n.length;++a<u;)if(!o.contains(n[a]))return!1;return!0}return this.each(("function"==typeof e?l:e?c:s)(n,e))}},function(t,e,n){"use strict";function r(t,e,n,r,i,o){for(var u,c=0,s=e.length,l=o.length;c<l;++c)(u=e[c])?(u.__data__=o[c],r[c]=u):n[c]=new a.b(t,o[c]);for(;c<s;++c)(u=e[c])&&(i[c]=u)}function i(t,e,n,r,i,o,u){var s,l,f,p={},h=e.length,d=o.length,v=new Array(h);for(s=0;s<h;++s)(l=e[s])&&(v[s]=f=c+u.call(l,l.__data__,s,e),f in p?i[s]=l:p[f]=l);for(s=0;s<d;++s)f=c+u.call(t,o[s],s,o),(l=p[f])?(r[s]=l,l.__data__=o[s],p[f]=null):n[s]=new a.b(t,o[s]);for(s=0;s<h;++s)(l=e[s])&&p[v[s]]===l&&(i[s]=l)}var o=n(7),a=n(131),u=n(246),c="$";e.a=function(t,e){if(!t)return y=new Array(this.size()),d=-1,this.each(function(t){y[++d]=t}),y;var a=e?i:r,c=this._parents,s=this._groups;"function"!=typeof t&&(t=n.i(u.a)(t));for(var l=s.length,f=new Array(l),p=new Array(l),h=new Array(l),d=0;d<l;++d){var v=c[d],g=s[d],m=g.length,y=t.call(v,v&&v.__data__,d,c),_=y.length,b=p[d]=new Array(_),x=f[d]=new Array(_),w=h[d]=new Array(m);a(v,g,b,x,w,y,e);for(var C,M,k=0,E=0;k<_;++k)if(C=b[k]){for(k>=E&&(E=k+1);!(M=x[E])&&++E<_;);C._next=M||null}}return f=new o.b(f,c),f._enter=p,f._exit=h,f}},function(t,e,n){"use strict";e.a=function(t){return arguments.length?this.property("__data__",t):this.node().__data__}},function(t,e,n){"use strict";function r(t,e,r){var i=n.i(a.a)(t),o=i.CustomEvent;o?o=new o(e,r):(o=i.document.createEvent("Event"),r?(o.initEvent(e,r.bubbles,r.cancelable),o.detail=r.detail):o.initEvent(e,!1,!1)),t.dispatchEvent(o)}function i(t,e){return function(){return r(this,t,e)}}function o(t,e){return function(){return r(this,t,e.apply(this,arguments))}}var a=n(73);e.a=function(t,e){return this.each(("function"==typeof e?o:i)(t,e))}},function(t,e,n){"use strict";e.a=function(t){for(var e=this._groups,n=0,r=e.length;n<r;++n)for(var i,o=e[n],a=0,u=o.length;a<u;++a)(i=o[a])&&t.call(i,i.__data__,a,o);return this}},function(t,e,n){"use strict";e.a=function(){return!this.node()}},function(t,e,n){"use strict";var r=n(132),i=n(7);e.a=function(){return new i.b(this._exit||this._groups.map(r.a),this._parents)}},function(t,e,n){"use strict";var r=n(7),i=n(130);e.a=function(t){"function"!=typeof t&&(t=n.i(i.a)(t));for(var e=this._groups,o=e.length,a=new Array(o),u=0;u<o;++u)for(var c,s=e[u],l=s.length,f=a[u]=[],p=0;p<l;++p)(c=s[p])&&t.call(c,c.__data__,p,s)&&f.push(c);return new r.b(a,this._parents)}},function(t,e,n){"use strict";function r(){this.innerHTML=""}function i(t){return function(){this.innerHTML=t}}function o(t){return function(){var e=t.apply(this,arguments);this.innerHTML=null==e?"":e}}e.a=function(t){return arguments.length?this.each(null==t?r:("function"==typeof t?o:i)(t)):this.node().innerHTML}},function(t,e,n){"use strict";function r(){return null}var i=n(66),o=n(71);e.a=function(t,e){var a="function"==typeof t?t:n.i(i.a)(t),u=null==e?r:"function"==typeof e?e:n.i(o.a)(e);return this.select(function(){return this.insertBefore(a.apply(this,arguments),u.apply(this,arguments)||null)})}},function(t,e,n){"use strict";function r(){this.previousSibling&&this.parentNode.insertBefore(this,this.parentNode.firstChild)}e.a=function(){return this.each(r)}},function(t,e,n){"use strict";var r=n(7);e.a=function(t){for(var e=this._groups,n=t._groups,i=e.length,o=n.length,a=Math.min(i,o),u=new Array(i),c=0;c<a;++c)for(var s,l=e[c],f=n[c],p=l.length,h=u[c]=new Array(p),d=0;d<p;++d)(s=l[d]||f[d])&&(h[d]=s);for(;c<i;++c)u[c]=e[c];return new r.b(u,this._parents)}},function(t,e,n){"use strict";e.a=function(){for(var t=this._groups,e=0,n=t.length;e<n;++e)for(var r=t[e],i=0,o=r.length;i<o;++i){var a=r[i];if(a)return a}return null}},function(t,e,n){"use strict";e.a=function(){var t=new Array(this.size()),e=-1;return this.each(function(){t[++e]=this}),t}},function(t,e,n){"use strict";e.a=function(){for(var t=this._groups,e=-1,n=t.length;++e<n;)for(var r,i=t[e],o=i.length-1,a=i[o];--o>=0;)(r=i[o])&&(a&&a!==r.nextSibling&&a.parentNode.insertBefore(r,a),a=r);return this}},function(t,e,n){"use strict";function r(t){return function(){delete this[t]}}function i(t,e){return function(){this[t]=e}}function o(t,e){return function(){var n=e.apply(this,arguments);null==n?delete this[t]:this[t]=n}}e.a=function(t,e){return arguments.length>1?this.each((null==e?r:"function"==typeof e?o:i)(t,e)):this.node()[t]}},function(t,e,n){"use strict";function r(){this.nextSibling&&this.parentNode.appendChild(this)}e.a=function(){return this.each(r)}},function(t,e,n){"use strict";function r(){var t=this.parentNode;t&&t.removeChild(this)}e.a=function(){return this.each(r)}},function(t,e,n){"use strict";var r=n(7),i=n(71);e.a=function(t){"function"!=typeof t&&(t=n.i(i.a)(t));for(var e=this._groups,o=e.length,a=new Array(o),u=0;u<o;++u)for(var c,s,l=e[u],f=l.length,p=a[u]=new Array(f),h=0;h<f;++h)(c=l[h])&&(s=t.call(c,c.__data__,h,l))&&("__data__"in c&&(s.__data__=c.__data__),p[h]=s);return new r.b(a,this._parents)}},function(t,e,n){"use strict";var r=n(7),i=n(133);e.a=function(t){"function"!=typeof t&&(t=n.i(i.a)(t));for(var e=this._groups,o=e.length,a=[],u=[],c=0;c<o;++c)for(var s,l=e[c],f=l.length,p=0;p<f;++p)(s=l[p])&&(a.push(t.call(s,s.__data__,p,l)),u.push(s));return new r.b(a,u)}},function(t,e,n){"use strict";e.a=function(){var t=0;return this.each(function(){++t}),t}},function(t,e,n){"use strict";function r(t,e){return t<e?-1:t>e?1:t>=e?0:NaN}var i=n(7);e.a=function(t){function e(e,n){return e&&n?t(e.__data__,n.__data__):!e-!n}t||(t=r);for(var n=this._groups,o=n.length,a=new Array(o),u=0;u<o;++u){for(var c,s=n[u],l=s.length,f=a[u]=new Array(l),p=0;p<l;++p)(c=s[p])&&(f[p]=c);f.sort(e)}return new i.b(a,this._parents).order()}},function(t,e,n){"use strict";function r(t){return function(){this.style.removeProperty(t)}}function i(t,e,n){return function(){this.style.setProperty(t,e,n)}}function o(t,e,n){return function(){var r=e.apply(this,arguments);null==r?this.style.removeProperty(t):this.style.setProperty(t,r,n)}}var a=n(73);e.a=function(t,e,u){var c;return arguments.length>1?this.each((null==e?r:"function"==typeof e?o:i)(t,e,null==u?"":u)):n.i(a.a)(c=this.node()).getComputedStyle(c,null).getPropertyValue(t)}},function(t,e,n){"use strict";function r(){this.textContent=""}function i(t){return function(){this.textContent=t}}function o(t){return function(){var e=t.apply(this,arguments);this.textContent=null==e?"":e}}e.a=function(t){return arguments.length?this.each(null==t?r:("function"==typeof t?o:i)(t)):this.node().textContent}},function(t,e,n){"use strict";var r=n(72),i=n(69);e.a=function(t,e,o){arguments.length<3&&(o=e,e=n.i(r.a)().changedTouches);for(var a,u=0,c=e?e.length:0;u<c;++u)if((a=e[u]).identifier===o)return n.i(i.a)(t,a);return null}},function(t,e,n){"use strict";var r=n(72),i=n(69);e.a=function(t,e){null==e&&(e=n.i(r.a)().touches);for(var o=0,a=e?e.length:0,u=new Array(a);o<a;++o)u[o]=n.i(i.a)(t,e[o]);return u}},function(t,e,n){"use strict";function r(t){return t.innerRadius}function i(t){return t.outerRadius}function o(t){return t.startAngle}function a(t){return t.endAngle}function u(t){return t&&t.padAngle}function c(t){return t>=1?h.d:t<=-1?-h.d:Math.asin(t)}function s(t,e,n,r,i,o,a,u){var c=n-t,s=r-e,l=a-i,f=u-o,p=(l*(e-o)-f*(t-i))/(f*c-l*s);return[t+p*c,e+p*s]}function l(t,e,n,r,i,o,a){var u=t-n,c=e-r,s=(a?o:-o)/Math.sqrt(u*u+c*c),l=s*c,f=-s*u,p=t+l,h=e+f,d=n+l,v=r+f,g=(p+d)/2,m=(h+v)/2,y=d-p,_=v-h,b=y*y+_*_,x=i-o,w=p*v-d*h,C=(_<0?-1:1)*Math.sqrt(Math.max(0,x*x*b-w*w)),M=(w*_-y*C)/b,k=(-w*y-_*C)/b,E=(w*_+y*C)/b,T=(-w*y+_*C)/b,S=M-g,P=k-m,N=E-g,A=T-m;return S*S+P*P>N*N+A*A&&(M=E,k=T),{cx:M,cy:k,x01:-l,y01:-f,x11:M*(i/x-1),y11:k*(i/x-1)}}var f=n(44),p=n(19),h=n(35);e.a=function(){function t(){var t,r,i=+e.apply(this,arguments),o=+d.apply(this,arguments),a=m.apply(this,arguments)-h.d,u=y.apply(this,arguments)-h.d,p=Math.abs(u-a),x=u>a;if(b||(b=t=n.i(f.a)()),o<i&&(r=o,o=i,i=r),o>h.a)if(p>h.c-h.a)b.moveTo(o*Math.cos(a),o*Math.sin(a)),b.arc(0,0,o,a,u,!x),i>h.a&&(b.moveTo(i*Math.cos(u),i*Math.sin(u)),b.arc(0,0,i,u,a,x));else{var w,C,M=a,k=u,E=a,T=u,S=p,P=p,N=_.apply(this,arguments)/2,A=N>h.a&&(g?+g.apply(this,arguments):Math.sqrt(i*i+o*o)),O=Math.min(Math.abs(o-i)/2,+v.apply(this,arguments)),I=O,D=O;
if(A>h.a){var R=c(A/i*Math.sin(N)),L=c(A/o*Math.sin(N));(S-=2*R)>h.a?(R*=x?1:-1,E+=R,T-=R):(S=0,E=T=(a+u)/2),(P-=2*L)>h.a?(L*=x?1:-1,M+=L,k-=L):(P=0,M=k=(a+u)/2)}var U=o*Math.cos(M),F=o*Math.sin(M),j=i*Math.cos(T),B=i*Math.sin(T);if(O>h.a){var W=o*Math.cos(k),V=o*Math.sin(k),z=i*Math.cos(E),H=i*Math.sin(E);if(p<h.b){var q=S>h.a?s(U,F,z,H,W,V,j,B):[j,B],Y=U-q[0],K=F-q[1],G=W-q[0],$=V-q[1],X=1/Math.sin(Math.acos((Y*G+K*$)/(Math.sqrt(Y*Y+K*K)*Math.sqrt(G*G+$*$)))/2),Z=Math.sqrt(q[0]*q[0]+q[1]*q[1]);I=Math.min(O,(i-Z)/(X-1)),D=Math.min(O,(o-Z)/(X+1))}}P>h.a?D>h.a?(w=l(z,H,U,F,o,D,x),C=l(W,V,j,B,o,D,x),b.moveTo(w.cx+w.x01,w.cy+w.y01),D<O?b.arc(w.cx,w.cy,D,Math.atan2(w.y01,w.x01),Math.atan2(C.y01,C.x01),!x):(b.arc(w.cx,w.cy,D,Math.atan2(w.y01,w.x01),Math.atan2(w.y11,w.x11),!x),b.arc(0,0,o,Math.atan2(w.cy+w.y11,w.cx+w.x11),Math.atan2(C.cy+C.y11,C.cx+C.x11),!x),b.arc(C.cx,C.cy,D,Math.atan2(C.y11,C.x11),Math.atan2(C.y01,C.x01),!x))):(b.moveTo(U,F),b.arc(0,0,o,M,k,!x)):b.moveTo(U,F),i>h.a&&S>h.a?I>h.a?(w=l(j,B,W,V,i,-I,x),C=l(U,F,z,H,i,-I,x),b.lineTo(w.cx+w.x01,w.cy+w.y01),I<O?b.arc(w.cx,w.cy,I,Math.atan2(w.y01,w.x01),Math.atan2(C.y01,C.x01),!x):(b.arc(w.cx,w.cy,I,Math.atan2(w.y01,w.x01),Math.atan2(w.y11,w.x11),!x),b.arc(0,0,i,Math.atan2(w.cy+w.y11,w.cx+w.x11),Math.atan2(C.cy+C.y11,C.cx+C.x11),x),b.arc(C.cx,C.cy,I,Math.atan2(C.y11,C.x11),Math.atan2(C.y01,C.x01),!x))):b.arc(0,0,i,T,E,x):b.lineTo(j,B)}else b.moveTo(0,0);if(b.closePath(),t)return b=null,t+""||null}var e=r,d=i,v=n.i(p.a)(0),g=null,m=o,y=a,_=u,b=null;return t.centroid=function(){var t=(+e.apply(this,arguments)+ +d.apply(this,arguments))/2,n=(+m.apply(this,arguments)+ +y.apply(this,arguments))/2-h.b/2;return[Math.cos(n)*t,Math.sin(n)*t]},t.innerRadius=function(r){return arguments.length?(e="function"==typeof r?r:n.i(p.a)(+r),t):e},t.outerRadius=function(e){return arguments.length?(d="function"==typeof e?e:n.i(p.a)(+e),t):d},t.cornerRadius=function(e){return arguments.length?(v="function"==typeof e?e:n.i(p.a)(+e),t):v},t.padRadius=function(e){return arguments.length?(g=null==e?null:"function"==typeof e?e:n.i(p.a)(+e),t):g},t.startAngle=function(e){return arguments.length?(m="function"==typeof e?e:n.i(p.a)(+e),t):m},t.endAngle=function(e){return arguments.length?(y="function"==typeof e?e:n.i(p.a)(+e),t):y},t.padAngle=function(e){return arguments.length?(_="function"==typeof e?e:n.i(p.a)(+e),t):_},t.context=function(e){return arguments.length?(b=null==e?null:e,t):b},t}},function(t,e,n){"use strict";n.d(e,"a",function(){return r});var r=Array.prototype.slice},function(t,e,n){"use strict";function r(t){this._context=t}var i=n(49),o=n(46);r.prototype={areaStart:i.a,areaEnd:i.a,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._y0=this._y1=this._y2=this._y3=this._y4=NaN,this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x2,this._y2),this._context.closePath();break;case 2:this._context.moveTo((this._x2+2*this._x3)/3,(this._y2+2*this._y3)/3),this._context.lineTo((this._x3+2*this._x2)/3,(this._y3+2*this._y2)/3),this._context.closePath();break;case 3:this.point(this._x2,this._y2),this.point(this._x3,this._y3),this.point(this._x4,this._y4)}},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._x2=t,this._y2=e;break;case 1:this._point=2,this._x3=t,this._y3=e;break;case 2:this._point=3,this._x4=t,this._y4=e,this._context.moveTo((this._x0+4*this._x1+t)/6,(this._y0+4*this._y1+e)/6);break;default:n.i(o.c)(this,t,e)}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t){this._context=t}var i=n(46);r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN,this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3;var r=(this._x0+4*this._x1+t)/6,o=(this._y0+4*this._y1+e)/6;this._line?this._context.lineTo(r,o):this._context.moveTo(r,o);break;case 3:this._point=4;default:n.i(i.c)(this,t,e)}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t,e){this._basis=new i.b(t),this._beta=e}var i=n(46);r.prototype={lineStart:function(){this._x=[],this._y=[],this._basis.lineStart()},lineEnd:function(){var t=this._x,e=this._y,n=t.length-1;if(n>0)for(var r,i=t[0],o=e[0],a=t[n]-i,u=e[n]-o,c=-1;++c<=n;)r=c/n,this._basis.point(this._beta*t[c]+(1-this._beta)*(i+r*a),this._beta*e[c]+(1-this._beta)*(o+r*u));this._x=this._y=null,this._basis.lineEnd()},point:function(t,e){this._x.push(+t),this._y.push(+e)}},e.a=function t(e){function n(t){return 1===e?new i.b(t):new r(t,e)}return n.beta=function(e){return t(+e)},n}(.85)},function(t,e,n){"use strict";function r(t,e){this._context=t,this._alpha=e}var i=n(136),o=n(49),a=n(74);r.prototype={areaStart:o.a,areaEnd:o.a,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN,this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x3,this._y3),this._context.closePath();break;case 2:this._context.lineTo(this._x3,this._y3),this._context.closePath();break;case 3:this.point(this._x3,this._y3),this.point(this._x4,this._y4),this.point(this._x5,this._y5)}},point:function(t,e){if(t=+t,e=+e,this._point){var r=this._x2-t,i=this._y2-e;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(r*r+i*i,this._alpha))}switch(this._point){case 0:this._point=1,this._x3=t,this._y3=e;break;case 1:this._point=2,this._context.moveTo(this._x4=t,this._y4=e);break;case 2:this._point=3,this._x5=t,this._y5=e;break;default:n.i(a.b)(this,t,e)}this._l01_a=this._l12_a,this._l12_a=this._l23_a,this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a,this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return e?new r(t,e):new i.b(t,0)}return n.alpha=function(e){return t(+e)},n}(.5)},function(t,e,n){"use strict";function r(t,e){this._context=t,this._alpha=e}var i=n(137),o=n(74);r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){if(t=+t,e=+e,this._point){var r=this._x2-t,i=this._y2-e;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(r*r+i*i,this._alpha))}switch(this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3,this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;default:n.i(o.b)(this,t,e)}this._l01_a=this._l12_a,this._l12_a=this._l23_a,this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a,this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return e?new r(t,e):new i.b(t,0)}return n.alpha=function(e){return t(+e)},n}(.5)},function(t,e,n){"use strict";function r(t){this._context=t}var i=n(49);r.prototype={areaStart:i.a,areaEnd:i.a,lineStart:function(){this._point=0},lineEnd:function(){this._point&&this._context.closePath()},point:function(t,e){t=+t,e=+e,this._point?this._context.lineTo(t,e):(this._point=1,this._context.moveTo(t,e))}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t){return t<0?-1:1}function i(t,e,n){var i=t._x1-t._x0,o=e-t._x1,a=(t._y1-t._y0)/(i||o<0&&-0),u=(n-t._y1)/(o||i<0&&-0),c=(a*o+u*i)/(i+o);return(r(a)+r(u))*Math.min(Math.abs(a),Math.abs(u),.5*Math.abs(c))||0}function o(t,e){var n=t._x1-t._x0;return n?(3*(t._y1-t._y0)/n-e)/2:e}function a(t,e,n){var r=t._x0,i=t._y0,o=t._x1,a=t._y1,u=(o-r)/3;t._context.bezierCurveTo(r+u,i+u*e,o-u,a-u*n,o,a)}function u(t){this._context=t}function c(t){this._context=new s(t)}function s(t){this._context=t}function l(t){return new u(t)}function f(t){return new c(t)}e.a=l,e.b=f,u.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=this._t0=NaN,this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x1,this._y1);break;case 3:a(this,this._t0,o(this,this._t0))}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){var n=NaN;if(t=+t,e=+e,t!==this._x1||e!==this._y1){switch(this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3,a(this,o(this,n=i(this,t,e)),n);break;default:a(this,this._t0,n=i(this,t,e))}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e,this._t0=n}}},(c.prototype=Object.create(u.prototype)).point=function(t,e){u.prototype.point.call(this,e,t)},s.prototype={moveTo:function(t,e){this._context.moveTo(e,t)},closePath:function(){this._context.closePath()},lineTo:function(t,e){this._context.lineTo(e,t)},bezierCurveTo:function(t,e,n,r,i,o){this._context.bezierCurveTo(e,t,r,n,o,i)}}},function(t,e,n){"use strict";function r(t){this._context=t}function i(t){var e,n,r=t.length-1,i=new Array(r),o=new Array(r),a=new Array(r);for(i[0]=0,o[0]=2,a[0]=t[0]+2*t[1],e=1;e<r-1;++e)i[e]=1,o[e]=4,a[e]=4*t[e]+2*t[e+1];for(i[r-1]=2,o[r-1]=7,a[r-1]=8*t[r-1]+t[r],e=1;e<r;++e)n=i[e]/o[e-1],o[e]-=n,a[e]-=n*a[e-1];for(i[r-1]=a[r-1]/o[r-1],e=r-2;e>=0;--e)i[e]=(a[e]-i[e+1])/o[e];for(o[r-1]=(t[r]+i[r-1])/2,e=0;e<r-1;++e)o[e]=2*t[e+1]-i[e+1];return[i,o]}r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=[],this._y=[]},lineEnd:function(){var t=this._x,e=this._y,n=t.length;if(n)if(this._line?this._context.lineTo(t[0],e[0]):this._context.moveTo(t[0],e[0]),2===n)this._context.lineTo(t[1],e[1]);else for(var r=i(t),o=i(e),a=0,u=1;u<n;++a,++u)this._context.bezierCurveTo(r[0][a],o[0][a],r[1][a],o[1][a],t[u],e[u]);(this._line||0!==this._line&&1===n)&&this._context.closePath(),this._line=1-this._line,this._x=this._y=null},point:function(t,e){this._x.push(+t),this._y.push(+e)}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t,e){this._context=t,this._t=e}function i(t){return new r(t,0)}function o(t){return new r(t,1)}e.c=i,e.b=o,r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=this._y=NaN,this._point=0},lineEnd:function(){0<this._t&&this._t<1&&2===this._point&&this._context.lineTo(this._x,this._y),(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line>=0&&(this._t=1-this._t,this._line=1-this._line)},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;default:if(this._t<=0)this._context.lineTo(this._x,e),this._context.lineTo(t,e);else{var n=this._x*(1-this._t)+t*this._t;this._context.lineTo(n,this._y),this._context.lineTo(n,e)}}this._x=t,this._y=e}},e.a=function(t){return new r(t,.5)}},function(t,e,n){"use strict";e.a=function(t,e){return e<t?-1:e>t?1:e>=t?0:NaN}},function(t,e,n){"use strict";e.a=function(t){return t}},function(t,e,n){"use strict";var r=n(36);e.a=function(t,e){if((o=t.length)>0){for(var i,o,a,u=0,c=t[0].length;u<c;++u){for(a=i=0;i<o;++i)a+=t[i][u][1]||0;if(a)for(i=0;i<o;++i)t[i][u][1]/=a}n.i(r.a)(t,e)}}},function(t,e,n){"use strict";var r=n(36);e.a=function(t,e){if((i=t.length)>0){for(var i,o=0,a=t[e[0]],u=a.length;o<u;++o){for(var c=0,s=0;c<i;++c)s+=t[c][o][1]||0;a[o][1]+=a[o][0]=-s/2}n.i(r.a)(t,e)}}},function(t,e,n){"use strict";var r=n(36);e.a=function(t,e){if((a=t.length)>0&&(o=(i=t[e[0]]).length)>0){for(var i,o,a,u=0,c=1;c<o;++c){for(var s=0,l=0,f=0;s<a;++s){for(var p=t[e[s]],h=p[c][1]||0,d=p[c-1][1]||0,v=(h-d)/2,g=0;g<s;++g){var m=t[e[g]],y=m[c][1]||0,_=m[c-1][1]||0;v+=y-_}l+=h,f+=v*h}i[c-1][1]+=i[c-1][0]=u,l&&(u-=f/l)}i[c-1][1]+=i[c-1][0]=u,n.i(r.a)(t,e)}}},function(t,e,n){"use strict";var r=n(76);e.a=function(t){return n.i(r.a)(t).reverse()}},function(t,e,n){"use strict";var r=n(37),i=n(76);e.a=function(t){var e,o,a=t.length,u=t.map(i.b),c=n.i(r.a)(t).sort(function(t,e){return u[e]-u[t]}),s=0,l=0,f=[],p=[];for(e=0;e<a;++e)o=c[e],s<l?(s+=u[o],f.push(o)):(l+=u[o],p.push(o));return p.reverse().concat(f)}},function(t,e,n){"use strict";var r=n(37);e.a=function(t){return n.i(r.a)(t).reverse()}},function(t,e,n){"use strict";var r=n(19),i=n(291),o=n(292),a=n(35);e.a=function(){function t(t){var n,r,i,o,p,h=t.length,d=0,v=new Array(h),g=new Array(h),m=+s.apply(this,arguments),y=Math.min(a.c,Math.max(-a.c,l.apply(this,arguments)-m)),_=Math.min(Math.abs(y)/h,f.apply(this,arguments)),b=_*(y<0?-1:1);for(n=0;n<h;++n)(p=g[v[n]=n]=+e(t[n],n,t))>0&&(d+=p);for(null!=u?v.sort(function(t,e){return u(g[t],g[e])}):null!=c&&v.sort(function(e,n){return c(t[e],t[n])}),n=0,i=d?(y-h*b)/d:0;n<h;++n,m=o)r=v[n],p=g[r],o=m+(p>0?p*i:0)+b,g[r]={data:t[r],index:n,value:p,startAngle:m,endAngle:o,padAngle:_};return g}var e=o.a,u=i.a,c=null,s=n.i(r.a)(0),l=n.i(r.a)(a.c),f=n.i(r.a)(0);return t.value=function(i){return arguments.length?(e="function"==typeof i?i:n.i(r.a)(+i),t):e},t.sortValues=function(e){return arguments.length?(u=e,c=null,t):u},t.sort=function(e){return arguments.length?(c=e,u=null,t):c},t.startAngle=function(e){return arguments.length?(s="function"==typeof e?e:n.i(r.a)(+e),t):s},t.endAngle=function(e){return arguments.length?(l="function"==typeof e?e:n.i(r.a)(+e),t):l},t.padAngle=function(e){return arguments.length?(f="function"==typeof e?e:n.i(r.a)(+e),t):f},t}},function(t,e,n){"use strict";var r=n(138),i=n(135),o=n(140);e.a=function(){var t=n.i(i.a)().curve(r.b),e=t.curve,a=t.lineX0,u=t.lineX1,c=t.lineY0,s=t.lineY1;return t.angle=t.x,delete t.x,t.startAngle=t.x0,delete t.x0,t.endAngle=t.x1,delete t.x1,t.radius=t.y,delete t.y,t.innerRadius=t.y0,delete t.y0,t.outerRadius=t.y1,delete t.y1,t.lineStartAngle=function(){return n.i(o.b)(a())},delete t.lineX0,t.lineEndAngle=function(){return n.i(o.b)(u())},delete t.lineX1,t.lineInnerRadius=function(){return n.i(o.b)(c())},delete t.lineY0,t.lineOuterRadius=function(){return n.i(o.b)(s())},delete t.lineY1,t.curve=function(t){return arguments.length?e(n.i(r.a)(t)):e()._curve},t}},function(t,e,n){"use strict";function r(t,e){return t[e]}var i=n(281),o=n(19),a=n(36),u=n(37);e.a=function(){function t(t){var n,r,i=e.apply(this,arguments),o=t.length,a=i.length,u=new Array(a);for(n=0;n<a;++n){for(var f,p=i[n],h=u[n]=new Array(o),d=0;d<o;++d)h[d]=f=[0,+l(t[d],p,d,t)],f.data=t[d];h.key=p}for(n=0,r=c(u);n<a;++n)u[r[n]].index=n;return s(u,r),u}var e=n.i(o.a)([]),c=u.a,s=a.a,l=r;return t.keys=function(r){return arguments.length?(e="function"==typeof r?r:n.i(o.a)(i.a.call(r)),t):e},t.value=function(e){return arguments.length?(l="function"==typeof e?e:n.i(o.a)(+e),t):l},t.order=function(e){return arguments.length?(c=null==e?u.a:"function"==typeof e?e:n.i(o.a)(i.a.call(e)),t):c},t.offset=function(e){return arguments.length?(s=null==e?a.a:e,t):s},t}},function(t,e,n){"use strict";var r=n(44),i=n(141),o=n(142),a=n(143),u=n(145),c=n(144),s=n(146),l=n(147),f=n(19);n.d(e,"b",function(){return p});var p=[i.a,o.a,a.a,c.a,u.a,s.a,l.a];e.a=function(){function t(){var t;if(a||(a=t=n.i(r.a)()),e.apply(this,arguments).draw(a,+o.apply(this,arguments)),t)return a=null,t+""||null}var e=n.i(f.a)(i.a),o=n.i(f.a)(64),a=null;return t.type=function(r){return arguments.length?(e="function"==typeof r?r:n.i(f.a)(r),t):e},t.size=function(e){return arguments.length?(o="function"==typeof e?e:n.i(f.a)(+e),t):o},t.context=function(e){return arguments.length?(a=null==e?null:e,t):a},t}},function(t,e,n){"use strict";function r(t){var e=new Date(t);return isNaN(e)?null:e}var i=n(148),o=n(78),a=+new Date("2000-01-01T00:00:00.000Z")?r:n.i(o.e)(i.b);e.a=a},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setHours(0,0,0,0)},function(t,e){t.setDate(t.getDate()+e)},function(t,e){return(e-t-(e.getTimezoneOffset()-t.getTimezoneOffset())*i.d)/i.b},function(t){return t.getDate()-1});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){var e=t.getTimezoneOffset()*i.d%i.c;e<0&&(e+=i.c),t.setTime(Math.floor((+t-e)/i.c)*i.c+e)},function(t,e){t.setTime(+t+e*i.c)},function(t,e){return(e-t)/i.c},function(t){return t.getHours()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(){},function(t,e){t.setTime(+t+e)},function(t,e){return e-t});i.every=function(t){return t=Math.floor(t),isFinite(t)&&t>0?t>1?n.i(r.a)(function(e){e.setTime(Math.floor(e/t)*t)},function(e,n){e.setTime(+e+n*t)},function(e,n){return(n-e)/t}):i:null},e.a=i;i.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setTime(Math.floor(t/i.d)*i.d)},function(t,e){t.setTime(+t+e*i.d)},function(t,e){return(e-t)/i.d},function(t){return t.getMinutes()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setDate(1),t.setHours(0,0,0,0)},function(t,e){t.setMonth(t.getMonth()+e)},function(t,e){return e.getMonth()-t.getMonth()+12*(e.getFullYear()-t.getFullYear())},function(t){return t.getMonth()});e.a=i;i.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setTime(Math.floor(t/i.e)*i.e)},function(t,e){t.setTime(+t+e*i.e)},function(t,e){return(e-t)/i.e},function(t){return t.getUTCSeconds()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setUTCHours(0,0,0,0)},function(t,e){t.setUTCDate(t.getUTCDate()+e)},function(t,e){return(e-t)/i.b},function(t){return t.getUTCDate()-1});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setUTCMinutes(0,0,0)},function(t,e){t.setTime(+t+e*i.c)},function(t,e){return(e-t)/i.c},function(t){return t.getUTCHours()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setUTCSeconds(0,0)},function(t,e){t.setTime(+t+e*i.d)},function(t,e){return(e-t)/i.d},function(t){return t.getUTCMinutes()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setUTCDate(1),t.setUTCHours(0,0,0,0)},function(t,e){t.setUTCMonth(t.getUTCMonth()+e)},function(t,e){return e.getUTCMonth()-t.getUTCMonth()+12*(e.getUTCFullYear()-t.getUTCFullYear())},function(t){return t.getUTCMonth()});e.a=i;i.range},function(t,e,n){"use strict";function r(t){return n.i(i.a)(function(e){e.setUTCDate(e.getUTCDate()-(e.getUTCDay()+7-t)%7),e.setUTCHours(0,0,0,0)},function(t,e){t.setUTCDate(t.getUTCDate()+7*e)},function(t,e){return(e-t)/o.a})}var i=n(5),o=n(13);n.d(e,"a",function(){return a}),n.d(e,"b",function(){return u});var a=r(0),u=r(1),c=r(2),s=r(3),l=r(4),f=r(5),p=r(6);a.range,u.range,c.range,s.range,l.range,f.range,p.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setUTCMonth(0,1),t.setUTCHours(0,0,0,0)},function(t,e){t.setUTCFullYear(t.getUTCFullYear()+e)},function(t,e){return e.getUTCFullYear()-t.getUTCFullYear()},function(t){return t.getUTCFullYear()});i.every=function(t){return isFinite(t=Math.floor(t))&&t>0?n.i(r.a)(function(e){e.setUTCFullYear(Math.floor(e.getUTCFullYear()/t)*t),e.setUTCMonth(0,1),e.setUTCHours(0,0,0,0)},function(e,n){e.setUTCFullYear(e.getUTCFullYear()+n*t)}):null},e.a=i;i.range},function(t,e,n){"use strict";function r(t){return n.i(i.a)(function(e){e.setDate(e.getDate()-(e.getDay()+7-t)%7),e.setHours(0,0,0,0)},function(t,e){t.setDate(t.getDate()+7*e)},function(t,e){return(e-t-(e.getTimezoneOffset()-t.getTimezoneOffset())*o.d)/o.a})}var i=n(5),o=n(13);n.d(e,"a",function(){return a}),n.d(e,"b",function(){return u});var a=r(0),u=r(1),c=r(2),s=r(3),l=r(4),f=r(5),p=r(6);a.range,u.range,c.range,s.range,l.range,f.range,p.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setMonth(0,1),t.setHours(0,0,0,0)},function(t,e){t.setFullYear(t.getFullYear()+e)},function(t,e){return e.getFullYear()-t.getFullYear()},function(t){return t.getFullYear()});i.every=function(t){return isFinite(t=Math.floor(t))&&t>0?n.i(r.a)(function(e){e.setFullYear(Math.floor(e.getFullYear()/t)*t),e.setMonth(0,1),e.setHours(0,0,0,0)},function(e,n){e.setFullYear(e.getFullYear()+n*t)}):null},e.a=i;i.range},function(t,e,n){"use strict";function r(t){return t.replace(i,function(t,e){return e.toUpperCase()})}var i=/-(.)/g;t.exports=r},function(t,e,n){"use strict";function r(t){return i(t.replace(o,"ms-"))}var i=n(318),o=/^-ms-/;t.exports=r},function(t,e,n){"use strict";function r(t,e){return!(!t||!e)&&(t===e||!i(t)&&(i(e)?r(t,e.parentNode):"contains"in t?t.contains(e):!!t.compareDocumentPosition&&!!(16&t.compareDocumentPosition(e))))}var i=n(328);t.exports=r},function(t,e,n){"use strict";function r(t){var e=t.length;if(Array.isArray(t)||"object"!=typeof t&&"function"!=typeof t?a(!1):void 0,"number"!=typeof e?a(!1):void 0,0===e||e-1 in t?void 0:a(!1),"function"==typeof t.callee?a(!1):void 0,t.hasOwnProperty)try{return Array.prototype.slice.call(t)}catch(t){}for(var n=Array(e),r=0;r<e;r++)n[r]=t[r];return n}function i(t){return!!t&&("object"==typeof t||"function"==typeof t)&&"length"in t&&!("setInterval"in t)&&"number"!=typeof t.nodeType&&(Array.isArray(t)||"callee"in t||"item"in t)}function o(t){return i(t)?Array.isArray(t)?t.slice():r(t):[t]}var a=n(0);t.exports=o},function(t,e,n){"use strict";function r(t){var e=t.match(l);return e&&e[1].toLowerCase()}function i(t,e){var n=s;s?void 0:c(!1);var i=r(t),o=i&&u(i);if(o){n.innerHTML=o[1]+t+o[2];for(var l=o[0];l--;)n=n.lastChild}else n.innerHTML=t;var f=n.getElementsByTagName("script");f.length&&(e?void 0:c(!1),a(f).forEach(e));for(var p=Array.from(n.childNodes);n.lastChild;)n.removeChild(n.lastChild);return p}var o=n(6),a=n(321),u=n(323),c=n(0),s=o.canUseDOM?document.createElement("div"):null,l=/^\s*<(\w+)/;t.exports=i},function(t,e,n){"use strict";function r(t){return a?void 0:o(!1),p.hasOwnProperty(t)||(t="*"),u.hasOwnProperty(t)||("*"===t?a.innerHTML="<link />":a.innerHTML="<"+t+"></"+t+">",u[t]=!a.firstChild),u[t]?p[t]:null}var i=n(6),o=n(0),a=i.canUseDOM?document.createElement("div"):null,u={},c=[1,'<select multiple="true">',"</select>"],s=[1,"<table>","</table>"],l=[3,"<table><tbody><tr>","</tr></tbody></table>"],f=[1,'<svg xmlns="http://www.w3.org/2000/svg">',"</svg>"],p={"*":[1,"?<div>","</div>"],area:[1,"<map>","</map>"],col:[2,"<table><tbody></tbody><colgroup>","</colgroup></table>"],legend:[1,"<fieldset>","</fieldset>"],param:[1,"<object>","</object>"],tr:[2,"<table><tbody>","</tbody></table>"],optgroup:c,option:c,caption:s,colgroup:s,tbody:s,tfoot:s,thead:s,td:l,th:l},h=["circle","clipPath","defs","ellipse","g","image","line","linearGradient","mask","path","pattern","polygon","polyline","radialGradient","rect","stop","text","tspan"];h.forEach(function(t){p[t]=f,u[t]=!0}),t.exports=r},function(t,e,n){"use strict";function r(t){return t===window?{x:window.pageXOffset||document.documentElement.scrollLeft,y:window.pageYOffset||document.documentElement.scrollTop}:{x:t.scrollLeft,y:t.scrollTop}}t.exports=r},function(t,e,n){"use strict";function r(t){return t.replace(i,"-$1").toLowerCase()}var i=/([A-Z])/g;t.exports=r},function(t,e,n){"use strict";function r(t){return i(t).replace(o,"-ms-")}var i=n(325),o=/^ms-/;t.exports=r},function(t,e,n){"use strict";function r(t){return!(!t||!("function"==typeof Node?t instanceof Node:"object"==typeof t&&"number"==typeof t.nodeType&&"string"==typeof t.nodeName))}t.exports=r},function(t,e,n){"use strict";function r(t){return i(t)&&3==t.nodeType}var i=n(327);t.exports=r},function(t,e,n){"use strict";var r=function(t){var e;for(e in t)if(t.hasOwnProperty(e))return e;return null};t.exports=r},function(t,e,n){"use strict";function r(t){var e={};return function(n){return e.hasOwnProperty(n)||(e[n]=t.call(this,n)),e[n]}}t.exports=r},function(t,e,n){"use strict";var r={Properties:{"aria-current":0,"aria-details":0,"aria-disabled":0,"aria-hidden":0,"aria-invalid":0,"aria-keyshortcuts":0,"aria-label":0,"aria-roledescription":0,"aria-autocomplete":0,"aria-checked":0,"aria-expanded":0,"aria-haspopup":0,"aria-level":0,"aria-modal":0,"aria-multiline":0,"aria-multiselectable":0,"aria-orientation":0,"aria-placeholder":0,"aria-pressed":0,"aria-readonly":0,"aria-required":0,"aria-selected":0,"aria-sort":0,"aria-valuemax":0,"aria-valuemin":0,"aria-valuenow":0,"aria-valuetext":0,"aria-atomic":0,"aria-busy":0,"aria-live":0,"aria-relevant":0,"aria-dropeffect":0,"aria-grabbed":0,"aria-activedescendant":0,"aria-colcount":0,"aria-colindex":0,"aria-colspan":0,"aria-controls":0,"aria-describedby":0,"aria-errormessage":0,"aria-flowto":0,"aria-labelledby":0,"aria-owns":0,"aria-posinset":0,"aria-rowcount":0,"aria-rowindex":0,"aria-rowspan":0,"aria-setsize":0},DOMAttributeNames:{},DOMPropertyNames:{}};t.exports=r},function(t,e,n){"use strict";var r=n(4),i=n(151),o={focusDOMComponent:function(){i(r.getNodeFromInstance(this))}};t.exports=o},function(t,e,n){"use strict";function r(){var t=window.opera;return"object"==typeof t&&"function"==typeof t.version&&parseInt(t.version(),10)<=12}function i(t){return(t.ctrlKey||t.altKey||t.metaKey)&&!(t.ctrlKey&&t.altKey)}function o(t){switch(t){case"topCompositionStart":return E.compositionStart;case"topCompositionEnd":return E.compositionEnd;case"topCompositionUpdate":return E.compositionUpdate}}function a(t,e){return"topKeyDown"===t&&e.keyCode===_}function u(t,e){switch(t){case"topKeyUp":return y.indexOf(e.keyCode)!==-1;case"topKeyDown":return e.keyCode!==_;case"topKeyPress":case"topMouseDown":case"topBlur":return!0;default:return!1}}function c(t){var e=t.detail;return"object"==typeof e&&"data"in e?e.data:null}function s(t,e,n,r){var i,s;if(b?i=o(t):S?u(t,n)&&(i=E.compositionEnd):a(t,n)&&(i=E.compositionStart),!i)return null;C&&(S||i!==E.compositionStart?i===E.compositionEnd&&S&&(s=S.getData()):S=v.getPooled(r));var l=g.getPooled(i,e,n,r);if(s)l.data=s;else{var f=c(n);null!==f&&(l.data=f)}return h.accumulateTwoPhaseDispatches(l),l}function l(t,e){switch(t){case"topCompositionEnd":return c(e);case"topKeyPress":var n=e.which;return n!==M?null:(T=!0,k);case"topTextInput":var r=e.data;return r===k&&T?null:r;default:return null}}function f(t,e){if(S){if("topCompositionEnd"===t||!b&&u(t,e)){var n=S.getData();return v.release(S),S=null,n}return null}switch(t){case"topPaste":return null;case"topKeyPress":return e.which&&!i(e)?String.fromCharCode(e.which):null;case"topCompositionEnd":return C?null:e.data;default:return null}}function p(t,e,n,r){var i;if(i=w?l(t,n):f(t,n),!i)return null;var o=m.getPooled(E.beforeInput,e,n,r);return o.data=i,h.accumulateTwoPhaseDispatches(o),o}var h=n(23),d=n(6),v=n(340),g=n(377),m=n(380),y=[9,13,27,32],_=229,b=d.canUseDOM&&"CompositionEvent"in window,x=null;d.canUseDOM&&"documentMode"in document&&(x=document.documentMode);var w=d.canUseDOM&&"TextEvent"in window&&!x&&!r(),C=d.canUseDOM&&(!b||x&&x>8&&x<=11),M=32,k=String.fromCharCode(M),E={beforeInput:{phasedRegistrationNames:{bubbled:"onBeforeInput",captured:"onBeforeInputCapture"},dependencies:["topCompositionEnd","topKeyPress","topTextInput","topPaste"]},compositionEnd:{phasedRegistrationNames:{bubbled:"onCompositionEnd",captured:"onCompositionEndCapture"},dependencies:["topBlur","topCompositionEnd","topKeyDown","topKeyPress","topKeyUp","topMouseDown"]},compositionStart:{phasedRegistrationNames:{bubbled:"onCompositionStart",captured:"onCompositionStartCapture"},dependencies:["topBlur","topCompositionStart","topKeyDown","topKeyPress","topKeyUp","topMouseDown"]},compositionUpdate:{phasedRegistrationNames:{bubbled:"onCompositionUpdate",captured:"onCompositionUpdateCapture"},dependencies:["topBlur","topCompositionUpdate","topKeyDown","topKeyPress","topKeyUp","topMouseDown"]}},T=!1,S=null,P={eventTypes:E,extractEvents:function(t,e,n,r){return[s(t,e,n,r),p(t,e,n,r)]}};t.exports=P},function(t,e,n){"use strict";var r=n(154),i=n(6),o=(n(9),n(319),n(386)),a=n(326),u=n(330),c=(n(1),u(function(t){return a(t)})),s=!1,l="cssFloat";if(i.canUseDOM){var f=document.createElement("div").style;try{f.font=""}catch(t){s=!0}void 0===document.documentElement.style.cssFloat&&(l="styleFloat")}var p={createMarkupForStyles:function(t,e){var n="";for(var r in t)if(t.hasOwnProperty(r)){var i=t[r];null!=i&&(n+=c(r)+":",n+=o(r,i,e)+";")}return n||null},setValueForStyles:function(t,e,n){var i=t.style;for(var a in e)if(e.hasOwnProperty(a)){var u=o(a,e[a],n);if("float"!==a&&"cssFloat"!==a||(a=l),u)i[a]=u;else{var c=s&&r.shorthandPropertyExpansions[a];if(c)for(var f in c)i[f]="";else i[a]=""}}}};t.exports=p},function(t,e,n){"use strict";function r(t){var e=t.nodeName&&t.nodeName.toLowerCase();return"select"===e||"input"===e&&"file"===t.type}function i(t){var e=C.getPooled(T.change,P,t,M(t));_.accumulateTwoPhaseDispatches(e),w.batchedUpdates(o,e)}function o(t){y.enqueueEvents(t),y.processEventQueue(!1)}function a(t,e){S=t,P=e,S.attachEvent("onchange",i)}function u(){S&&(S.detachEvent("onchange",i),S=null,P=null)}function c(t,e){if("topChange"===t)return e}function s(t,e,n){"topFocus"===t?(u(),a(e,n)):"topBlur"===t&&u()}function l(t,e){S=t,P=e,N=t.value,A=Object.getOwnPropertyDescriptor(t.constructor.prototype,"value"),Object.defineProperty(S,"value",D),S.attachEvent?S.attachEvent("onpropertychange",p):S.addEventListener("propertychange",p,!1)}function f(){S&&(delete S.value,S.detachEvent?S.detachEvent("onpropertychange",p):S.removeEventListener("propertychange",p,!1),S=null,P=null,N=null,A=null)}function p(t){if("value"===t.propertyName){var e=t.srcElement.value;e!==N&&(N=e,i(t))}}function h(t,e){if("topInput"===t)return e}function d(t,e,n){"topFocus"===t?(f(),l(e,n)):"topBlur"===t&&f()}function v(t,e){if(("topSelectionChange"===t||"topKeyUp"===t||"topKeyDown"===t)&&S&&S.value!==N)return N=S.value,P}function g(t){return t.nodeName&&"input"===t.nodeName.toLowerCase()&&("checkbox"===t.type||"radio"===t.type)}function m(t,e){if("topClick"===t)return e}var y=n(22),_=n(23),b=n(6),x=n(4),w=n(11),C=n(14),M=n(93),k=n(94),E=n(170),T={change:{phasedRegistrationNames:{bubbled:"onChange",captured:"onChangeCapture"},dependencies:["topBlur","topChange","topClick","topFocus","topInput","topKeyDown","topKeyUp","topSelectionChange"]}},S=null,P=null,N=null,A=null,O=!1;b.canUseDOM&&(O=k("change")&&(!document.documentMode||document.documentMode>8));var I=!1;b.canUseDOM&&(I=k("input")&&(!document.documentMode||document.documentMode>11));var D={get:function(){return A.get.call(this)},set:function(t){N=""+t,A.set.call(this,t)}},R={eventTypes:T,extractEvents:function(t,e,n,i){var o,a,u=e?x.getNodeFromInstance(e):window;if(r(u)?O?o=c:a=s:E(u)?I?o=h:(o=v,a=d):g(u)&&(o=m),o){var l=o(t,e);if(l){var f=C.getPooled(T.change,l,n,i);return f.type="change",_.accumulateTwoPhaseDispatches(f),f}}a&&a(t,u,e)}};t.exports=R},function(t,e,n){"use strict";var r=n(2),i=n(20),o=n(6),a=n(322),u=n(8),c=(n(0),{dangerouslyReplaceNodeWithMarkup:function(t,e){if(o.canUseDOM?void 0:r("56"),e?void 0:r("57"),"HTML"===t.nodeName?r("58"):void 0,"string"==typeof e){var n=a(e,u)[0];t.parentNode.replaceChild(n,t)}else i.replaceChildWithTree(t,e)}});t.exports=c},function(t,e,n){"use strict";var r=["ResponderEventPlugin","SimpleEventPlugin","TapEventPlugin","EnterLeaveEventPlugin","ChangeEventPlugin","SelectEventPlugin","BeforeInputEventPlugin"];t.exports=r},function(t,e,n){"use strict";var r=n(23),i=n(4),o=n(52),a={mouseEnter:{registrationName:"onMouseEnter",dependencies:["topMouseOut","topMouseOver"]},mouseLeave:{registrationName:"onMouseLeave",dependencies:["topMouseOut","topMouseOver"]}},u={eventTypes:a,extractEvents:function(t,e,n,u){if("topMouseOver"===t&&(n.relatedTarget||n.fromElement))return null;
if("topMouseOut"!==t&&"topMouseOver"!==t)return null;var c;if(u.window===u)c=u;else{var s=u.ownerDocument;c=s?s.defaultView||s.parentWindow:window}var l,f;if("topMouseOut"===t){l=e;var p=n.relatedTarget||n.toElement;f=p?i.getClosestInstanceFromNode(p):null}else l=null,f=e;if(l===f)return null;var h=null==l?c:i.getNodeFromInstance(l),d=null==f?c:i.getNodeFromInstance(f),v=o.getPooled(a.mouseLeave,l,n,u);v.type="mouseleave",v.target=h,v.relatedTarget=d;var g=o.getPooled(a.mouseEnter,f,n,u);return g.type="mouseenter",g.target=d,g.relatedTarget=h,r.accumulateEnterLeaveDispatches(v,g,l,f),[v,g]}};t.exports=u},function(t,e,n){"use strict";var r={topAbort:null,topAnimationEnd:null,topAnimationIteration:null,topAnimationStart:null,topBlur:null,topCanPlay:null,topCanPlayThrough:null,topChange:null,topClick:null,topCompositionEnd:null,topCompositionStart:null,topCompositionUpdate:null,topContextMenu:null,topCopy:null,topCut:null,topDoubleClick:null,topDrag:null,topDragEnd:null,topDragEnter:null,topDragExit:null,topDragLeave:null,topDragOver:null,topDragStart:null,topDrop:null,topDurationChange:null,topEmptied:null,topEncrypted:null,topEnded:null,topError:null,topFocus:null,topInput:null,topInvalid:null,topKeyDown:null,topKeyPress:null,topKeyUp:null,topLoad:null,topLoadedData:null,topLoadedMetadata:null,topLoadStart:null,topMouseDown:null,topMouseMove:null,topMouseOut:null,topMouseOver:null,topMouseUp:null,topPaste:null,topPause:null,topPlay:null,topPlaying:null,topProgress:null,topRateChange:null,topReset:null,topScroll:null,topSeeked:null,topSeeking:null,topSelectionChange:null,topStalled:null,topSubmit:null,topSuspend:null,topTextInput:null,topTimeUpdate:null,topTouchCancel:null,topTouchEnd:null,topTouchMove:null,topTouchStart:null,topTransitionEnd:null,topVolumeChange:null,topWaiting:null,topWheel:null},i={topLevelTypes:r};t.exports=i},function(t,e,n){"use strict";function r(t){this._root=t,this._startText=this.getText(),this._fallbackText=null}var i=n(3),o=n(17),a=n(168);i(r.prototype,{destructor:function(){this._root=null,this._startText=null,this._fallbackText=null},getText:function(){return"value"in this._root?this._root.value:this._root[a()]},getData:function(){if(this._fallbackText)return this._fallbackText;var t,e,n=this._startText,r=n.length,i=this.getText(),o=i.length;for(t=0;t<r&&n[t]===i[t];t++);var a=r-t;for(e=1;e<=a&&n[r-e]===i[o-e];e++);var u=e>1?1-e:void 0;return this._fallbackText=i.slice(t,u),this._fallbackText}}),o.addPoolingTo(r),t.exports=r},function(t,e,n){"use strict";var r=n(21),i=r.injection.MUST_USE_PROPERTY,o=r.injection.HAS_BOOLEAN_VALUE,a=r.injection.HAS_NUMERIC_VALUE,u=r.injection.HAS_POSITIVE_NUMERIC_VALUE,c=r.injection.HAS_OVERLOADED_BOOLEAN_VALUE,s={isCustomAttribute:RegExp.prototype.test.bind(new RegExp("^(data|aria)-["+r.ATTRIBUTE_NAME_CHAR+"]*$")),Properties:{accept:0,acceptCharset:0,accessKey:0,action:0,allowFullScreen:o,allowTransparency:0,alt:0,as:0,async:o,autoComplete:0,autoPlay:o,capture:o,cellPadding:0,cellSpacing:0,charSet:0,challenge:0,checked:i|o,cite:0,classID:0,className:0,cols:u,colSpan:0,content:0,contentEditable:0,contextMenu:0,controls:o,coords:0,crossOrigin:0,data:0,dateTime:0,default:o,defer:o,dir:0,disabled:o,download:c,draggable:0,encType:0,form:0,formAction:0,formEncType:0,formMethod:0,formNoValidate:o,formTarget:0,frameBorder:0,headers:0,height:0,hidden:o,high:0,href:0,hrefLang:0,htmlFor:0,httpEquiv:0,icon:0,id:0,inputMode:0,integrity:0,is:0,keyParams:0,keyType:0,kind:0,label:0,lang:0,list:0,loop:o,low:0,manifest:0,marginHeight:0,marginWidth:0,max:0,maxLength:0,media:0,mediaGroup:0,method:0,min:0,minLength:0,multiple:i|o,muted:i|o,name:0,nonce:0,noValidate:o,open:o,optimum:0,pattern:0,placeholder:0,playsInline:o,poster:0,preload:0,profile:0,radioGroup:0,readOnly:o,referrerPolicy:0,rel:0,required:o,reversed:o,role:0,rows:u,rowSpan:a,sandbox:0,scope:0,scoped:o,scrolling:0,seamless:o,selected:i|o,shape:0,size:u,sizes:0,span:u,spellCheck:0,src:0,srcDoc:0,srcLang:0,srcSet:0,start:a,step:0,style:0,summary:0,tabIndex:0,target:0,title:0,type:0,useMap:0,value:0,width:0,wmode:0,wrap:0,about:0,datatype:0,inlist:0,prefix:0,property:0,resource:0,typeof:0,vocab:0,autoCapitalize:0,autoCorrect:0,autoSave:0,color:0,itemProp:0,itemScope:o,itemType:0,itemID:0,itemRef:0,results:0,security:0,unselectable:0},DOMAttributeNames:{acceptCharset:"accept-charset",className:"class",htmlFor:"for",httpEquiv:"http-equiv"},DOMPropertyNames:{}};t.exports=s},function(t,e,n){"use strict";(function(e){function r(t,e,n,r){var i=void 0===t[n];null!=e&&i&&(t[n]=o(e,!0))}var i=n(24),o=n(169),a=(n(84),n(95)),u=n(172);n(1);"undefined"!=typeof e&&e.env,1;var c={instantiateChildren:function(t,e,n,i){if(null==t)return null;var o={};return u(t,r,o),o},updateChildren:function(t,e,n,r,u,c,s,l,f){if(e||t){var p,h;for(p in e)if(e.hasOwnProperty(p)){h=t&&t[p];var d=h&&h._currentElement,v=e[p];if(null!=h&&a(d,v))i.receiveComponent(h,v,u,l),e[p]=h;else{h&&(r[p]=i.getHostNode(h),i.unmountComponent(h,!1));var g=o(v,!0);e[p]=g;var m=i.mountComponent(g,u,c,s,l,f);n.push(m)}}for(p in t)!t.hasOwnProperty(p)||e&&e.hasOwnProperty(p)||(h=t[p],r[p]=i.getHostNode(h),i.unmountComponent(h,!1))}},unmountChildren:function(t,e){for(var n in t)if(t.hasOwnProperty(n)){var r=t[n];i.unmountComponent(r,e)}}};t.exports=c}).call(e,n(153))},function(t,e,n){"use strict";var r=n(81),i=n(350),o={processChildrenUpdates:i.dangerouslyProcessChildrenUpdates,replaceNodeWithMarkup:r.dangerouslyReplaceNodeWithMarkup};t.exports=o},function(t,e,n){"use strict";function r(t){}function i(t,e){}function o(t){return!(!t.prototype||!t.prototype.isReactComponent)}function a(t){return!(!t.prototype||!t.prototype.isPureReactComponent)}var u=n(2),c=n(3),s=n(26),l=n(86),f=n(15),p=n(87),h=n(40),d=(n(9),n(164)),v=n(24),g=n(38),m=(n(0),n(80)),y=n(95),_=(n(1),{ImpureClass:0,PureClass:1,StatelessFunctional:2});r.prototype.render=function(){var t=h.get(this)._currentElement.type,e=t(this.props,this.context,this.updater);return i(t,e),e};var b=1,x={construct:function(t){this._currentElement=t,this._rootNodeID=0,this._compositeType=null,this._instance=null,this._hostParent=null,this._hostContainerInfo=null,this._updateBatchNumber=null,this._pendingElement=null,this._pendingStateQueue=null,this._pendingReplaceState=!1,this._pendingForceUpdate=!1,this._renderedNodeType=null,this._renderedComponent=null,this._context=null,this._mountOrder=0,this._topLevelWrapper=null,this._pendingCallbacks=null,this._calledComponentWillUnmount=!1},mountComponent:function(t,e,n,c){this._context=c,this._mountOrder=b++,this._hostParent=e,this._hostContainerInfo=n;var l,f=this._currentElement.props,p=this._processContext(c),d=this._currentElement.type,v=t.getUpdateQueue(),m=o(d),y=this._constructComponent(m,f,p,v);m||null!=y&&null!=y.render?a(d)?this._compositeType=_.PureClass:this._compositeType=_.ImpureClass:(l=y,i(d,l),null===y||y===!1||s.isValidElement(y)?void 0:u("105",d.displayName||d.name||"Component"),y=new r(d),this._compositeType=_.StatelessFunctional);y.props=f,y.context=p,y.refs=g,y.updater=v,this._instance=y,h.set(y,this);var x=y.state;void 0===x&&(y.state=x=null),"object"!=typeof x||Array.isArray(x)?u("106",this.getName()||"ReactCompositeComponent"):void 0,this._pendingStateQueue=null,this._pendingReplaceState=!1,this._pendingForceUpdate=!1;var w;return w=y.unstable_handleError?this.performInitialMountWithErrorHandling(l,e,n,t,c):this.performInitialMount(l,e,n,t,c),y.componentDidMount&&t.getReactMountReady().enqueue(y.componentDidMount,y),w},_constructComponent:function(t,e,n,r){return this._constructComponentWithoutOwner(t,e,n,r)},_constructComponentWithoutOwner:function(t,e,n,r){var i=this._currentElement.type;return t?new i(e,n,r):i(e,n,r)},performInitialMountWithErrorHandling:function(t,e,n,r,i){var o,a=r.checkpoint();try{o=this.performInitialMount(t,e,n,r,i)}catch(u){r.rollback(a),this._instance.unstable_handleError(u),this._pendingStateQueue&&(this._instance.state=this._processPendingState(this._instance.props,this._instance.context)),a=r.checkpoint(),this._renderedComponent.unmountComponent(!0),r.rollback(a),o=this.performInitialMount(t,e,n,r,i)}return o},performInitialMount:function(t,e,n,r,i){var o=this._instance,a=0;o.componentWillMount&&(o.componentWillMount(),this._pendingStateQueue&&(o.state=this._processPendingState(o.props,o.context))),void 0===t&&(t=this._renderValidatedComponent());var u=d.getType(t);this._renderedNodeType=u;var c=this._instantiateReactComponent(t,u!==d.EMPTY);this._renderedComponent=c;var s=v.mountComponent(c,r,e,n,this._processChildContext(i),a);return s},getHostNode:function(){return v.getHostNode(this._renderedComponent)},unmountComponent:function(t){if(this._renderedComponent){var e=this._instance;if(e.componentWillUnmount&&!e._calledComponentWillUnmount)if(e._calledComponentWillUnmount=!0,t){var n=this.getName()+".componentWillUnmount()";p.invokeGuardedCallback(n,e.componentWillUnmount.bind(e))}else e.componentWillUnmount();this._renderedComponent&&(v.unmountComponent(this._renderedComponent,t),this._renderedNodeType=null,this._renderedComponent=null,this._instance=null),this._pendingStateQueue=null,this._pendingReplaceState=!1,this._pendingForceUpdate=!1,this._pendingCallbacks=null,this._pendingElement=null,this._context=null,this._rootNodeID=0,this._topLevelWrapper=null,h.remove(e)}},_maskContext:function(t){var e=this._currentElement.type,n=e.contextTypes;if(!n)return g;var r={};for(var i in n)r[i]=t[i];return r},_processContext:function(t){var e=this._maskContext(t);return e},_processChildContext:function(t){var e,n=this._currentElement.type,r=this._instance;if(r.getChildContext&&(e=r.getChildContext()),e){"object"!=typeof n.childContextTypes?u("107",this.getName()||"ReactCompositeComponent"):void 0;for(var i in e)i in n.childContextTypes?void 0:u("108",this.getName()||"ReactCompositeComponent",i);return c({},t,e)}return t},_checkContextTypes:function(t,e,n){},receiveComponent:function(t,e,n){var r=this._currentElement,i=this._context;this._pendingElement=null,this.updateComponent(e,r,t,i,n)},performUpdateIfNecessary:function(t){null!=this._pendingElement?v.receiveComponent(this,this._pendingElement,t,this._context):null!==this._pendingStateQueue||this._pendingForceUpdate?this.updateComponent(t,this._currentElement,this._currentElement,this._context,this._context):this._updateBatchNumber=null},updateComponent:function(t,e,n,r,i){var o=this._instance;null==o?u("136",this.getName()||"ReactCompositeComponent"):void 0;var a,c=!1;this._context===i?a=o.context:(a=this._processContext(i),c=!0);var s=e.props,l=n.props;e!==n&&(c=!0),c&&o.componentWillReceiveProps&&o.componentWillReceiveProps(l,a);var f=this._processPendingState(l,a),p=!0;this._pendingForceUpdate||(o.shouldComponentUpdate?p=o.shouldComponentUpdate(l,f,a):this._compositeType===_.PureClass&&(p=!m(s,l)||!m(o.state,f))),this._updateBatchNumber=null,p?(this._pendingForceUpdate=!1,this._performComponentUpdate(n,l,f,a,t,i)):(this._currentElement=n,this._context=i,o.props=l,o.state=f,o.context=a)},_processPendingState:function(t,e){var n=this._instance,r=this._pendingStateQueue,i=this._pendingReplaceState;if(this._pendingReplaceState=!1,this._pendingStateQueue=null,!r)return n.state;if(i&&1===r.length)return r[0];for(var o=c({},i?r[0]:n.state),a=i?1:0;a<r.length;a++){var u=r[a];c(o,"function"==typeof u?u.call(n,o,t,e):u)}return o},_performComponentUpdate:function(t,e,n,r,i,o){var a,u,c,s=this._instance,l=Boolean(s.componentDidUpdate);l&&(a=s.props,u=s.state,c=s.context),s.componentWillUpdate&&s.componentWillUpdate(e,n,r),this._currentElement=t,this._context=o,s.props=e,s.state=n,s.context=r,this._updateRenderedComponent(i,o),l&&i.getReactMountReady().enqueue(s.componentDidUpdate.bind(s,a,u,c),s)},_updateRenderedComponent:function(t,e){var n=this._renderedComponent,r=n._currentElement,i=this._renderValidatedComponent(),o=0;if(y(r,i))v.receiveComponent(n,i,t,this._processChildContext(e));else{var a=v.getHostNode(n);v.unmountComponent(n,!1);var u=d.getType(i);this._renderedNodeType=u;var c=this._instantiateReactComponent(i,u!==d.EMPTY);this._renderedComponent=c;var s=v.mountComponent(c,t,this._hostParent,this._hostContainerInfo,this._processChildContext(e),o);this._replaceNodeWithMarkup(a,s,n)}},_replaceNodeWithMarkup:function(t,e,n){l.replaceNodeWithMarkup(t,e,n)},_renderValidatedComponentWithoutOwnerOrContext:function(){var t,e=this._instance;return t=e.render()},_renderValidatedComponent:function(){var t;if(this._compositeType!==_.StatelessFunctional){f.current=this;try{t=this._renderValidatedComponentWithoutOwnerOrContext()}finally{f.current=null}}else t=this._renderValidatedComponentWithoutOwnerOrContext();return null===t||t===!1||s.isValidElement(t)?void 0:u("109",this.getName()||"ReactCompositeComponent"),t},attachRef:function(t,e){var n=this.getPublicInstance();null==n?u("110"):void 0;var r=e.getPublicInstance(),i=n.refs===g?n.refs={}:n.refs;i[t]=r},detachRef:function(t){var e=this.getPublicInstance().refs;delete e[t]},getName:function(){var t=this._currentElement.type,e=this._instance&&this._instance.constructor;return t.displayName||e&&e.displayName||t.name||e&&e.name||null},getPublicInstance:function(){var t=this._instance;return this._compositeType===_.StatelessFunctional?null:t},_instantiateReactComponent:null};t.exports=x},function(t,e,n){"use strict";var r=n(4),i=n(358),o=n(163),a=n(24),u=n(11),c=n(371),s=n(387),l=n(167),f=n(395);n(1);i.inject();var p={findDOMNode:s,render:o.render,unmountComponentAtNode:o.unmountComponentAtNode,version:c,unstable_batchedUpdates:u.batchedUpdates,unstable_renderSubtreeIntoContainer:f};"undefined"!=typeof __REACT_DEVTOOLS_GLOBAL_HOOK__&&"function"==typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.inject&&__REACT_DEVTOOLS_GLOBAL_HOOK__.inject({ComponentTree:{getClosestInstanceFromNode:r.getClosestInstanceFromNode,getNodeFromInstance:function(t){return t._renderedComponent&&(t=l(t)),t?r.getNodeFromInstance(t):null}},Mount:o,Reconciler:a});t.exports=p},function(t,e,n){"use strict";function r(t){if(t){var e=t._currentElement._owner||null;if(e){var n=e.getName();if(n)return" This DOM node was rendered by `"+n+"`."}}return""}function i(t,e){e&&(G[t._tag]&&(null!=e.children||null!=e.dangerouslySetInnerHTML?v("137",t._tag,t._currentElement._owner?" Check the render method of "+t._currentElement._owner.getName()+".":""):void 0),null!=e.dangerouslySetInnerHTML&&(null!=e.children?v("60"):void 0,"object"==typeof e.dangerouslySetInnerHTML&&V in e.dangerouslySetInnerHTML?void 0:v("61")),null!=e.style&&"object"!=typeof e.style?v("62",r(t)):void 0)}function o(t,e,n,r){if(!(r instanceof I)){var i=t._hostContainerInfo,o=i._node&&i._node.nodeType===H,u=o?i._node:i._ownerDocument;F(e,u),r.getReactMountReady().enqueue(a,{inst:t,registrationName:e,listener:n})}}function a(){var t=this;C.putListener(t.inst,t.registrationName,t.listener)}function u(){var t=this;S.postMountWrapper(t)}function c(){var t=this;A.postMountWrapper(t)}function s(){var t=this;P.postMountWrapper(t)}function l(){var t=this;t._rootNodeID?void 0:v("63");var e=U(t);switch(e?void 0:v("64"),t._tag){case"iframe":case"object":t._wrapperState.listeners=[k.trapBubbledEvent("topLoad","load",e)];break;case"video":case"audio":t._wrapperState.listeners=[];for(var n in q)q.hasOwnProperty(n)&&t._wrapperState.listeners.push(k.trapBubbledEvent(n,q[n],e));break;case"source":t._wrapperState.listeners=[k.trapBubbledEvent("topError","error",e)];break;case"img":t._wrapperState.listeners=[k.trapBubbledEvent("topError","error",e),k.trapBubbledEvent("topLoad","load",e)];break;case"form":t._wrapperState.listeners=[k.trapBubbledEvent("topReset","reset",e),k.trapBubbledEvent("topSubmit","submit",e)];break;case"input":case"select":case"textarea":t._wrapperState.listeners=[k.trapBubbledEvent("topInvalid","invalid",e)]}}function f(){N.postUpdateWrapper(this)}function p(t){Z.call(X,t)||($.test(t)?void 0:v("65",t),X[t]=!0)}function h(t,e){return t.indexOf("-")>=0||null!=e.is}function d(t){var e=t.type;p(e),this._currentElement=t,this._tag=e.toLowerCase(),this._namespaceURI=null,this._renderedChildren=null,this._previousStyle=null,this._previousStyleCopy=null,this._hostNode=null,this._hostParent=null,this._rootNodeID=0,this._domID=0,this._hostContainerInfo=null,this._wrapperState=null,this._topLevelWrapper=null,this._flags=0}var v=n(2),g=n(3),m=n(332),y=n(334),_=n(20),b=n(82),x=n(21),w=n(156),C=n(22),M=n(83),k=n(51),E=n(157),T=n(4),S=n(351),P=n(352),N=n(158),A=n(355),O=(n(9),n(364)),I=n(369),D=(n(8),n(54)),R=(n(0),n(94),n(80),n(96),n(1),E),L=C.deleteListener,U=T.getNodeFromInstance,F=k.listenTo,j=M.registrationNameModules,B={string:!0,number:!0},W="style",V="__html",z={children:null,dangerouslySetInnerHTML:null,suppressContentEditableWarning:null},H=11,q={topAbort:"abort",topCanPlay:"canplay",topCanPlayThrough:"canplaythrough",topDurationChange:"durationchange",topEmptied:"emptied",topEncrypted:"encrypted",topEnded:"ended",topError:"error",topLoadedData:"loadeddata",topLoadedMetadata:"loadedmetadata",topLoadStart:"loadstart",topPause:"pause",topPlay:"play",topPlaying:"playing",topProgress:"progress",topRateChange:"ratechange",topSeeked:"seeked",topSeeking:"seeking",topStalled:"stalled",topSuspend:"suspend",topTimeUpdate:"timeupdate",topVolumeChange:"volumechange",topWaiting:"waiting"},Y={area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0},K={listing:!0,pre:!0,textarea:!0},G=g({menuitem:!0},Y),$=/^[a-zA-Z][a-zA-Z:_\.\-\d]*$/,X={},Z={}.hasOwnProperty,Q=1;d.displayName="ReactDOMComponent",d.Mixin={mountComponent:function(t,e,n,r){this._rootNodeID=Q++,this._domID=n._idCounter++,this._hostParent=e,this._hostContainerInfo=n;var o=this._currentElement.props;switch(this._tag){case"audio":case"form":case"iframe":case"img":case"link":case"object":case"source":case"video":this._wrapperState={listeners:null},t.getReactMountReady().enqueue(l,this);break;case"input":S.mountWrapper(this,o,e),o=S.getHostProps(this,o),t.getReactMountReady().enqueue(l,this);break;case"option":P.mountWrapper(this,o,e),o=P.getHostProps(this,o);break;case"select":N.mountWrapper(this,o,e),o=N.getHostProps(this,o),t.getReactMountReady().enqueue(l,this);break;case"textarea":A.mountWrapper(this,o,e),o=A.getHostProps(this,o),t.getReactMountReady().enqueue(l,this)}i(this,o);var a,f;null!=e?(a=e._namespaceURI,f=e._tag):n._tag&&(a=n._namespaceURI,f=n._tag),(null==a||a===b.svg&&"foreignobject"===f)&&(a=b.html),a===b.html&&("svg"===this._tag?a=b.svg:"math"===this._tag&&(a=b.mathml)),this._namespaceURI=a;var p;if(t.useCreateElement){var h,d=n._ownerDocument;if(a===b.html)if("script"===this._tag){var v=d.createElement("div"),g=this._currentElement.type;v.innerHTML="<"+g+"></"+g+">",h=v.removeChild(v.firstChild)}else h=o.is?d.createElement(this._currentElement.type,o.is):d.createElement(this._currentElement.type);else h=d.createElementNS(a,this._currentElement.type);T.precacheNode(this,h),this._flags|=R.hasCachedChildNodes,this._hostParent||w.setAttributeForRoot(h),this._updateDOMProperties(null,o,t);var y=_(h);this._createInitialChildren(t,o,r,y),p=y}else{var x=this._createOpenTagMarkupAndPutListeners(t,o),C=this._createContentMarkup(t,o,r);p=!C&&Y[this._tag]?x+"/>":x+">"+C+"</"+this._currentElement.type+">"}switch(this._tag){case"input":t.getReactMountReady().enqueue(u,this),o.autoFocus&&t.getReactMountReady().enqueue(m.focusDOMComponent,this);break;case"textarea":t.getReactMountReady().enqueue(c,this),o.autoFocus&&t.getReactMountReady().enqueue(m.focusDOMComponent,this);break;case"select":o.autoFocus&&t.getReactMountReady().enqueue(m.focusDOMComponent,this);break;case"button":o.autoFocus&&t.getReactMountReady().enqueue(m.focusDOMComponent,this);break;case"option":t.getReactMountReady().enqueue(s,this)}return p},_createOpenTagMarkupAndPutListeners:function(t,e){var n="<"+this._currentElement.type;for(var r in e)if(e.hasOwnProperty(r)){var i=e[r];if(null!=i)if(j.hasOwnProperty(r))i&&o(this,r,i,t);else{r===W&&(i&&(i=this._previousStyleCopy=g({},e.style)),i=y.createMarkupForStyles(i,this));var a=null;null!=this._tag&&h(this._tag,e)?z.hasOwnProperty(r)||(a=w.createMarkupForCustomAttribute(r,i)):a=w.createMarkupForProperty(r,i),a&&(n+=" "+a)}}return t.renderToStaticMarkup?n:(this._hostParent||(n+=" "+w.createMarkupForRoot()),n+=" "+w.createMarkupForID(this._domID))},_createContentMarkup:function(t,e,n){var r="",i=e.dangerouslySetInnerHTML;if(null!=i)null!=i.__html&&(r=i.__html);else{var o=B[typeof e.children]?e.children:null,a=null!=o?null:e.children;if(null!=o)r=D(o);else if(null!=a){var u=this.mountChildren(a,t,n);r=u.join("")}}return K[this._tag]&&"\n"===r.charAt(0)?"\n"+r:r},_createInitialChildren:function(t,e,n,r){var i=e.dangerouslySetInnerHTML;if(null!=i)null!=i.__html&&_.queueHTML(r,i.__html);else{var o=B[typeof e.children]?e.children:null,a=null!=o?null:e.children;if(null!=o)""!==o&&_.queueText(r,o);else if(null!=a)for(var u=this.mountChildren(a,t,n),c=0;c<u.length;c++)_.queueChild(r,u[c])}},receiveComponent:function(t,e,n){var r=this._currentElement;this._currentElement=t,this.updateComponent(e,r,t,n)},updateComponent:function(t,e,n,r){var o=e.props,a=this._currentElement.props;switch(this._tag){case"input":o=S.getHostProps(this,o),a=S.getHostProps(this,a);break;case"option":o=P.getHostProps(this,o),a=P.getHostProps(this,a);break;case"select":o=N.getHostProps(this,o),a=N.getHostProps(this,a);break;case"textarea":o=A.getHostProps(this,o),a=A.getHostProps(this,a)}switch(i(this,a),this._updateDOMProperties(o,a,t),this._updateDOMChildren(o,a,t,r),this._tag){case"input":S.updateWrapper(this);break;case"textarea":A.updateWrapper(this);break;case"select":t.getReactMountReady().enqueue(f,this)}},_updateDOMProperties:function(t,e,n){var r,i,a;for(r in t)if(!e.hasOwnProperty(r)&&t.hasOwnProperty(r)&&null!=t[r])if(r===W){var u=this._previousStyleCopy;for(i in u)u.hasOwnProperty(i)&&(a=a||{},a[i]="");this._previousStyleCopy=null}else j.hasOwnProperty(r)?t[r]&&L(this,r):h(this._tag,t)?z.hasOwnProperty(r)||w.deleteValueForAttribute(U(this),r):(x.properties[r]||x.isCustomAttribute(r))&&w.deleteValueForProperty(U(this),r);for(r in e){var c=e[r],s=r===W?this._previousStyleCopy:null!=t?t[r]:void 0;if(e.hasOwnProperty(r)&&c!==s&&(null!=c||null!=s))if(r===W)if(c?c=this._previousStyleCopy=g({},c):this._previousStyleCopy=null,s){for(i in s)!s.hasOwnProperty(i)||c&&c.hasOwnProperty(i)||(a=a||{},a[i]="");for(i in c)c.hasOwnProperty(i)&&s[i]!==c[i]&&(a=a||{},a[i]=c[i])}else a=c;else if(j.hasOwnProperty(r))c?o(this,r,c,n):s&&L(this,r);else if(h(this._tag,e))z.hasOwnProperty(r)||w.setValueForAttribute(U(this),r,c);else if(x.properties[r]||x.isCustomAttribute(r)){var l=U(this);null!=c?w.setValueForProperty(l,r,c):w.deleteValueForProperty(l,r)}}a&&y.setValueForStyles(U(this),a,this)},_updateDOMChildren:function(t,e,n,r){var i=B[typeof t.children]?t.children:null,o=B[typeof e.children]?e.children:null,a=t.dangerouslySetInnerHTML&&t.dangerouslySetInnerHTML.__html,u=e.dangerouslySetInnerHTML&&e.dangerouslySetInnerHTML.__html,c=null!=i?null:t.children,s=null!=o?null:e.children,l=null!=i||null!=a,f=null!=o||null!=u;null!=c&&null==s?this.updateChildren(null,n,r):l&&!f&&this.updateTextContent(""),null!=o?i!==o&&this.updateTextContent(""+o):null!=u?a!==u&&this.updateMarkup(""+u):null!=s&&this.updateChildren(s,n,r)},getHostNode:function(){return U(this)},unmountComponent:function(t){switch(this._tag){case"audio":case"form":case"iframe":case"img":case"link":case"object":case"source":case"video":var e=this._wrapperState.listeners;if(e)for(var n=0;n<e.length;n++)e[n].remove();break;case"html":case"head":case"body":v("66",this._tag)}this.unmountChildren(t),T.uncacheNode(this),C.deleteAllListeners(this),this._rootNodeID=0,this._domID=0,this._wrapperState=null},getPublicInstance:function(){return U(this)}},g(d.prototype,d.Mixin,O.Mixin),t.exports=d},function(t,e,n){"use strict";function r(t,e){var n={_topLevelWrapper:t,_idCounter:1,_ownerDocument:e?e.nodeType===i?e:e.ownerDocument:null,_node:e,_tag:e?e.nodeName.toLowerCase():null,_namespaceURI:e?e.namespaceURI:null};return n}var i=(n(96),9);t.exports=r},function(t,e,n){"use strict";var r=n(3),i=n(20),o=n(4),a=function(t){this._currentElement=null,this._hostNode=null,this._hostParent=null,this._hostContainerInfo=null,this._domID=0};r(a.prototype,{mountComponent:function(t,e,n,r){var a=n._idCounter++;this._domID=a,this._hostParent=e,this._hostContainerInfo=n;var u=" react-empty: "+this._domID+" ";if(t.useCreateElement){var c=n._ownerDocument,s=c.createComment(u);return o.precacheNode(this,s),i(s)}return t.renderToStaticMarkup?"":"<!--"+u+"-->"},receiveComponent:function(){},getHostNode:function(){return o.getNodeFromInstance(this)},unmountComponent:function(){o.uncacheNode(this)}}),t.exports=a},function(t,e,n){"use strict";var r={useCreateElement:!0,useFiber:!1};t.exports=r},function(t,e,n){"use strict";var r=n(81),i=n(4),o={dangerouslyProcessChildrenUpdates:function(t,e){var n=i.getNodeFromInstance(t);r.processUpdates(n,e)}};t.exports=o},function(t,e,n){"use strict";function r(){this._rootNodeID&&f.updateWrapper(this)}function i(t){var e=this._currentElement.props,n=c.executeOnChange(e,t);l.asap(r,this);var i=e.name;if("radio"===e.type&&null!=i){for(var a=s.getNodeFromInstance(this),u=a;u.parentNode;)u=u.parentNode;for(var f=u.querySelectorAll("input[name="+JSON.stringify(""+i)+'][type="radio"]'),p=0;p<f.length;p++){var h=f[p];if(h!==a&&h.form===a.form){var d=s.getInstanceFromNode(h);d?void 0:o("90"),l.asap(r,d)}}}return n}var o=n(2),a=n(3),u=n(156),c=n(85),s=n(4),l=n(11),f=(n(0),n(1),{getHostProps:function(t,e){var n=c.getValue(e),r=c.getChecked(e),i=a({type:void 0,step:void 0,min:void 0,max:void 0},e,{defaultChecked:void 0,defaultValue:void 0,value:null!=n?n:t._wrapperState.initialValue,checked:null!=r?r:t._wrapperState.initialChecked,onChange:t._wrapperState.onChange});return i},mountWrapper:function(t,e){var n=e.defaultValue;t._wrapperState={initialChecked:null!=e.checked?e.checked:e.defaultChecked,initialValue:null!=e.value?e.value:n,listeners:null,onChange:i.bind(t)}},updateWrapper:function(t){var e=t._currentElement.props,n=e.checked;null!=n&&u.setValueForProperty(s.getNodeFromInstance(t),"checked",n||!1);var r=s.getNodeFromInstance(t),i=c.getValue(e);if(null!=i){var o=""+i;o!==r.value&&(r.value=o)}else null==e.value&&null!=e.defaultValue&&r.defaultValue!==""+e.defaultValue&&(r.defaultValue=""+e.defaultValue),null==e.checked&&null!=e.defaultChecked&&(r.defaultChecked=!!e.defaultChecked)},postMountWrapper:function(t){var e=t._currentElement.props,n=s.getNodeFromInstance(t);switch(e.type){case"submit":case"reset":break;case"color":case"date":case"datetime":case"datetime-local":case"month":case"time":case"week":n.value="",n.value=n.defaultValue;break;default:n.value=n.value}var r=n.name;""!==r&&(n.name=""),n.defaultChecked=!n.defaultChecked,n.defaultChecked=!n.defaultChecked,""!==r&&(n.name=r)}});t.exports=f},function(t,e,n){"use strict";function r(t){var e="";return o.Children.forEach(t,function(t){null!=t&&("string"==typeof t||"number"==typeof t?e+=t:c||(c=!0))}),e}var i=n(3),o=n(26),a=n(4),u=n(158),c=(n(1),!1),s={mountWrapper:function(t,e,n){var i=null;if(null!=n){var o=n;"optgroup"===o._tag&&(o=o._hostParent),null!=o&&"select"===o._tag&&(i=u.getSelectValueContext(o))}var a=null;if(null!=i){var c;if(c=null!=e.value?e.value+"":r(e.children),a=!1,Array.isArray(i)){for(var s=0;s<i.length;s++)if(""+i[s]===c){a=!0;break}}else a=""+i===c}t._wrapperState={selected:a}},postMountWrapper:function(t){var e=t._currentElement.props;if(null!=e.value){var n=a.getNodeFromInstance(t);n.setAttribute("value",e.value)}},getHostProps:function(t,e){var n=i({selected:void 0,children:void 0},e);null!=t._wrapperState.selected&&(n.selected=t._wrapperState.selected);var o=r(e.children);return o&&(n.children=o),n}};t.exports=s},function(t,e,n){"use strict";function r(t,e,n,r){return t===n&&e===r}function i(t){var e=document.selection,n=e.createRange(),r=n.text.length,i=n.duplicate();i.moveToElementText(t),i.setEndPoint("EndToStart",n);var o=i.text.length,a=o+r;return{start:o,end:a}}function o(t){var e=window.getSelection&&window.getSelection();if(!e||0===e.rangeCount)return null;var n=e.anchorNode,i=e.anchorOffset,o=e.focusNode,a=e.focusOffset,u=e.getRangeAt(0);try{u.startContainer.nodeType,u.endContainer.nodeType}catch(t){return null}var c=r(e.anchorNode,e.anchorOffset,e.focusNode,e.focusOffset),s=c?0:u.toString().length,l=u.cloneRange();l.selectNodeContents(t),l.setEnd(u.startContainer,u.startOffset);var f=r(l.startContainer,l.startOffset,l.endContainer,l.endOffset),p=f?0:l.toString().length,h=p+s,d=document.createRange();d.setStart(n,i),d.setEnd(o,a);var v=d.collapsed;return{start:v?h:p,end:v?p:h}}function a(t,e){var n,r,i=document.selection.createRange().duplicate();void 0===e.end?(n=e.start,r=n):e.start>e.end?(n=e.end,r=e.start):(n=e.start,r=e.end),i.moveToElementText(t),i.moveStart("character",n),i.setEndPoint("EndToStart",i),i.moveEnd("character",r-n),i.select()}function u(t,e){if(window.getSelection){var n=window.getSelection(),r=t[l()].length,i=Math.min(e.start,r),o=void 0===e.end?i:Math.min(e.end,r);if(!n.extend&&i>o){var a=o;o=i,i=a}var u=s(t,i),c=s(t,o);if(u&&c){var f=document.createRange();f.setStart(u.node,u.offset),n.removeAllRanges(),i>o?(n.addRange(f),n.extend(c.node,c.offset)):(f.setEnd(c.node,c.offset),n.addRange(f))}}}var c=n(6),s=n(392),l=n(168),f=c.canUseDOM&&"selection"in document&&!("getSelection"in window),p={getOffsets:f?i:o,setOffsets:f?a:u};t.exports=p},function(t,e,n){"use strict";var r=n(2),i=n(3),o=n(81),a=n(20),u=n(4),c=n(54),s=(n(0),n(96),function(t){this._currentElement=t,this._stringText=""+t,this._hostNode=null,this._hostParent=null,this._domID=0,this._mountIndex=0,this._closingComment=null,this._commentNodes=null});i(s.prototype,{mountComponent:function(t,e,n,r){var i=n._idCounter++,o=" react-text: "+i+" ",s=" /react-text ";if(this._domID=i,this._hostParent=e,t.useCreateElement){var l=n._ownerDocument,f=l.createComment(o),p=l.createComment(s),h=a(l.createDocumentFragment());return a.queueChild(h,a(f)),this._stringText&&a.queueChild(h,a(l.createTextNode(this._stringText))),a.queueChild(h,a(p)),u.precacheNode(this,f),this._closingComment=p,h}var d=c(this._stringText);return t.renderToStaticMarkup?d:"<!--"+o+"-->"+d+"<!--"+s+"-->"},receiveComponent:function(t,e){if(t!==this._currentElement){this._currentElement=t;var n=""+t;if(n!==this._stringText){this._stringText=n;var r=this.getHostNode();o.replaceDelimitedText(r[0],r[1],n)}}},getHostNode:function(){var t=this._commentNodes;if(t)return t;if(!this._closingComment)for(var e=u.getNodeFromInstance(this),n=e.nextSibling;;){if(null==n?r("67",this._domID):void 0,8===n.nodeType&&" /react-text "===n.nodeValue){this._closingComment=n;break}n=n.nextSibling}return t=[this._hostNode,this._closingComment],this._commentNodes=t,t},unmountComponent:function(){this._closingComment=null,this._commentNodes=null,u.uncacheNode(this)}}),t.exports=s},function(t,e,n){"use strict";function r(){this._rootNodeID&&l.updateWrapper(this)}function i(t){var e=this._currentElement.props,n=u.executeOnChange(e,t);return s.asap(r,this),n}var o=n(2),a=n(3),u=n(85),c=n(4),s=n(11),l=(n(0),n(1),{getHostProps:function(t,e){null!=e.dangerouslySetInnerHTML?o("91"):void 0;var n=a({},e,{value:void 0,defaultValue:void 0,children:""+t._wrapperState.initialValue,onChange:t._wrapperState.onChange});return n},mountWrapper:function(t,e){var n=u.getValue(e),r=n;if(null==n){var a=e.defaultValue,c=e.children;null!=c&&(null!=a?o("92"):void 0,Array.isArray(c)&&(c.length<=1?void 0:o("93"),c=c[0]),a=""+c),null==a&&(a=""),r=a}t._wrapperState={initialValue:""+r,listeners:null,onChange:i.bind(t)}},updateWrapper:function(t){var e=t._currentElement.props,n=c.getNodeFromInstance(t),r=u.getValue(e);if(null!=r){var i=""+r;i!==n.value&&(n.value=i),null==e.defaultValue&&(n.defaultValue=i)}null!=e.defaultValue&&(n.defaultValue=e.defaultValue)},postMountWrapper:function(t){var e=c.getNodeFromInstance(t),n=e.textContent;
n===t._wrapperState.initialValue&&(e.value=n)}});t.exports=l},function(t,e,n){"use strict";function r(t,e){"_hostNode"in t?void 0:c("33"),"_hostNode"in e?void 0:c("33");for(var n=0,r=t;r;r=r._hostParent)n++;for(var i=0,o=e;o;o=o._hostParent)i++;for(;n-i>0;)t=t._hostParent,n--;for(;i-n>0;)e=e._hostParent,i--;for(var a=n;a--;){if(t===e)return t;t=t._hostParent,e=e._hostParent}return null}function i(t,e){"_hostNode"in t?void 0:c("35"),"_hostNode"in e?void 0:c("35");for(;e;){if(e===t)return!0;e=e._hostParent}return!1}function o(t){return"_hostNode"in t?void 0:c("36"),t._hostParent}function a(t,e,n){for(var r=[];t;)r.push(t),t=t._hostParent;var i;for(i=r.length;i-- >0;)e(r[i],"captured",n);for(i=0;i<r.length;i++)e(r[i],"bubbled",n)}function u(t,e,n,i,o){for(var a=t&&e?r(t,e):null,u=[];t&&t!==a;)u.push(t),t=t._hostParent;for(var c=[];e&&e!==a;)c.push(e),e=e._hostParent;var s;for(s=0;s<u.length;s++)n(u[s],"bubbled",i);for(s=c.length;s-- >0;)n(c[s],"captured",o)}var c=n(2);n(0);t.exports={isAncestor:i,getLowestCommonAncestor:r,getParentInstance:o,traverseTwoPhase:a,traverseEnterLeave:u}},function(t,e,n){"use strict";function r(){this.reinitializeTransaction()}var i=n(3),o=n(11),a=n(53),u=n(8),c={initialize:u,close:function(){p.isBatchingUpdates=!1}},s={initialize:u,close:o.flushBatchedUpdates.bind(o)},l=[s,c];i(r.prototype,a,{getTransactionWrappers:function(){return l}});var f=new r,p={isBatchingUpdates:!1,batchedUpdates:function(t,e,n,r,i,o){var a=p.isBatchingUpdates;return p.isBatchingUpdates=!0,a?t(e,n,r,i,o):f.perform(t,null,e,n,r,i,o)}};t.exports=p},function(t,e,n){"use strict";function r(){C||(C=!0,y.EventEmitter.injectReactEventListener(m),y.EventPluginHub.injectEventPluginOrder(u),y.EventPluginUtils.injectComponentTree(p),y.EventPluginUtils.injectTreeTraversal(d),y.EventPluginHub.injectEventPluginsByName({SimpleEventPlugin:w,EnterLeaveEventPlugin:c,ChangeEventPlugin:a,SelectEventPlugin:x,BeforeInputEventPlugin:o}),y.HostComponent.injectGenericComponentClass(f),y.HostComponent.injectTextComponentClass(v),y.DOMProperty.injectDOMPropertyConfig(i),y.DOMProperty.injectDOMPropertyConfig(s),y.DOMProperty.injectDOMPropertyConfig(b),y.EmptyComponent.injectEmptyComponentFactory(function(t){return new h(t)}),y.Updates.injectReconcileTransaction(_),y.Updates.injectBatchingStrategy(g),y.Component.injectEnvironment(l))}var i=n(331),o=n(333),a=n(335),u=n(337),c=n(338),s=n(341),l=n(343),f=n(346),p=n(4),h=n(348),d=n(356),v=n(354),g=n(357),m=n(361),y=n(362),_=n(367),b=n(372),x=n(373),w=n(374),C=!1;t.exports={inject:r}},function(t,e,n){"use strict";var r="function"==typeof Symbol&&Symbol.for&&Symbol.for("react.element")||60103;t.exports=r},function(t,e,n){"use strict";function r(t){i.enqueueEvents(t),i.processEventQueue(!1)}var i=n(22),o={handleTopLevel:function(t,e,n,o){var a=i.extractEvents(t,e,n,o);r(a)}};t.exports=o},function(t,e,n){"use strict";function r(t){for(;t._hostParent;)t=t._hostParent;var e=f.getNodeFromInstance(t),n=e.parentNode;return f.getClosestInstanceFromNode(n)}function i(t,e){this.topLevelType=t,this.nativeEvent=e,this.ancestors=[]}function o(t){var e=h(t.nativeEvent),n=f.getClosestInstanceFromNode(e),i=n;do t.ancestors.push(i),i=i&&r(i);while(i);for(var o=0;o<t.ancestors.length;o++)n=t.ancestors[o],v._handleTopLevel(t.topLevelType,n,t.nativeEvent,h(t.nativeEvent))}function a(t){var e=d(window);t(e)}var u=n(3),c=n(150),s=n(6),l=n(17),f=n(4),p=n(11),h=n(93),d=n(324);u(i.prototype,{destructor:function(){this.topLevelType=null,this.nativeEvent=null,this.ancestors.length=0}}),l.addPoolingTo(i,l.twoArgumentPooler);var v={_enabled:!0,_handleTopLevel:null,WINDOW_HANDLE:s.canUseDOM?window:null,setHandleTopLevel:function(t){v._handleTopLevel=t},setEnabled:function(t){v._enabled=!!t},isEnabled:function(){return v._enabled},trapBubbledEvent:function(t,e,n){return n?c.listen(n,e,v.dispatchEvent.bind(null,t)):null},trapCapturedEvent:function(t,e,n){return n?c.capture(n,e,v.dispatchEvent.bind(null,t)):null},monitorScrollValue:function(t){var e=a.bind(null,t);c.listen(window,"scroll",e)},dispatchEvent:function(t,e){if(v._enabled){var n=i.getPooled(t,e);try{p.batchedUpdates(o,n)}finally{i.release(n)}}}};t.exports=v},function(t,e,n){"use strict";var r=n(21),i=n(22),o=n(50),a=n(86),u=n(159),c=n(51),s=n(161),l=n(11),f={Component:a.injection,DOMProperty:r.injection,EmptyComponent:u.injection,EventPluginHub:i.injection,EventPluginUtils:o.injection,EventEmitter:c.injection,HostComponent:s.injection,Updates:l.injection};t.exports=f},function(t,e,n){"use strict";var r=n(385),i=/\/?>/,o=/^<\!\-\-/,a={CHECKSUM_ATTR_NAME:"data-react-checksum",addChecksumToMarkup:function(t){var e=r(t);return o.test(t)?t:t.replace(i," "+a.CHECKSUM_ATTR_NAME+'="'+e+'"$&')},canReuseMarkup:function(t,e){var n=e.getAttribute(a.CHECKSUM_ATTR_NAME);n=n&&parseInt(n,10);var i=r(t);return i===n}};t.exports=a},function(t,e,n){"use strict";function r(t,e,n){return{type:"INSERT_MARKUP",content:t,fromIndex:null,fromNode:null,toIndex:n,afterNode:e}}function i(t,e,n){return{type:"MOVE_EXISTING",content:null,fromIndex:t._mountIndex,fromNode:p.getHostNode(t),toIndex:n,afterNode:e}}function o(t,e){return{type:"REMOVE_NODE",content:null,fromIndex:t._mountIndex,fromNode:e,toIndex:null,afterNode:null}}function a(t){return{type:"SET_MARKUP",content:t,fromIndex:null,fromNode:null,toIndex:null,afterNode:null}}function u(t){return{type:"TEXT_CONTENT",content:t,fromIndex:null,fromNode:null,toIndex:null,afterNode:null}}function c(t,e){return e&&(t=t||[],t.push(e)),t}function s(t,e){f.processChildrenUpdates(t,e)}var l=n(2),f=n(86),p=(n(40),n(9),n(15),n(24)),h=n(342),d=(n(8),n(388)),v=(n(0),{Mixin:{_reconcilerInstantiateChildren:function(t,e,n){return h.instantiateChildren(t,e,n)},_reconcilerUpdateChildren:function(t,e,n,r,i,o){var a,u=0;return a=d(e,u),h.updateChildren(t,a,n,r,i,this,this._hostContainerInfo,o,u),a},mountChildren:function(t,e,n){var r=this._reconcilerInstantiateChildren(t,e,n);this._renderedChildren=r;var i=[],o=0;for(var a in r)if(r.hasOwnProperty(a)){var u=r[a],c=0,s=p.mountComponent(u,e,this,this._hostContainerInfo,n,c);u._mountIndex=o++,i.push(s)}return i},updateTextContent:function(t){var e=this._renderedChildren;h.unmountChildren(e,!1);for(var n in e)e.hasOwnProperty(n)&&l("118");var r=[u(t)];s(this,r)},updateMarkup:function(t){var e=this._renderedChildren;h.unmountChildren(e,!1);for(var n in e)e.hasOwnProperty(n)&&l("118");var r=[a(t)];s(this,r)},updateChildren:function(t,e,n){this._updateChildren(t,e,n)},_updateChildren:function(t,e,n){var r=this._renderedChildren,i={},o=[],a=this._reconcilerUpdateChildren(r,t,o,i,e,n);if(a||r){var u,l=null,f=0,h=0,d=0,v=null;for(u in a)if(a.hasOwnProperty(u)){var g=r&&r[u],m=a[u];g===m?(l=c(l,this.moveChild(g,v,f,h)),h=Math.max(g._mountIndex,h),g._mountIndex=f):(g&&(h=Math.max(g._mountIndex,h)),l=c(l,this._mountChildAtIndex(m,o[d],v,f,e,n)),d++),f++,v=p.getHostNode(m)}for(u in i)i.hasOwnProperty(u)&&(l=c(l,this._unmountChild(r[u],i[u])));l&&s(this,l),this._renderedChildren=a}},unmountChildren:function(t){var e=this._renderedChildren;h.unmountChildren(e,t),this._renderedChildren=null},moveChild:function(t,e,n,r){if(t._mountIndex<r)return i(t,e,n)},createChild:function(t,e,n){return r(n,e,t._mountIndex)},removeChild:function(t,e){return o(t,e)},_mountChildAtIndex:function(t,e,n,r,i,o){return t._mountIndex=r,this.createChild(t,n,e)},_unmountChild:function(t,e){var n=this.removeChild(t,e);return t._mountIndex=null,n}}});t.exports=v},function(t,e,n){"use strict";function r(t){return!(!t||"function"!=typeof t.attachRef||"function"!=typeof t.detachRef)}var i=n(2),o=(n(0),{addComponentAsRefTo:function(t,e,n){r(n)?void 0:i("119"),n.attachRef(e,t)},removeComponentAsRefFrom:function(t,e,n){r(n)?void 0:i("120");var o=n.getPublicInstance();o&&o.refs[e]===t.getPublicInstance()&&n.detachRef(e)}});t.exports=o},function(t,e,n){"use strict";var r="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED";t.exports=r},function(t,e,n){"use strict";function r(t){this.reinitializeTransaction(),this.renderToStaticMarkup=!1,this.reactMountReady=o.getPooled(null),this.useCreateElement=t}var i=n(3),o=n(155),a=n(17),u=n(51),c=n(162),s=(n(9),n(53)),l=n(88),f={initialize:c.getSelectionInformation,close:c.restoreSelection},p={initialize:function(){var t=u.isEnabled();return u.setEnabled(!1),t},close:function(t){u.setEnabled(t)}},h={initialize:function(){this.reactMountReady.reset()},close:function(){this.reactMountReady.notifyAll()}},d=[f,p,h],v={getTransactionWrappers:function(){return d},getReactMountReady:function(){return this.reactMountReady},getUpdateQueue:function(){return l},checkpoint:function(){return this.reactMountReady.checkpoint()},rollback:function(t){this.reactMountReady.rollback(t)},destructor:function(){o.release(this.reactMountReady),this.reactMountReady=null}};i(r.prototype,s,v),a.addPoolingTo(r),t.exports=r},function(t,e,n){"use strict";function r(t,e,n){"function"==typeof t?t(e.getPublicInstance()):o.addComponentAsRefTo(e,t,n)}function i(t,e,n){"function"==typeof t?t(null):o.removeComponentAsRefFrom(e,t,n)}var o=n(365),a={};a.attachRefs=function(t,e){if(null!==e&&"object"==typeof e){var n=e.ref;null!=n&&r(n,t,e._owner)}},a.shouldUpdateRefs=function(t,e){var n=null,r=null;null!==t&&"object"==typeof t&&(n=t.ref,r=t._owner);var i=null,o=null;return null!==e&&"object"==typeof e&&(i=e.ref,o=e._owner),n!==i||"string"==typeof i&&o!==r},a.detachRefs=function(t,e){if(null!==e&&"object"==typeof e){var n=e.ref;null!=n&&i(n,t,e._owner)}},t.exports=a},function(t,e,n){"use strict";function r(t){this.reinitializeTransaction(),this.renderToStaticMarkup=t,this.useCreateElement=!1,this.updateQueue=new u(this)}var i=n(3),o=n(17),a=n(53),u=(n(9),n(370)),c=[],s={enqueue:function(){}},l={getTransactionWrappers:function(){return c},getReactMountReady:function(){return s},getUpdateQueue:function(){return this.updateQueue},destructor:function(){},checkpoint:function(){},rollback:function(){}};i(r.prototype,a,l),o.addPoolingTo(r),t.exports=r},function(t,e,n){"use strict";function r(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function i(t,e){}var o=n(88),a=(n(1),function(){function t(e){r(this,t),this.transaction=e}return t.prototype.isMounted=function(t){return!1},t.prototype.enqueueCallback=function(t,e,n){this.transaction.isInTransaction()&&o.enqueueCallback(t,e,n)},t.prototype.enqueueForceUpdate=function(t){this.transaction.isInTransaction()?o.enqueueForceUpdate(t):i(t,"forceUpdate")},t.prototype.enqueueReplaceState=function(t,e){this.transaction.isInTransaction()?o.enqueueReplaceState(t,e):i(t,"replaceState")},t.prototype.enqueueSetState=function(t,e){this.transaction.isInTransaction()?o.enqueueSetState(t,e):i(t,"setState")},t}());t.exports=a},function(t,e,n){"use strict";t.exports="15.4.2"},function(t,e,n){"use strict";var r={xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace"},i={accentHeight:"accent-height",accumulate:0,additive:0,alignmentBaseline:"alignment-baseline",allowReorder:"allowReorder",alphabetic:0,amplitude:0,arabicForm:"arabic-form",ascent:0,attributeName:"attributeName",attributeType:"attributeType",autoReverse:"autoReverse",azimuth:0,baseFrequency:"baseFrequency",baseProfile:"baseProfile",baselineShift:"baseline-shift",bbox:0,begin:0,bias:0,by:0,calcMode:"calcMode",capHeight:"cap-height",clip:0,clipPath:"clip-path",clipRule:"clip-rule",clipPathUnits:"clipPathUnits",colorInterpolation:"color-interpolation",colorInterpolationFilters:"color-interpolation-filters",colorProfile:"color-profile",colorRendering:"color-rendering",contentScriptType:"contentScriptType",contentStyleType:"contentStyleType",cursor:0,cx:0,cy:0,d:0,decelerate:0,descent:0,diffuseConstant:"diffuseConstant",direction:0,display:0,divisor:0,dominantBaseline:"dominant-baseline",dur:0,dx:0,dy:0,edgeMode:"edgeMode",elevation:0,enableBackground:"enable-background",end:0,exponent:0,externalResourcesRequired:"externalResourcesRequired",fill:0,fillOpacity:"fill-opacity",fillRule:"fill-rule",filter:0,filterRes:"filterRes",filterUnits:"filterUnits",floodColor:"flood-color",floodOpacity:"flood-opacity",focusable:0,fontFamily:"font-family",fontSize:"font-size",fontSizeAdjust:"font-size-adjust",fontStretch:"font-stretch",fontStyle:"font-style",fontVariant:"font-variant",fontWeight:"font-weight",format:0,from:0,fx:0,fy:0,g1:0,g2:0,glyphName:"glyph-name",glyphOrientationHorizontal:"glyph-orientation-horizontal",glyphOrientationVertical:"glyph-orientation-vertical",glyphRef:"glyphRef",gradientTransform:"gradientTransform",gradientUnits:"gradientUnits",hanging:0,horizAdvX:"horiz-adv-x",horizOriginX:"horiz-origin-x",ideographic:0,imageRendering:"image-rendering",in:0,in2:0,intercept:0,k:0,k1:0,k2:0,k3:0,k4:0,kernelMatrix:"kernelMatrix",kernelUnitLength:"kernelUnitLength",kerning:0,keyPoints:"keyPoints",keySplines:"keySplines",keyTimes:"keyTimes",lengthAdjust:"lengthAdjust",letterSpacing:"letter-spacing",lightingColor:"lighting-color",limitingConeAngle:"limitingConeAngle",local:0,markerEnd:"marker-end",markerMid:"marker-mid",markerStart:"marker-start",markerHeight:"markerHeight",markerUnits:"markerUnits",markerWidth:"markerWidth",mask:0,maskContentUnits:"maskContentUnits",maskUnits:"maskUnits",mathematical:0,mode:0,numOctaves:"numOctaves",offset:0,opacity:0,operator:0,order:0,orient:0,orientation:0,origin:0,overflow:0,overlinePosition:"overline-position",overlineThickness:"overline-thickness",paintOrder:"paint-order",panose1:"panose-1",pathLength:"pathLength",patternContentUnits:"patternContentUnits",patternTransform:"patternTransform",patternUnits:"patternUnits",pointerEvents:"pointer-events",points:0,pointsAtX:"pointsAtX",pointsAtY:"pointsAtY",pointsAtZ:"pointsAtZ",preserveAlpha:"preserveAlpha",preserveAspectRatio:"preserveAspectRatio",primitiveUnits:"primitiveUnits",r:0,radius:0,refX:"refX",refY:"refY",renderingIntent:"rendering-intent",repeatCount:"repeatCount",repeatDur:"repeatDur",requiredExtensions:"requiredExtensions",requiredFeatures:"requiredFeatures",restart:0,result:0,rotate:0,rx:0,ry:0,scale:0,seed:0,shapeRendering:"shape-rendering",slope:0,spacing:0,specularConstant:"specularConstant",specularExponent:"specularExponent",speed:0,spreadMethod:"spreadMethod",startOffset:"startOffset",stdDeviation:"stdDeviation",stemh:0,stemv:0,stitchTiles:"stitchTiles",stopColor:"stop-color",stopOpacity:"stop-opacity",strikethroughPosition:"strikethrough-position",strikethroughThickness:"strikethrough-thickness",string:0,stroke:0,strokeDasharray:"stroke-dasharray",strokeDashoffset:"stroke-dashoffset",strokeLinecap:"stroke-linecap",strokeLinejoin:"stroke-linejoin",strokeMiterlimit:"stroke-miterlimit",strokeOpacity:"stroke-opacity",strokeWidth:"stroke-width",surfaceScale:"surfaceScale",systemLanguage:"systemLanguage",tableValues:"tableValues",targetX:"targetX",targetY:"targetY",textAnchor:"text-anchor",textDecoration:"text-decoration",textRendering:"text-rendering",textLength:"textLength",to:0,transform:0,u1:0,u2:0,underlinePosition:"underline-position",underlineThickness:"underline-thickness",unicode:0,unicodeBidi:"unicode-bidi",unicodeRange:"unicode-range",unitsPerEm:"units-per-em",vAlphabetic:"v-alphabetic",vHanging:"v-hanging",vIdeographic:"v-ideographic",vMathematical:"v-mathematical",values:0,vectorEffect:"vector-effect",version:0,vertAdvY:"vert-adv-y",vertOriginX:"vert-origin-x",vertOriginY:"vert-origin-y",viewBox:"viewBox",viewTarget:"viewTarget",visibility:0,widths:0,wordSpacing:"word-spacing",writingMode:"writing-mode",x:0,xHeight:"x-height",x1:0,x2:0,xChannelSelector:"xChannelSelector",xlinkActuate:"xlink:actuate",xlinkArcrole:"xlink:arcrole",xlinkHref:"xlink:href",xlinkRole:"xlink:role",xlinkShow:"xlink:show",xlinkTitle:"xlink:title",xlinkType:"xlink:type",xmlBase:"xml:base",xmlns:0,xmlnsXlink:"xmlns:xlink",xmlLang:"xml:lang",xmlSpace:"xml:space",y:0,y1:0,y2:0,yChannelSelector:"yChannelSelector",z:0,zoomAndPan:"zoomAndPan"},o={Properties:{},DOMAttributeNamespaces:{xlinkActuate:r.xlink,xlinkArcrole:r.xlink,xlinkHref:r.xlink,xlinkRole:r.xlink,xlinkShow:r.xlink,xlinkTitle:r.xlink,xlinkType:r.xlink,xmlBase:r.xml,xmlLang:r.xml,xmlSpace:r.xml},DOMAttributeNames:{}};Object.keys(i).forEach(function(t){o.Properties[t]=0,i[t]&&(o.DOMAttributeNames[t]=i[t])}),t.exports=o},function(t,e,n){"use strict";function r(t){if("selectionStart"in t&&c.hasSelectionCapabilities(t))return{start:t.selectionStart,end:t.selectionEnd};if(window.getSelection){var e=window.getSelection();return{anchorNode:e.anchorNode,anchorOffset:e.anchorOffset,focusNode:e.focusNode,focusOffset:e.focusOffset}}if(document.selection){var n=document.selection.createRange();return{parentElement:n.parentElement(),text:n.text,top:n.boundingTop,left:n.boundingLeft}}}function i(t,e){if(y||null==v||v!==l())return null;var n=r(v);if(!m||!p(m,n)){m=n;var i=s.getPooled(d.select,g,t,e);return i.type="select",i.target=v,o.accumulateTwoPhaseDispatches(i),i}return null}var o=n(23),a=n(6),u=n(4),c=n(162),s=n(14),l=n(152),f=n(170),p=n(80),h=a.canUseDOM&&"documentMode"in document&&document.documentMode<=11,d={select:{phasedRegistrationNames:{bubbled:"onSelect",captured:"onSelectCapture"},dependencies:["topBlur","topContextMenu","topFocus","topKeyDown","topKeyUp","topMouseDown","topMouseUp","topSelectionChange"]}},v=null,g=null,m=null,y=!1,_=!1,b={eventTypes:d,extractEvents:function(t,e,n,r){if(!_)return null;var o=e?u.getNodeFromInstance(e):window;switch(t){case"topFocus":(f(o)||"true"===o.contentEditable)&&(v=o,g=e,m=null);break;case"topBlur":v=null,g=null,m=null;break;case"topMouseDown":y=!0;break;case"topContextMenu":case"topMouseUp":return y=!1,i(n,r);case"topSelectionChange":if(h)break;case"topKeyDown":case"topKeyUp":return i(n,r)}return null},didPutListener:function(t,e,n){"onSelect"===e&&(_=!0)}};t.exports=b},function(t,e,n){"use strict";function r(t){return"."+t._rootNodeID}function i(t){return"button"===t||"input"===t||"select"===t||"textarea"===t}var o=n(2),a=n(150),u=n(23),c=n(4),s=n(375),l=n(376),f=n(14),p=n(379),h=n(381),d=n(52),v=n(378),g=n(382),m=n(383),y=n(25),_=n(384),b=n(8),x=n(91),w=(n(0),{}),C={};["abort","animationEnd","animationIteration","animationStart","blur","canPlay","canPlayThrough","click","contextMenu","copy","cut","doubleClick","drag","dragEnd","dragEnter","dragExit","dragLeave","dragOver","dragStart","drop","durationChange","emptied","encrypted","ended","error","focus","input","invalid","keyDown","keyPress","keyUp","load","loadedData","loadedMetadata","loadStart","mouseDown","mouseMove","mouseOut","mouseOver","mouseUp","paste","pause","play","playing","progress","rateChange","reset","scroll","seeked","seeking","stalled","submit","suspend","timeUpdate","touchCancel","touchEnd","touchMove","touchStart","transitionEnd","volumeChange","waiting","wheel"].forEach(function(t){var e=t[0].toUpperCase()+t.slice(1),n="on"+e,r="top"+e,i={phasedRegistrationNames:{bubbled:n,captured:n+"Capture"},dependencies:[r]};w[t]=i,C[r]=i});var M={},k={eventTypes:w,extractEvents:function(t,e,n,r){var i=C[t];if(!i)return null;var a;switch(t){case"topAbort":case"topCanPlay":case"topCanPlayThrough":case"topDurationChange":case"topEmptied":case"topEncrypted":case"topEnded":case"topError":case"topInput":case"topInvalid":case"topLoad":case"topLoadedData":case"topLoadedMetadata":case"topLoadStart":case"topPause":case"topPlay":case"topPlaying":case"topProgress":case"topRateChange":case"topReset":case"topSeeked":case"topSeeking":case"topStalled":case"topSubmit":case"topSuspend":case"topTimeUpdate":case"topVolumeChange":case"topWaiting":a=f;break;case"topKeyPress":if(0===x(n))return null;case"topKeyDown":case"topKeyUp":a=h;break;case"topBlur":case"topFocus":a=p;break;case"topClick":if(2===n.button)return null;case"topDoubleClick":case"topMouseDown":case"topMouseMove":case"topMouseUp":case"topMouseOut":case"topMouseOver":case"topContextMenu":a=d;break;case"topDrag":case"topDragEnd":case"topDragEnter":case"topDragExit":case"topDragLeave":case"topDragOver":case"topDragStart":case"topDrop":a=v;break;case"topTouchCancel":case"topTouchEnd":case"topTouchMove":case"topTouchStart":a=g;break;case"topAnimationEnd":case"topAnimationIteration":case"topAnimationStart":a=s;break;case"topTransitionEnd":a=m;break;case"topScroll":a=y;break;case"topWheel":a=_;break;case"topCopy":case"topCut":case"topPaste":a=l}a?void 0:o("86",t);var c=a.getPooled(i,e,n,r);return u.accumulateTwoPhaseDispatches(c),c},didPutListener:function(t,e,n){if("onClick"===e&&!i(t._tag)){var o=r(t),u=c.getNodeFromInstance(t);M[o]||(M[o]=a.listen(u,"click",b))}},willDeleteListener:function(t,e){if("onClick"===e&&!i(t._tag)){var n=r(t);M[n].remove(),delete M[n]}}};t.exports=k},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={animationName:null,elapsedTime:null,pseudoElement:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={clipboardData:function(t){return"clipboardData"in t?t.clipboardData:window.clipboardData}};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={data:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(52),o={dataTransfer:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o={relatedTarget:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={data:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o=n(91),a=n(389),u=n(92),c={key:a,location:null,ctrlKey:null,shiftKey:null,altKey:null,metaKey:null,repeat:null,locale:null,getModifierState:u,charCode:function(t){return"keypress"===t.type?o(t):0},keyCode:function(t){return"keydown"===t.type||"keyup"===t.type?t.keyCode:0},which:function(t){return"keypress"===t.type?o(t):"keydown"===t.type||"keyup"===t.type?t.keyCode:0}};i.augmentClass(r,c),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o=n(92),a={touches:null,targetTouches:null,changedTouches:null,altKey:null,metaKey:null,ctrlKey:null,shiftKey:null,getModifierState:o};i.augmentClass(r,a),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={propertyName:null,elapsedTime:null,pseudoElement:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(52),o={deltaX:function(t){return"deltaX"in t?t.deltaX:"wheelDeltaX"in t?-t.wheelDeltaX:0},deltaY:function(t){return"deltaY"in t?t.deltaY:"wheelDeltaY"in t?-t.wheelDeltaY:"wheelDelta"in t?-t.wheelDelta:0},deltaZ:null,deltaMode:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t){for(var e=1,n=0,r=0,o=t.length,a=o&-4;r<a;){for(var u=Math.min(r+4096,a);r<u;r+=4)n+=(e+=t.charCodeAt(r))+(e+=t.charCodeAt(r+1))+(e+=t.charCodeAt(r+2))+(e+=t.charCodeAt(r+3));e%=i,n%=i}for(;r<o;r++)n+=e+=t.charCodeAt(r);return e%=i,n%=i,e|n<<16}var i=65521;t.exports=r},function(t,e,n){"use strict";function r(t,e,n){var r=null==e||"boolean"==typeof e||""===e;if(r)return"";var i=isNaN(e);if(i||0===e||o.hasOwnProperty(t)&&o[t])return""+e;if("string"==typeof e){e=e.trim()}return e+"px"}var i=n(154),o=(n(1),i.isUnitlessNumber);t.exports=r},function(t,e,n){"use strict";function r(t){if(null==t)return null;if(1===t.nodeType)return t;var e=a.get(t);return e?(e=u(e),e?o.getNodeFromInstance(e):null):void("function"==typeof t.render?i("44"):i("45",Object.keys(t)))}var i=n(2),o=(n(15),n(4)),a=n(40),u=n(167);n(0),n(1);t.exports=r},function(t,e,n){"use strict";(function(e){function r(t,e,n,r){if(t&&"object"==typeof t){var i=t,o=void 0===i[n];o&&null!=e&&(i[n]=e)}}function i(t,e){if(null==t)return t;var n={};return o(t,r,n),n}var o=(n(84),n(172));n(1);"undefined"!=typeof e&&e.env,1,t.exports=i}).call(e,n(153))},function(t,e,n){"use strict";function r(t){if(t.key){var e=o[t.key]||t.key;if("Unidentified"!==e)return e}if("keypress"===t.type){var n=i(t);return 13===n?"Enter":String.fromCharCode(n)}return"keydown"===t.type||"keyup"===t.type?a[t.keyCode]||"Unidentified":""}var i=n(91),o={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},a={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"};t.exports=r},function(t,e,n){"use strict";function r(t){var e=t&&(i&&t[i]||t[o]);if("function"==typeof e)return e}var i="function"==typeof Symbol&&Symbol.iterator,o="@@iterator";t.exports=r},function(t,e,n){"use strict";function r(){return i++}var i=1;t.exports=r},function(t,e,n){"use strict";function r(t){for(;t&&t.firstChild;)t=t.firstChild;return t}function i(t){for(;t;){if(t.nextSibling)return t.nextSibling;t=t.parentNode}}function o(t,e){for(var n=r(t),o=0,a=0;n;){if(3===n.nodeType){if(a=o+n.textContent.length,o<=e&&a>=e)return{node:n,offset:e-o};o=a}n=r(i(n))}}t.exports=o},function(t,e,n){"use strict";function r(t,e){var n={};return n[t.toLowerCase()]=e.toLowerCase(),n["Webkit"+t]="webkit"+e,n["Moz"+t]="moz"+e,n["ms"+t]="MS"+e,n["O"+t]="o"+e.toLowerCase(),n}function i(t){if(u[t])return u[t];if(!a[t])return t;var e=a[t];for(var n in e)if(e.hasOwnProperty(n)&&n in c)return u[t]=e[n];return""}var o=n(6),a={animationend:r("Animation","AnimationEnd"),animationiteration:r("Animation","AnimationIteration"),animationstart:r("Animation","AnimationStart"),transitionend:r("Transition","TransitionEnd")},u={},c={};o.canUseDOM&&(c=document.createElement("div").style,"AnimationEvent"in window||(delete a.animationend.animation,delete a.animationiteration.animation,delete a.animationstart.animation),"TransitionEvent"in window||delete a.transitionend.transition),t.exports=i},function(t,e,n){"use strict";function r(t){return'"'+i(t)+'"'}var i=n(54);t.exports=r},function(t,e,n){"use strict";var r=n(163);t.exports=r.renderSubtreeIntoContainer},function(t,e,n){"use strict";function r(t,e){var n=l.extractSingleTouch(e);return n?n[t.page]:t.page in e?e[t.page]:e[t.client]+f[t.envScroll]}function i(t,e){var n=r(b.x,e),i=r(b.y,e);return Math.pow(Math.pow(n-t.x,2)+Math.pow(i-t.y,2),.5)}function o(t){return{tapMoveThreshold:g,ignoreMouseThreshold:m,eventTypes:C,extractEvents:function(e,n,o,a){if(!h(e)&&!d(e))return null;if(v(e))_=M();else if(t(_,M()))return null;var u=null,l=i(y,o);return d(e)&&l<g&&(u=s.getPooled(C.touchTap,n,o,a)),h(e)?(y.x=r(b.x,o),y.y=r(b.y,o)):d(e)&&(y.x=0,y.y=0),c.accumulateTwoPhaseDispatches(u),u}}}var a=n(339),u=n(50),c=n(23),s=n(25),l=n(397),f=n(89),p=n(329),h=(a.topLevelTypes,u.isStartish),d=u.isEndish,v=function(t){var e=["topTouchCancel","topTouchEnd","topTouchStart","topTouchMove"];return e.indexOf(t)>=0},g=10,m=750,y={x:null,y:null},_=null,b={x:{page:"pageX",client:"clientX",envScroll:"currentPageScrollLeft"},y:{page:"pageY",client:"clientY",envScroll:"currentPageScrollTop"}},x=["topTouchStart","topTouchCancel","topTouchEnd","topTouchMove"],w=["topMouseDown","topMouseMove","topMouseUp"].concat(x),C={touchTap:{phasedRegistrationNames:{bubbled:p({onTouchTap:null}),captured:p({onTouchTapCapture:null})},dependencies:w}},M=function(){return Date.now?Date.now:function(){return+new Date}}();t.exports=o},function(t,e){var n={extractSingleTouch:function(t){var e=t.touches,n=t.changedTouches,r=e&&e.length>0,i=n&&n.length>0;return!r&&i?n[0]:r?e[0]:t}};t.exports=n},function(t,e){t.exports=function(t,e){if(t&&e-t<750)return!0}},function(t,e,n){"use strict";function r(t){var e=/[=:]/g,n={"=":"=0",":":"=2"},r=(""+t).replace(e,function(t){return n[t]});return"$"+r}function i(t){var e=/(=0|=2)/g,n={"=0":"=","=2":":"},r="."===t[0]&&"$"===t[1]?t.substring(2):t.substring(1);return(""+r).replace(e,function(t){return n[t]})}var o={escape:r,unescape:i};t.exports=o},function(t,e,n){"use strict";var r=n(28),i=(n(0),function(t){var e=this;if(e.instancePool.length){var n=e.instancePool.pop();return e.call(n,t),n}return new e(t)}),o=function(t,e){var n=this;if(n.instancePool.length){var r=n.instancePool.pop();return n.call(r,t,e),r}return new n(t,e)},a=function(t,e,n){var r=this;if(r.instancePool.length){var i=r.instancePool.pop();return r.call(i,t,e,n),i}return new r(t,e,n)},u=function(t,e,n,r){var i=this;if(i.instancePool.length){var o=i.instancePool.pop();return i.call(o,t,e,n,r),o}return new i(t,e,n,r)},c=function(t){var e=this;t instanceof e?void 0:r("25"),t.destructor(),e.instancePool.length<e.poolSize&&e.instancePool.push(t)},s=10,l=i,f=function(t,e){var n=t;return n.instancePool=[],n.getPooled=e||l,n.poolSize||(n.poolSize=s),n.release=c,n},p={addPoolingTo:f,oneArgumentPooler:i,twoArgumentPooler:o,threeArgumentPooler:a,fourArgumentPooler:u};t.exports=p},function(t,e,n){"use strict";function r(t){return(""+t).replace(b,"$&/")}function i(t,e){this.func=t,this.context=e,this.count=0}function o(t,e,n){var r=t.func,i=t.context;r.call(i,e,t.count++)}function a(t,e,n){if(null==t)return t;var r=i.getPooled(e,n);m(t,o,r),i.release(r)}function u(t,e,n,r){this.result=t,this.keyPrefix=e,this.func=n,this.context=r,this.count=0}function c(t,e,n){var i=t.result,o=t.keyPrefix,a=t.func,u=t.context,c=a.call(u,e,t.count++);Array.isArray(c)?s(c,i,n,g.thatReturnsArgument):null!=c&&(v.isValidElement(c)&&(c=v.cloneAndReplaceKey(c,o+(!c.key||e&&e.key===c.key?"":r(c.key)+"/")+n)),i.push(c))}function s(t,e,n,i,o){var a="";null!=n&&(a=r(n)+"/");var s=u.getPooled(e,a,i,o);m(t,c,s),u.release(s)}function l(t,e,n){if(null==t)return t;var r=[];return s(t,r,null,e,n),r}function f(t,e,n){return null}function p(t,e){return m(t,f,null)}function h(t){var e=[];return s(t,e,null,g.thatReturnsArgument),e}var d=n(400),v=n(27),g=n(8),m=n(409),y=d.twoArgumentPooler,_=d.fourArgumentPooler,b=/\/+/g;i.prototype.destructor=function(){this.func=null,this.context=null,this.count=0},d.addPoolingTo(i,y),u.prototype.destructor=function(){this.result=null,this.keyPrefix=null,this.func=null,this.context=null,this.count=0},d.addPoolingTo(u,_);var x={forEach:a,map:l,mapIntoWithKeyPrefixInternal:s,count:p,toArray:h};t.exports=x},function(t,e,n){"use strict";function r(t){return t}function i(t,e){var n=b.hasOwnProperty(e)?b[e]:null;w.hasOwnProperty(e)&&("OVERRIDE_BASE"!==n?p("73",e):void 0),t&&("DEFINE_MANY"!==n&&"DEFINE_MANY_MERGED"!==n?p("74",e):void 0)}function o(t,e){if(e){"function"==typeof e?p("75"):void 0,v.isValidElement(e)?p("76"):void 0;var n=t.prototype,r=n.__reactAutoBindPairs;e.hasOwnProperty(y)&&x.mixins(t,e.mixins);for(var o in e)if(e.hasOwnProperty(o)&&o!==y){var a=e[o],u=n.hasOwnProperty(o);if(i(u,o),x.hasOwnProperty(o))x[o](t,a);else{var l=b.hasOwnProperty(o),f="function"==typeof a,h=f&&!l&&!u&&e.autobind!==!1;if(h)r.push(o,a),n[o]=a;else if(u){var d=b[o];!l||"DEFINE_MANY_MERGED"!==d&&"DEFINE_MANY"!==d?p("77",d,o):void 0,"DEFINE_MANY_MERGED"===d?n[o]=c(n[o],a):"DEFINE_MANY"===d&&(n[o]=s(n[o],a))}else n[o]=a}}}else;}function a(t,e){if(e)for(var n in e){var r=e[n];if(e.hasOwnProperty(n)){var i=n in x;i?p("78",n):void 0;var o=n in t;o?p("79",n):void 0,t[n]=r}}}function u(t,e){t&&e&&"object"==typeof t&&"object"==typeof e?void 0:p("80");for(var n in e)e.hasOwnProperty(n)&&(void 0!==t[n]?p("81",n):void 0,t[n]=e[n]);return t}function c(t,e){return function(){var n=t.apply(this,arguments),r=e.apply(this,arguments);if(null==n)return r;if(null==r)return n;var i={};return u(i,n),u(i,r),i}}function s(t,e){return function(){t.apply(this,arguments),e.apply(this,arguments)}}function l(t,e){var n=e.bind(t);return n;
}function f(t){for(var e=t.__reactAutoBindPairs,n=0;n<e.length;n+=2){var r=e[n],i=e[n+1];t[r]=l(t,i)}}var p=n(28),h=n(3),d=n(97),v=n(27),g=(n(175),n(98)),m=n(38),y=(n(0),n(1),"mixins"),_=[],b={mixins:"DEFINE_MANY",statics:"DEFINE_MANY",propTypes:"DEFINE_MANY",contextTypes:"DEFINE_MANY",childContextTypes:"DEFINE_MANY",getDefaultProps:"DEFINE_MANY_MERGED",getInitialState:"DEFINE_MANY_MERGED",getChildContext:"DEFINE_MANY_MERGED",render:"DEFINE_ONCE",componentWillMount:"DEFINE_MANY",componentDidMount:"DEFINE_MANY",componentWillReceiveProps:"DEFINE_MANY",shouldComponentUpdate:"DEFINE_ONCE",componentWillUpdate:"DEFINE_MANY",componentDidUpdate:"DEFINE_MANY",componentWillUnmount:"DEFINE_MANY",updateComponent:"OVERRIDE_BASE"},x={displayName:function(t,e){t.displayName=e},mixins:function(t,e){if(e)for(var n=0;n<e.length;n++)o(t,e[n])},childContextTypes:function(t,e){t.childContextTypes=h({},t.childContextTypes,e)},contextTypes:function(t,e){t.contextTypes=h({},t.contextTypes,e)},getDefaultProps:function(t,e){t.getDefaultProps?t.getDefaultProps=c(t.getDefaultProps,e):t.getDefaultProps=e},propTypes:function(t,e){t.propTypes=h({},t.propTypes,e)},statics:function(t,e){a(t,e)},autobind:function(){}},w={replaceState:function(t,e){this.updater.enqueueReplaceState(this,t),e&&this.updater.enqueueCallback(this,e,"replaceState")},isMounted:function(){return this.updater.isMounted(this)}},C=function(){};h(C.prototype,d.prototype,w);var M={createClass:function(t){var e=r(function(t,n,r){this.__reactAutoBindPairs.length&&f(this),this.props=t,this.context=n,this.refs=m,this.updater=r||g,this.state=null;var i=this.getInitialState?this.getInitialState():null;"object"!=typeof i||Array.isArray(i)?p("82",e.displayName||"ReactCompositeComponent"):void 0,this.state=i});e.prototype=new C,e.prototype.constructor=e,e.prototype.__reactAutoBindPairs=[],_.forEach(o.bind(null,e)),o(e,t),e.getDefaultProps&&(e.defaultProps=e.getDefaultProps()),e.prototype.render?void 0:p("83");for(var n in b)e.prototype[n]||(e.prototype[n]=null);return e},injection:{injectMixin:function(t){_.push(t)}}};t.exports=M},function(t,e,n){"use strict";var r=n(27),i=r.createFactory,o={a:i("a"),abbr:i("abbr"),address:i("address"),area:i("area"),article:i("article"),aside:i("aside"),audio:i("audio"),b:i("b"),base:i("base"),bdi:i("bdi"),bdo:i("bdo"),big:i("big"),blockquote:i("blockquote"),body:i("body"),br:i("br"),button:i("button"),canvas:i("canvas"),caption:i("caption"),cite:i("cite"),code:i("code"),col:i("col"),colgroup:i("colgroup"),data:i("data"),datalist:i("datalist"),dd:i("dd"),del:i("del"),details:i("details"),dfn:i("dfn"),dialog:i("dialog"),div:i("div"),dl:i("dl"),dt:i("dt"),em:i("em"),embed:i("embed"),fieldset:i("fieldset"),figcaption:i("figcaption"),figure:i("figure"),footer:i("footer"),form:i("form"),h1:i("h1"),h2:i("h2"),h3:i("h3"),h4:i("h4"),h5:i("h5"),h6:i("h6"),head:i("head"),header:i("header"),hgroup:i("hgroup"),hr:i("hr"),html:i("html"),i:i("i"),iframe:i("iframe"),img:i("img"),input:i("input"),ins:i("ins"),kbd:i("kbd"),keygen:i("keygen"),label:i("label"),legend:i("legend"),li:i("li"),link:i("link"),main:i("main"),map:i("map"),mark:i("mark"),menu:i("menu"),menuitem:i("menuitem"),meta:i("meta"),meter:i("meter"),nav:i("nav"),noscript:i("noscript"),object:i("object"),ol:i("ol"),optgroup:i("optgroup"),option:i("option"),output:i("output"),p:i("p"),param:i("param"),picture:i("picture"),pre:i("pre"),progress:i("progress"),q:i("q"),rp:i("rp"),rt:i("rt"),ruby:i("ruby"),s:i("s"),samp:i("samp"),script:i("script"),section:i("section"),select:i("select"),small:i("small"),source:i("source"),span:i("span"),strong:i("strong"),style:i("style"),sub:i("sub"),summary:i("summary"),sup:i("sup"),table:i("table"),tbody:i("tbody"),td:i("td"),textarea:i("textarea"),tfoot:i("tfoot"),th:i("th"),thead:i("thead"),time:i("time"),title:i("title"),tr:i("tr"),track:i("track"),u:i("u"),ul:i("ul"),var:i("var"),video:i("video"),wbr:i("wbr"),circle:i("circle"),clipPath:i("clipPath"),defs:i("defs"),ellipse:i("ellipse"),g:i("g"),image:i("image"),line:i("line"),linearGradient:i("linearGradient"),mask:i("mask"),path:i("path"),pattern:i("pattern"),polygon:i("polygon"),polyline:i("polyline"),radialGradient:i("radialGradient"),rect:i("rect"),stop:i("stop"),svg:i("svg"),text:i("text"),tspan:i("tspan")};t.exports=o},function(t,e,n){"use strict";function r(t,e){return t===e?0!==t||1/t===1/e:t!==t&&e!==e}function i(t){this.message=t,this.stack=""}function o(t){function e(e,n,r,o,a,u,c){o=o||E,u=u||r;if(null==n[r]){var s=w[a];return e?new i(null===n[r]?"The "+s+" `"+u+"` is marked as required "+("in `"+o+"`, but its value is `null`."):"The "+s+" `"+u+"` is marked as required in "+("`"+o+"`, but its value is `undefined`.")):null}return t(n,r,o,a,u)}var n=e.bind(null,!1);return n.isRequired=e.bind(null,!0),n}function a(t){function e(e,n,r,o,a,u){var c=e[n],s=y(c);if(s!==t){var l=w[o],f=_(c);return new i("Invalid "+l+" `"+a+"` of type "+("`"+f+"` supplied to `"+r+"`, expected ")+("`"+t+"`."))}return null}return o(e)}function u(){return o(M.thatReturns(null))}function c(t){function e(e,n,r,o,a){if("function"!=typeof t)return new i("Property `"+a+"` of component `"+r+"` has invalid PropType notation inside arrayOf.");var u=e[n];if(!Array.isArray(u)){var c=w[o],s=y(u);return new i("Invalid "+c+" `"+a+"` of type "+("`"+s+"` supplied to `"+r+"`, expected an array."))}for(var l=0;l<u.length;l++){var f=t(u,l,r,o,a+"["+l+"]",C);if(f instanceof Error)return f}return null}return o(e)}function s(){function t(t,e,n,r,o){var a=t[e];if(!x.isValidElement(a)){var u=w[r],c=y(a);return new i("Invalid "+u+" `"+o+"` of type "+("`"+c+"` supplied to `"+n+"`, expected a single ReactElement."))}return null}return o(t)}function l(t){function e(e,n,r,o,a){if(!(e[n]instanceof t)){var u=w[o],c=t.name||E,s=b(e[n]);return new i("Invalid "+u+" `"+a+"` of type "+("`"+s+"` supplied to `"+r+"`, expected ")+("instance of `"+c+"`."))}return null}return o(e)}function f(t){function e(e,n,o,a,u){for(var c=e[n],s=0;s<t.length;s++)if(r(c,t[s]))return null;var l=w[a],f=JSON.stringify(t);return new i("Invalid "+l+" `"+u+"` of value `"+c+"` "+("supplied to `"+o+"`, expected one of "+f+"."))}return Array.isArray(t)?o(e):M.thatReturnsNull}function p(t){function e(e,n,r,o,a){if("function"!=typeof t)return new i("Property `"+a+"` of component `"+r+"` has invalid PropType notation inside objectOf.");var u=e[n],c=y(u);if("object"!==c){var s=w[o];return new i("Invalid "+s+" `"+a+"` of type "+("`"+c+"` supplied to `"+r+"`, expected an object."))}for(var l in u)if(u.hasOwnProperty(l)){var f=t(u,l,r,o,a+"."+l,C);if(f instanceof Error)return f}return null}return o(e)}function h(t){function e(e,n,r,o,a){for(var u=0;u<t.length;u++){var c=t[u];if(null==c(e,n,r,o,a,C))return null}var s=w[o];return new i("Invalid "+s+" `"+a+"` supplied to "+("`"+r+"`."))}return Array.isArray(t)?o(e):M.thatReturnsNull}function d(){function t(t,e,n,r,o){if(!g(t[e])){var a=w[r];return new i("Invalid "+a+" `"+o+"` supplied to "+("`"+n+"`, expected a ReactNode."))}return null}return o(t)}function v(t){function e(e,n,r,o,a){var u=e[n],c=y(u);if("object"!==c){var s=w[o];return new i("Invalid "+s+" `"+a+"` of type `"+c+"` "+("supplied to `"+r+"`, expected `object`."))}for(var l in t){var f=t[l];if(f){var p=f(u,l,r,o,a+"."+l,C);if(p)return p}}return null}return o(e)}function g(t){switch(typeof t){case"number":case"string":case"undefined":return!0;case"boolean":return!t;case"object":if(Array.isArray(t))return t.every(g);if(null===t||x.isValidElement(t))return!0;var e=k(t);if(!e)return!1;var n,r=e.call(t);if(e!==t.entries){for(;!(n=r.next()).done;)if(!g(n.value))return!1}else for(;!(n=r.next()).done;){var i=n.value;if(i&&!g(i[1]))return!1}return!0;default:return!1}}function m(t,e){return"symbol"===t||("Symbol"===e["@@toStringTag"]||"function"==typeof Symbol&&e instanceof Symbol)}function y(t){var e=typeof t;return Array.isArray(t)?"array":t instanceof RegExp?"object":m(e,t)?"symbol":e}function _(t){var e=y(t);if("object"===e){if(t instanceof Date)return"date";if(t instanceof RegExp)return"regexp"}return e}function b(t){return t.constructor&&t.constructor.name?t.constructor.name:E}var x=n(27),w=n(175),C=n(405),M=n(8),k=n(177),E=(n(1),"<<anonymous>>"),T={array:a("array"),bool:a("boolean"),func:a("function"),number:a("number"),object:a("object"),string:a("string"),symbol:a("symbol"),any:u(),arrayOf:c,element:s(),instanceOf:l,node:d(),objectOf:p,oneOf:f,oneOfType:h,shape:v};i.prototype=Error.prototype,t.exports=T},function(t,e,n){"use strict";var r="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED";t.exports=r},function(t,e,n){"use strict";function r(t,e,n){this.props=t,this.context=e,this.refs=c,this.updater=n||u}function i(){}var o=n(3),a=n(97),u=n(98),c=n(38);i.prototype=a.prototype,r.prototype=new i,r.prototype.constructor=r,o(r.prototype,a.prototype),r.prototype.isPureReactComponent=!0,t.exports=r},function(t,e,n){"use strict";t.exports="15.4.2"},function(t,e,n){"use strict";function r(t){return o.isValidElement(t)?void 0:i("143"),t}var i=n(28),o=n(27);n(0);t.exports=r},function(t,e,n){"use strict";function r(t,e){return t&&"object"==typeof t&&null!=t.key?s.escape(t.key):e.toString(36)}function i(t,e,n,o){var p=typeof t;if("undefined"!==p&&"boolean"!==p||(t=null),null===t||"string"===p||"number"===p||"object"===p&&t.$$typeof===u)return n(o,t,""===e?l+r(t,0):e),1;var h,d,v=0,g=""===e?l:e+f;if(Array.isArray(t))for(var m=0;m<t.length;m++)h=t[m],d=g+r(h,m),v+=i(h,d,n,o);else{var y=c(t);if(y){var _,b=y.call(t);if(y!==t.entries)for(var x=0;!(_=b.next()).done;)h=_.value,d=g+r(h,x++),v+=i(h,d,n,o);else for(;!(_=b.next()).done;){var w=_.value;w&&(h=w[1],d=g+s.escape(w[0])+f+r(h,0),v+=i(h,d,n,o))}}else if("object"===p){var C="",M=String(t);a("31","[object Object]"===M?"object with keys {"+Object.keys(t).join(", ")+"}":M,C)}}return v}function o(t,e,n){return null==t?0:i(t,"",e,n)}var a=n(28),u=(n(15),n(174)),c=n(177),s=(n(0),n(399)),l=(n(1),"."),f=":";t.exports=o},function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}var i=n(41),o=r(i),a=n(182),u=r(a),c=n(183),s=r(c),l=n(181),f=r(l),p=n(180),h=r(p),d=n(179),v=r(d);(0,s.default)(),window.SHAP={SimpleListVisualizer:f.default,AdditiveForceVisualizer:h.default,AdditiveForceArrayVisualizer:v.default,React:o.default,ReactDom:u.default}}]);</script>



    
![png](output_328_1.png)
    


This plot is made of many dots. Each dot has three characteristics:

- Vertical location shows what feature it is depicting
- Color shows whether that feature was high or low for that row of the dataset
- Horizontal location shows whether the effect of that value caused a higher or lower prediction.

For example, the point on the extreme left of Feature 2 that corresponds to lower values for "Interest Rate" and it reduces the prediction by 2.0.

Some things you should be able to easily pick out:
- High values of "Feature 2 (Interest Rate)" caused higher predictions, and low values caused low predictions.
- High values of "Feature 13 (Fico Score)" caused lower predictions, and low values caused higher predictions.
- High values of "Feature 0 (Loan Amount)" caused higher predictions, and low values caused a feeble (0.25) decrease in predictions.

If you look for long enough, there's a lot of information in this graph.


```python
X_test.columns.values[0]
```




    'loan_amnt'



<a id="7"></a>
# 7. Conclusion

We applied machine learning methods to predict the probability that a requested loan on LendingClub will charge off. After training and evaluating different models (logistic regression, random forest, and k-nearest neighbors), we found that all models performed similarly according to a cross-validated AUROC score on the training data. We selected logistic regression (with ridge penalty) because it was the fastest model to train, and this model obtained an AUROC score of 0.689 on a test set consisting of the most recent 10% of the loans.

This model, while far from perfect, can provide a somewhat informed prediction of the likelihood that a loan will charge off, using only data available to potential investors before the loan is fully funded.

We also found that, according to linear measures of correlation between the predictors and the response, the most important variables for predicting charge-off are the loan interest rate and term, and the borrower's FICO score and debt-to-income ratio.


```python

```
