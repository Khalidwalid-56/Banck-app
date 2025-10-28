import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression , Ridge ,Lasso
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,r2_score



 

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd



from sklearn.preprocessing import StandardScaler



np.random.seed(42)
n_samples= 10000

# normale transzction
amount = np.random.lognormal(mean=4 , sigma=0.5 , size=n_samples)
time = np.random.choice(np.random.uniform(0, 24, n_samples), size=n_samples)





# Fraudulent transaction (higher amounts , unusual time)

fraud_amount = np.random.lognormal(mean=6 , sigma=1 , size=100)
fraud_time = np.random.choice([np.random.uniform(0,3), np.random.uniform(22,24)], 100)

time = np.concatenate([time, fraud_time])
amount = np.concatenate([amount, fraud_amount])



data = pd.DataFrame({'Amount':amount , 'Time':time})



#Standardize the features

scaler= StandardScaler()

data_scaled = scaler.fit_transform(data)



#Apply Isolation Forest

from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.01, random_state=42)


predictions=  iso_forest.fit_predict(data_scaled)

#Add predictions to the dataframe
data['Anomaly']  = predictions




# Plot result
plt.figure(figsize=(10, 6))
plt.scatter(data[ 'Time'],  data ['Amount'],  c=data ['Anomaly'],  cmap='viridis')

plt.xlabel('Time of Day')
plt.ylabel('Transaction Amount')
plt.title('Credit Card Fraud Detection')
plt.colorbar(label='Anomaly (-1) / Normal (1)')

plt.show()

print(f"Number of detected anomalies: {sum(predictions == -1)}")







