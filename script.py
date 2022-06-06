import pandas as pd

#Read all csv files
data1 = pd.read_csv('0031_M10.csv', 
                    delimiter = ' ', 
                    header=None)
data2 = pd.read_csv('0031_M20.csv', 
                    delimiter = ' ', 
                    header=None)
data3 = pd.read_csv('0031_L20.csv', 
                    delimiter = ' ', 
                    header=None)
data4 = pd.read_csv('0031_L30.csv', 
                    delimiter = ' ', 
                    header=None)
data5 = pd.read_csv('0031_L40.csv', 
                    delimiter = ' ', 
                    header=None)

#add labels 1 for M10 model, 2 for M20 model and so on upto 5 for L40 simulation model
data1['label'] = 1

data2['label'] = 2

data3['label'] = 3

data4['label'] = 4

data5['label'] = 5

#combine all data of all 5 simulation models
data_all = pd.concat([data1, data2, data3, data4, data5])


data_all.reset_index(drop=True, inplace=True)

data_all.columns = ["numberlist","masslist","canlist","diameterlist","dens1list",
                 "freefalllist","xlist","ylist","zlist","magnlist","gravlist",
                 "kturblist","therlist","velxlist","velylist","velzlist",
                 "lenxlist","lenylist","lenzlist","cellnumberlist","energyratio1list",
                 "veldispersion_nontherm1Dlist","soundspeedavglist",
                 "mass_sr2_incgslist","veldispersion_nontherm1D_sr2list",
                 "soundspeedavg_sr2list","alfvenspeedavg_sr2list",
                 "average_density_srvollist","average_density_srmasslist",
                 "radius_sr1list","radius_sr2list","new_radius_sr2list",
                 "new_bondi_accretion1list","new_bondi_accretion2list",
                 "coremag_pressurelist","sr2mag_pressurelist","dens2list",
                 "energyratio2list","label"]


x = data_all.iloc[:,:-1]

y = data_all.label
#Now lets just test logistic regression

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5, random_state=1)

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier()

classifier.fit(x_train,y_train)

predictions=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# Feature scaling
from sklearn.preprocessing import StandardScaler
#
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

cols = ["number","mass","can","diameter","dens1",
                 "freefall","x","y","z","magn","grav",
                 "kturb","ther","velx","vely","velz",
                 "lenx","leny","lenz","cellnumber","energyratio1",
                 "veldispersion_nontherm1D","soundspeedavg",
                 "mass_sr2_incgs","veldispersion_nontherm1D_sr2",
                 "soundspeedavg_sr2","alfvenspeedavg_sr2",
                 "average_density_srvol","average_density_srmass",
                 "radius_sr1","radius_sr2","new_radius_sr2",
                 "new_bondi_accretion1","new_bondi_accretion2",
                 "coremag_pressure","sr2mag_pressure","dens2",
                 "energyratio2"]
x_train_std = pd.DataFrame(x_train_std, columns=cols)
x_test_std = pd.DataFrame(x_test_std, columns=cols)

#
# Training / Test Dataframe
#
from sklearn.ensemble import RandomForestClassifier
#
# Train the mode
#
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train_std, y_train.values.ravel())

importances = clf.feature_importances_
#
# Sort the feature importance in descending order

import numpy as np
#
sorted_indices = np.argsort(importances)[::-1]

import matplotlib.pyplot as plt
 
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

