# %%
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from collections import OrderedDict
from sklearn import preprocessing
import scipy.cluster.hierarchy as spc
from scipy.cluster.hierarchy import dendrogram, linkage

# %%
data_csv = pd.read_csv("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/heart_disease/data/heart.csv")
# %%
data_csv
X = data_csv.iloc[:,0:-1]
X=(X-X.mean())/X.std()

Y = data_csv.iloc[:,-1]
# %%
fig = plt.figure(figsize=(10,6))
sns.heatmap(data_csv.corr(),vmin=-1, vmax=1)
plt.plot()
# %%
data_normalized = (data_csv-data_csv.mean())/data_csv.std()
sns.clustermap(data_normalized.corr(), method="complete", cmap='RdBu', annot=True, 
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12))
# %%
number_components = 5
pca = PCA(n_components=number_components)
components = pca.fit_transform(X)
principalDf = pd.DataFrame(data = components
             , columns = ["principal component {}".format(x+1) for x in range(number_components)])
finalDf = pd.concat([principalDf, Y], axis = 1)
# %%
colors = ["g","r"]
fig = plt.figure(figsize=(10,6))
for row in finalDf.to_numpy():
    plt.plot(row[0],row[1],label=int(row[-1]),color=colors[int(row[-1])],marker="s",linestyle="None")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA analysis")
# %%
fig = plt.figure(figsize=(10,6))
variation_sum = 0
plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_)
plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of PC")
plt.ylabel("Variance explained")
plt.title("PCA analysis")
plt.show()

# %%
fig = plt.figure(figsize=(10,6))
colors = ["g","r","b","y","c"]
labels = ["PCA 1","PCA 2"]
bar_width = 0.2
for i in range(2):
    plt.bar(np.arange(1,1+X.shape[-1]) + bar_width*i,pca.components_[i,:],bar_width,color=colors[i],label=labels[i])
plt.xticks(np.arange(1,1+X.shape[-1]),list(data_csv.columns[:-1]),rotation=90)
plt.legend()
plt.xlabel("Features")
plt.ylabel("Component coeffecient")
plt.title("PCA components")
plt.show()
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=2)
# %%
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)# %%

# %%
import eli5
from eli5.sklearn import PermutationImportance
# How does randomly shuffling specific features affect the prediction outcomes?
# If they get way worse the feature most likely is important for the predictions in the model
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

random_forrest_classifier = RandomForestClassifier(n_estimators=100,random_state=0)
y_train_pred = cross_val_predict(random_forrest_classifier, train_X, train_y, cv=3)
print("Confusion matrix")
print(confusion_matrix(train_y, y_train_pred))
print("The classifier is only correct:",precision_score(train_y, y_train_pred),"percent of the time")
print("It only detects",recall_score(train_y, y_train_pred),"heart disease")
# %%
# Partial Dependence Plots
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
feature_names = data_csv.columns.to_list()[:-1]
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='ca')
# plot it
pdp.pdp_plot(pdp_goals, 'ca')
plt.show()
# %%
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='thalach')
# plot it
pdp.pdp_plot(pdp_goals, 'thalach')
plt.show()
# %%
for feature in feature_names:
    # Create the data that we will plot
    pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature=feature)
    # plot it
    pdp.pdp_plot(pdp_goals, feature)
    plt.show()
# %%
# 2D Partial Dependence Plots
features_to_plot = ['thalach', 'cp']
inter1  =  pdp.pdp_interact(model=my_model, dataset=val_X, model_features=feature_names, features=features_to_plot)
pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()
# %%
# SHAP values
row_to_show = 2
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


my_model.predict_proba(data_for_prediction_array)
# %%
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
# %%
# Meaning the chance of getting heart disease is decreased by the features in blue and increased by the features in red
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# %%
# The average absolute value of SHAP value for each prediction
shap_values = explainer.shap_values(val_X)
shap.summary_plot(shap_values[1], val_X, plot_type="bar")

# %%
# The shap values for each feature, their value and their sign
shap.summary_plot(shap_values[1], val_X)

# %%
