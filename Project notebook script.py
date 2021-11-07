# %% [markdown]
# # Take-home Assessment
# ## Data Exploration

# %%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import scipy.optimize
import sklearn.decomposition
import sklearn.manifold
import sklearn.model_selection
import os

%matplotlib inline 


# %%
def load_data(path):
    csv_path = os.path.join(path, "e-shop clothing 2008.csv")
    return pd.read_csv(csv_path, delimiter= ';')
    
commerce = load_data("") 
commerce.info()

# %%
commerce.shape

# %% [markdown]
# Tells us all the column names, their count and datatype. This shows we have no apparent missing values (all columns have same count). This means that we shouldn't need to use an imputer or other techniques to replace any NaN values in any of the columns.
# 
# We also see that there is only one non-numeric column: page 2 (clothing model). We will have to deal with this for our algorithms.

# %%
commerce.head()

# %% [markdown]
# Above we can see the first 5 entries in the data, giving us a better idea of the layout. We can also see what kind of data page 2 (clothing model) contains: codes of letter-number that represent each model/product.
# 
# Looking at the unique values in page 2:

# %%
commerce["page 2 (clothing model)"].unique().size

# %% [markdown]
# 217 unique clothing models. Looking at their frequency within the data:

# %%
commerce["page 2 (clothing model)"].value_counts()

# %% [markdown]
# Looking at some common metrics for the dataset:

# %%
commerce.describe()

# %% [markdown]
# Obviously, some of these metrics are more useful than others. Firstly, we see that every record is in 2008, so the year column is pointless and could be dropped for the learning later.
# 
# 
# Country, colour etc. are other categories that don't make much sense in this format; taking the mean of codes arbitrarily assigned to countries does not give useful information. 

# %%
commerce.hist(bins=50, figsize=(20,15))
plt.show()

# %% [markdown]
# Some of these histograms are more useful than others. Analysing more useful ones:
# 
# - Month, day: Almost uniform, though more orders in earlier months and more orders on days earlier in the month too
# - Order: Can see that most orders take very few clicks, with number of clicks dying off exponentially
# - Almost all orders are from one country: Poland
# - Session ID graph requires more bins of smaller size, so we can't draw much from this graph about any individual data, though we can say that, on average, they are all similar sizes.
# - Page 1 shows trousers were most frequently bought, every other item (skirts, blouses, sale) were bought equally frequently
# - Colour: most products are black and blue 
# - Location: slightly skewed to items top left and middle. Both right values are low.
# - Model photogrpahy shows far more en face photos than profile
# - Price appears normally distributed
# - Price 2 shows slightly more products were above average price than otherwise, but close
# - Pages decays exponentially, suggesting people are more likely to buy from earlier pages.

# %%
commerce["session ID"].unique().size

# %% [markdown]
# Knowing that min and max are 1 and 24026 respectively, we deduce that the values are just every unique integer in that range.

# %%
commerce["session ID"].value_counts()

# %% [markdown]
# ### Correlations

# %%
corr_matrix = commerce.corr()
corr_matrix

# %% [markdown]
# Largest correlations:
# 
# - Month/Session ID: strong positive
# 
# - Price/Price 2: strong negative
# 
# - Page/Model photography: weak positive
# 
# Looking at just price 2:
# 

# %%
corr_matrix["price 2"].sort_values(ascending=False) 

# %% [markdown]
# This gives us a good idea of what might be useful for our models later.
# 
# Scatter plot of most correlated (with price 2):

# %%
y = commerce["price 2"]

scatter_x = commerce["price"]
scatter_y = commerce["page 1 (main category)"]
group = y
cmap = matplotlib.cm.get_cmap('jet')
cdict = {1: cmap(0.5), 2: cmap(0.9)}


fig, ax = plt.subplots(figsize=(8, 6))
for g in np.unique(group):
    ix = group == g
    ax.scatter(scatter_x[ix], scatter_y[ix], c=np.array([cdict[g]]), label = g, s = 100, marker = "H",linewidth=2, alpha = 0.1)
ax.legend()
plt.xlabel('Price')
plt.ylabel('Page 1 (main category')

plt.show()

# %% [markdown]
# We see that the price 2 data points are separable for each clothing category, suggesting that if we figured out the separating value (which should just be the mean) for each one we could predict price 2 without any machine learning techniques. 

# %% [markdown]
# ### Excluding features
# It definitely makes sense to remove the year feature, as this adds no information to the dataset. It also has a standard deviation of zero, which will cause issues if we ever scale by dividing by standard deviation.
# 
# I will also drop the page 2 (clothing model) values for simplicity. There are too many values to feasibly consider one hot encoding/label binarising them.
# 
# I believe session id should also be dropped, as although it may improve accuracy of the model, it is not helpful to the task we are predicting for.
# 
# After training the models with this set up I will also consider dropping the price category and retraining, which directly gives price 2 for the reason above.
# 
# 

# %%
commerce = commerce.drop("year", axis=1)
commerce = commerce.drop("page 2 (clothing model)", axis=1)
commerce = commerce.drop("session ID", axis=1)

# %% [markdown]
# ## Split data into test and train
# 

# %%
X = commerce.drop("price 2", axis=1).values
y = commerce["price 2"].values

# %% [markdown]
# Stratified:

# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.get_n_splits(X, y)
print(split)       

for train_index, test_index in split.split(X, y):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %% [markdown]
# Compare proportions:

# %%
def subset_proportions(subset):
    props = {}
    for value in set(subset):
        data_value = [i for i in subset if i==value]
        props[value] = len(data_value) / len(subset)
    return props
    
compare_props = pd.DataFrame({
    "Overall": subset_proportions(y),

    "Stratified tr": subset_proportions(y_train),
    "Stratified ts": subset_proportions(y_test),
})
compare_props["Strat. tr %error"] = 100 * compare_props["Stratified tr"] / compare_props["Overall"] - 100
compare_props["Strat. ts %error"] = 100 * compare_props["Stratified ts"] / compare_props["Overall"] - 100


compare_props.sort_index()

# %% [markdown]
# This shows that the stratified split represents the original dataset very well. We use stratified sampling to avoid introducing any sampling bias (more frequent values dominate a sample). This could be introduced if we chose random sampling instead.
# 

# %% [markdown]
# ### Transformations and data preparation
# 
# Don't really need to do any large transformations here; no imputing required and for now we are ignoring the textual data. 
# 
# We can still do feature scaling:

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
commerce_scaled = scaler.fit_transform(commerce)

# %% [markdown]
# ## Machine Learning implementation
# 
# This task requires us to predict the target value of price 2, which takes value 1 if they paid above average price for this product, otherwise 2. This means that it is a binary classification problem, and the algorithms we use should reflect that.
# 

# %%
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

sgd = SGDClassifier(max_iter=5, tol=None, random_state=42, loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
sgd.fit(X_train, y_train) 
print("Stochastic gradient descent:", cross_val_score(sgd, X_train, y_train, cv=5, scoring="accuracy").mean())

# %%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print("Logistic regression:", cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy").mean())

# %%
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB() 
gnb.fit(X_train, y_train)

print("Gaussian naive Bayes':", cross_val_score(gnb, X_train, y_train, cv=5, scoring="accuracy").mean())

# %%
from sklearn.kernel_approximation import RBFSampler

rbf_features = RBFSampler(gamma=0.001, random_state=42)
X_train_features = rbf_features.fit_transform(X_train)
sgd_rbf = SGDClassifier(max_iter=100, random_state=42, loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
sgd_rbf.fit(X_train_features, y_train) 

print("SGD RBF, gamma = 0.001:",cross_val_score(sgd_rbf, X_train_features, y_train, cv=5, scoring="accuracy").mean())


# %% [markdown]
# Testing different gamma values:

# %%
fig, ax = plt.subplots(figsize=(8, 6))
accuracies = []
for i in range (-4,2):
    rbf_features = RBFSampler(gamma=10**i, random_state=42)
    X_train_features = rbf_features.fit_transform(X_train)
    sgd_rbf100 = SGDClassifier(max_iter=100, random_state=42, loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
    sgd_rbf100.fit(X_train_features, y_train) 
    accuracies.append(cross_val_score(sgd_rbf100, X_train_features, y_train, cv=5, scoring="accuracy").mean())

plt.bar(range(-4,2),accuracies)
plt.xlabel("Log 10 of Gamma")
plt.ylabel("CV accuracies mean")
plt.show

# %% [markdown]
# Shows that changing gamma across range 0.001-100 gives almost no change in accuracy, and that it fluctuates within this. Highest is given by 0.001, so train model on that:

# %%
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

print("Random Forest:",cross_val_score(rnd_clf, X_train, y_train, cv=5, scoring="accuracy").mean())

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
print("AdaBoost:",cross_val_score(ada_clf, X_train, y_train, cv=5, scoring="accuracy").mean())

# %% [markdown]
# ## Evaluation
# 
# Starting with a base estimator for comparison:

# %%
from sklearn.base import BaseEstimator

class NotXClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        out = np.zeros((len(X)), dtype=int)+2
        return out
    
not_premium_clf = NotXClassifier()

cross_val_score(not_premium_clf, X_test, y_test, cv=5, scoring="accuracy").mean()

# %% [markdown]
# Now looking at accuracies of each model:

# %%
from sklearn.metrics import precision_score, recall_score, f1_score


X_test_features = rbf_features.transform(X_test)

print("Logistic regression:", cross_val_score(log_reg, X_test, y_test, cv=5, scoring="accuracy").mean())
print("Gaussian naive Bayes':", cross_val_score(gnb, X_test, y_test, cv=5, scoring="accuracy").mean())
print("Stochastic gradient descent:",cross_val_score(sgd, X_test, y_test, cv=5, scoring="accuracy").mean())
print("SGD RBF:",cross_val_score(sgd_rbf, X_test_features, y_test, cv=5, scoring="accuracy").mean())
print("Random forest:",cross_val_score(rnd_clf, X_test, y_test, cv=5, scoring="accuracy").mean())
print("AdaBoost:",cross_val_score(ada_clf, X_test, y_test, cv=5, scoring="accuracy").mean())


# %% [markdown]
# AdaBoost gets the best accuracy of the different algorithms. Stochastic gradient does the worst, even with kernel mappping. Now looking at some other metrics:

# %%
print("Precision:\n")
print("Logistic regression:", cross_val_score(log_reg, X_test, y_test, cv=5, scoring="precision").mean())
print("Gaussian naive Bayes':", cross_val_score(gnb, X_test, y_test, cv=5, scoring="precision").mean())
print("Stochastic gradient descent:",cross_val_score(sgd, X_test, y_test, cv=5, scoring="precision").mean())
print("SGD RBF:",cross_val_score(sgd_rbf, X_test_features, y_test, cv=5, scoring="precision").mean())
print("Random forest:",cross_val_score(rnd_clf, X_test, y_test, cv=5, scoring="precision").mean())
print("AdaBoost:",cross_val_score(ada_clf, X_test, y_test, cv=5, scoring="precision").mean())

# %%
print("Recall:\n")
print("Logistic regression:", cross_val_score(log_reg, X_test, y_test, cv=5, scoring="recall").mean())
print("Gaussian naive Bayes':", cross_val_score(gnb, X_test, y_test, cv=5, scoring="recall").mean())
print("Stochastic gradient descent:",cross_val_score(sgd, X_test, y_test, cv=5, scoring="recall").mean())
print("SGD RBF:",cross_val_score(sgd_rbf, X_test_features, y_test, cv=5, scoring="recall").mean())
print("Random forest:",cross_val_score(rnd_clf, X_test, y_test, cv=5, scoring="recall").mean())
print("AdaBoost:",cross_val_score(ada_clf, X_test, y_test, cv=5, scoring="recall").mean())

# %%
print("F1:\n")
print("Logistic regression:", cross_val_score(log_reg, X_test, y_test, cv=5, scoring="f1").mean())
print("Gaussian naive Bayes':", cross_val_score(gnb, X_test, y_test, cv=5, scoring="f1").mean())
print("Stochastic gradient descent:",cross_val_score(sgd, X_test, y_test, cv=5, scoring="f1").mean())
print("SGD RBF:",cross_val_score(sgd_rbf, X_test_features, y_test, cv=5, scoring="f1").mean())
print("Random forest:",cross_val_score(rnd_clf, X_test, y_test, cv=5, scoring="f1").mean())
print("AdaBoost:",cross_val_score(ada_clf, X_test, y_test, cv=5, scoring="f1").mean())

# %% [markdown]
# ### Confusion Matrices
# 
# I picked a few of the classifiers with more interesting results to produce confusion matrices for:

# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_test_pred = cross_val_predict(sgd, X_test, y_test, cv=5)
print("Stochastic Gradient Descent:")
confusion_matrix(y_test, y_test_pred)

# %% [markdown]
# Predicted positive too many times.

# %%
y_test_pred = cross_val_predict(gnb, X_test, y_test, cv=5)
print("Gaussian naive Bayes':")
confusion_matrix(y_test, y_test_pred)

# %% [markdown]
# Predicted negative too many times

# %%
y_test_pred = cross_val_predict(rnd_clf, X_test, y_test, cv=5)
print("Random forest:")
confusion_matrix(y_test, y_test_pred)

# %% [markdown]
# Only got false positives, not any false negatives

# %%
y_test_pred = cross_val_predict(ada_clf, X_test, y_test, cv=5)
print("AdaBoost:")
confusion_matrix(y_test, y_test_pred)

# %% [markdown]
# No mistakes

# %% [markdown]
# ### Feature importance
# 
# Taking a deeper look at the importances in the random forest classifier:

# %%
importances = zip(commerce.drop("price 2", axis=1).columns, rnd_clf.feature_importances_)
importances = sorted(importances, key = lambda x: x[1], reverse=True) 
for name, score in importances:
    print(name, ":", score)

# %%
importances = zip(commerce.drop("price 2", axis=1).columns, ada_clf.feature_importances_)
importances = sorted(importances, key = lambda x: x[1], reverse=True) 
for name, score in importances:
    print(name, ":", score)

# %% [markdown]
# Here we see that, as previously thought, price 2 is almost entirely dependent on price and the clothing type (page 1). Interestingly, AdaBoost and the forest regressor use very different weights for each of these values, but achieve similar levels of success in prediciting price 2.
# 
# What is perhaps more interesting is the relative importance of colour in the forest regressor compared to the other features. This would suggest that the colour of an item of clothing is what would push a consumer to pay a premium for an item. 
# 
# Both also weigh page and location fairly high, suggesting that where it is on the page and how many pages they have to scroll through is also a factor. One possible reason for this is that the customer is able to view more items as they go through the pages, giving them more data to calculate an idea of the average price they should pay for an item. The shop could make use of this by displaying more expensive items earlier.

# %% [markdown]
# ## Simple mean testing method
# 
# I decided to implement a naive method that just checks if a price is above or below the mean price for that product. This method therefore just uses the price and page 1 (main category) attributes to predict price 2.

# %%
clothing_cats = commerce["page 1 (main category)"].unique()
means = np.zeros(4)

#Calculate mean for each clothing category, 1-4
for c in clothing_cats:
    rows = np.where(commerce["page 1 (main category)"] == c)
    tot = 0
    for x in rows[0]:
        tot += commerce["price"][x]
    means[c-1] = (tot/len(rows[0]))

#Predict based on whether it is greater than mean 
def predict(X):
    y_preds = np.zeros(len(X), int)
    y_preds += 2
    ones = np.where(X[:,8] > means[X[:,4]-1])
    y_preds[ones] = 1
    return y_preds

def calc_score(preds, targets):
    correct = np.sum(preds == targets)
    score = correct/len(preds)
    return score

y_pred = predict(X_train)
print(calc_score(y_pred, y_train))



# %%
y_pred = predict(X_test)
print(calc_score(y_pred, y_test))

# %% [markdown]
# And also 100% accuracy on test set!

# %% [markdown]
# ## Alternatives
# Since the data was so easy to predict for the ensemble classifiers, I decided to retrain them, but without the important price category. This will give a better idea as to which other attributes are important when determining if the customer has paid a premium.

# %%
commerce_dropped = commerce.drop("price", axis= 1)
X2 = commerce_dropped.drop("price 2", axis=1).values
y2 = commerce_dropped["price 2"].values

# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.get_n_splits(X2, y2)
print(split)       

for train_index, test_index in split.split(X2, y2):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X2_train, X2_test = X2[train_index], X2[test_index]
    y2_train, y2_test = y2[train_index], y2[test_index]

print(X2_train.shape, y2_train.shape, X2_test.shape, y2_test.shape)

# %%
gnb2 = GaussianNB() 
gnb2.fit(X2_train, y2_train)

print("Gaussian naive Bayes':", cross_val_score(gnb2, X2_train, y2_train, cv=5, scoring="accuracy").mean())

# %%
ada_clf2 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf2.fit(X2_train, y2_train)
print("AdaBoost:",cross_val_score(ada_clf2, X2_train, y2_train, cv=5, scoring="accuracy").mean())

# %% [markdown]
# Significantly worse results from what were the best two classifiers before. Tweaking hyperparameters:

# %%
ada_clf2 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=300,
    algorithm="SAMME.R", learning_rate=0.1, random_state=42)
ada_clf2.fit(X2_train, y2_train)
print("AdaBoost:",cross_val_score(ada_clf2, X2_train, y2_train, cv=5, scoring="accuracy").mean())

# %% [markdown]
# Learning rate adjustment accuracies:
# - Learning rate 0.01, accuracy = 0.66
# - Learning rate 1, accuracy = 0.62
# - Learning rate 0.1 accuracy = 0.68
# 
# Number of estimators increased accuracy but also increased execution time, so I settled at 300, which pushed it above the 70% accuracy mark.
# 
# Now looking at performance on the test set:

# %%
print("Accuracy:",cross_val_score(ada_clf2, X2_test, y2_test, cv=5, scoring="accuracy").mean())
print("Precision:",cross_val_score(ada_clf2, X2_test, y2_test, cv=5, scoring="precision").mean())
print("Recall:",cross_val_score(ada_clf2, X2_test, y2_test, cv=5, scoring="recall").mean())
print("F1:",cross_val_score(ada_clf2, X2_test, y2_test, cv=5, scoring="f1").mean())

y2_test_pred = cross_val_predict(ada_clf2, X2_test, y2_test, cv=5)
confusion_matrix(y2_test, y2_test_pred)

# %% [markdown]
# Again looking at importances:

# %%
importances = zip(commerce.drop("price 2", axis=1).drop("price", axis=1).columns, ada_clf2.feature_importances_)
importances = sorted(importances, key = lambda x: x[1], reverse=True) 
for name, score in importances:
    print(name, ":", score)

# %% [markdown]
# This now more clearly shows the importance of the colour of the item when determining what made a user pay a premium.

# %%
commerce1 = load_data("") 
commerce1 = commerce1.drop("year", axis = 1).drop("price", axis=1)
y3 = commerce1["price 2"].values
commerce1 = commerce1.drop("price 2", axis = 1)

# %% [markdown]
# Small pipeline to convert catergorical values and add scaling:

# %%
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer 

cat_attribs = ["page 2 (clothing model)"]
num_attribs = list(commerce1.drop(["page 2 (clothing model)"], axis=1))

class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self
    def transform(self, X, y=0):
        return self.encoder.transform(X)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', CustomLabelBinarizer()),
    ])


full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


commerce_prepared = full_pipeline.fit_transform(commerce1)
print(commerce_prepared.shape)
commerce_prepared

# %% [markdown]
# Split into test and train:

# %%
X3 = commerce_prepared

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.get_n_splits(X3, y3)
print(split)       

for train_index, test_index in split.split(X3, y3):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X3_train, X3_test = X3[train_index], X3[test_index]
    y3_train, y3_test = y3[train_index], y3[test_index]

print(X3_train.shape, y3_train.shape, X3_test.shape, y3_test.shape)

# %%
gnb3 = GaussianNB() 
gnb3.fit(X3_train, y3_train)

print("Gaussian naive Bayes':", cross_val_score(gnb3, X3_train, y3_train, cv=5, scoring="accuracy").mean())

# %% [markdown]
# Significantly better results, even without price!

# %%
ada_clf3 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf3.fit(X3_train, y3_train)
print("AdaBoost:",cross_val_score(ada_clf3, X3_train, y3_train, cv=5, scoring="accuracy").mean())

# %% [markdown]
# On test data:

# %%
print("Gaussian naive Bayes':", cross_val_score(gnb3, X3_test, y3_test, cv=5, scoring="accuracy").mean())
print("AdaBoost:",cross_val_score(ada_clf3, X3_test, y3_test, cv=5, scoring="accuracy").mean())

# %%

column_names = np.concatenate((commerce1.drop("page 2 (clothing model)", axis=1).columns.values, commerce1["page 2 (clothing model)"].unique()))
importances = zip(column_names, ada_clf3.feature_importances_)
importances = sorted(importances, key = lambda x: x[1], reverse=True) 
for name, score in importances:
    print(name, ":", score)

# %% [markdown]
# What we notice here is that, even though each of these new features has a very low weighting individually, the sheer number of them massively improves the predicition of price 2. 
# 
# The conclusion we can draw from this is that people are more likely to pay a premium depending on who is modelling the item.

# %% [markdown]
# ## Visualisation and dimensionality reduction
# For this we will reduce the size of the data we use, to reduce the time these algorithms take, especially tSNE.

# %%
# Reduce size
commerce_small = commerce[:1000]
commerce_small.shape

# %%
features = commerce.columns

with plt.rc_context({'figure.figsize': (20,11)}):
    fig,ax = plt.subplots(len(features), len(features), sharex='col', sharey='row')

# We'll plot histograms on the diagonal, so they shouldn't share y-axis with the scatter plots
for i in range(len(features)):
    ax[i,i].get_shared_y_axes().remove(ax[i,i])

# Plot histograms or scatter plots as appropriate
for i,c in enumerate(features):
    for j,d in enumerate(features):
        if i == j:
            ax[i,j].hist(commerce_small[d], bins=30)
        else:
            ax[i,j].scatter(commerce_small[d], commerce_small[c], alpha=.2)

# Rotate tick labels to make them legible}
for i,c in enumerate(features):
    ax[i,0].set_ylabel(c, rotation=0, horizontalalignment='right')
for j,d in enumerate(features):
    ax[len(features)-1,j].set_xlabel(d, rotation=-30, ha='left')

plt.show()

# %% [markdown]
# ### PCA
# 

# %%

features = commerce_small.columns
X = commerce_small[features].values

# rescale the features, so they have the same variance
for k in range(len(features)):
    X[:,k] = X[:,k] / np.std(X[:,k])

pca = sklearn.decomposition.PCA()
pca_result = pca.fit_transform(X)

p1,p2 = pca_result[:,0], pca_result[:,1]

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(p1, p2, alpha=.2)

for lvl in np.unique(commerce_small['price 2']):
    s = commerce_small['price 2']
    i = (s == lvl)
    plt.scatter(p1[i], p2[i], label=lvl, alpha = 0.5)

plt.show()

# %%
tsne = sklearn.manifold.TSNE(n_components=2, verbose=0)
tsne_results = tsne.fit_transform(X)

p1,p2 = tsne_results[:,0], tsne_results[:,1]

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(p1, p2, alpha=.2)

for lvl in np.unique(commerce_small['price 2']):
    s = commerce_small['price 2']
    i = (s == lvl)
    plt.scatter(p1[i], p2[i], label=lvl, alpha = 0.5)

ax.set_aspect('equal')
plt.show()

# %% [markdown]
# Dropping the price feature

# %%
commerce_small = commerce_small.drop(["price"], axis=1)
features = commerce_small.columns
X = commerce_small[features].values

# rescale the features, so they have the same variance
for k in range(len(features)):
    X[:,k] = X[:,k] / np.std(X[:,k])

pca = sklearn.decomposition.PCA()
pca_result = pca.fit_transform(X)

p1,p2 = pca_result[:,0], pca_result[:,1]

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(p1, p2, alpha=.2)

for lvl in np.unique(commerce_small['price 2']):
    s = commerce_small['price 2']
    i = (s == lvl)
    plt.scatter(p1[i], p2[i], label=lvl, alpha = 0.5)

plt.show()

# %%
tsne = sklearn.manifold.TSNE(n_components=2, verbose=0)
tsne_results = tsne.fit_transform(X)

p1,p2 = tsne_results[:,0], tsne_results[:,1]

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(p1, p2, alpha=.2)

for lvl in np.unique(commerce_small['price 2']):
    s = commerce_small['price 2']
    i = (s == lvl)
    plt.scatter(p1[i], p2[i], label=lvl, alpha = 0.5)

ax.set_aspect('equal')
plt.show()

# %%
tsne = sklearn.manifold.TSNE(n_components=2, verbose=0, perplexity = 5)
tsne_results = tsne.fit_transform(X)

p1,p2 = tsne_results[:,0], tsne_results[:,1]

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(p1, p2, alpha=.2)

for lvl in np.unique(commerce_small['price 2']):
    s = commerce_small['price 2']
    i = (s == lvl)
    plt.scatter(p1[i], p2[i], label=lvl, alpha = 0.5)

ax.set_aspect('equal')
plt.show()

# %%
tsne = sklearn.manifold.TSNE(n_components=2, verbose=0, perplexity = 50)
tsne_results = tsne.fit_transform(X)

p1,p2 = tsne_results[:,0], tsne_results[:,1]

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(p1, p2, alpha=.2)

for lvl in np.unique(commerce_small['price 2']):
    s = commerce_small['price 2']
    i = (s == lvl)
    plt.scatter(p1[i], p2[i], label=lvl, alpha = 0.5)

ax.set_aspect('equal')
plt.show()

# %% [markdown]
# Plotting clothing colour against location with price 2 marked:

# %%
y = commerce["price 2"]

scatter_x = commerce["colour"]
scatter_y = commerce["location"]
group = y
cmap = matplotlib.cm.get_cmap('jet')
cdict = {1: cmap(0.5), 2: cmap(0.9)}


fig, ax = plt.subplots(figsize=(8, 6))
for g in np.unique(group):
    ix = group == g
    ax.scatter(scatter_x[ix], scatter_y[ix], c=np.array([cdict[g]]), label = g, s = 100, marker = "H",linewidth=2, alpha = 0.1)
ax.legend()
plt.xlabel('Colour')
plt.ylabel('Location')

plt.show()

# %%



