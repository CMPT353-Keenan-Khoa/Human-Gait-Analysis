import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier


### x, y are known data input. x_predict is value want to predict
def linear_reg(x,y, x_predict):
    X = np.stack([x['pace']], axis=1)  # join 1D array to 2D
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = LinearRegression(fit_intercept=True)  # Create a model
    model.fit(X_train, y_train)     #Train the data
    print("lineary regression score on train: ",model.score(X_train, y_train))
    print("linear regression score: ",model.score(X_valid, y_valid))
    ##Predict data for x_predict value
    # X_fit =  np.stack([x_predict['step'], x_predict['height']], axis=1)
    X_fit = x_predict
    y_fit = model.predict(X_fit)
    print(y_fit)

    plt.plot (x, y, 'b.')
    plt.plot(X_fit, y_fit, 'g.')
    plt.plot(x, model.predict(X), 'r-')
    plt.show()
    return

def polynomial_reg(x,y, x_predict):
    model = make_pipeline(
        PolynomialFeatures(degree=5, include_bias=True),
        LinearRegression(fit_intercept=False)
    )
    X = np.stack([x['pace']], axis=1)  # join 1D array to 2D
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model.fit(X_train, y_train)
    print("poly regression score on train: ",model.score(X_train, y_train))
    print("poly regression score: ",model.score(X_valid, y_valid))   

    # model.fit(X, y)
    #Predict data for x_predict value
    X_fit = x_predict
    y_fit = model.predict(X_fit)
    print(y_fit)
    return


def classification_model(x, y, x_predict):
    X_train, X_valid, y_train, y_valid = train_test_split(x, y)\


    rf_model = RandomForestClassifier(n_estimators=500, max_depth=5)#, min_samples_leaf=0.1)
    rf_model.fit(X_train, y_train)
    y_rf_fit = rf_model.predict(x_predict)


    ## If the data is imbalance between male and female. We can rebalance it
    # female = data[data['gender'] == 1]
    # male = data[data['gender'] == 0].sample(n=male.shape[0])
    # balanced_data = female.append(male)
    
    print('Random Forest Model: ',rf_model.score(X_valid, y_valid))
    print("Random Forest predict: ",y_rf_fit)

    return


def get_clusters(X):
    """
    Find clusters of the weather data.
    """
    model = make_pipeline(
        KMeans(n_clusters=5)
    )
    model.fit(X)
    return model.predict(X)

def get_pca(X):
    """
    Transform data to 2D points for plotting. Should return an array with shape (n, 2).
    """
    flatten_model = make_pipeline(
        MinMaxScaler(),
        PCA(2)
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2

if __name__ == "__main__":
    filename = sys.argv[1]
    data = pd.read_csv(filename)
    # x = data[['pace']]
    # y = data['height']
    # x_predict= np.array([[0.9]])
    # linear_reg(x, y , x_predict)
    # polynomial_reg(x, y , x_predict)
    #plt.plot (data.loc[data['range']==1]['pace'], data.loc[data['range']==1]['step_length'], 'ro')
    #plt.plot (data.loc[data['range']==2]['pace'], data.loc[data['range']==2]['step_length'], 'go')
    #plt.plot (data.loc[data['range']==3]['pace'], data.loc[data['range']==3]['step_length'], 'yo')
    #plt.plot (data.loc[data['range']==4]['pace'], data.loc[data['range']==4]['step_length'], 'bo')
    
    #plt.plot (data.loc[data['range']==1]['step_length'], 'ro')
    #plt.plot (data.loc[data['range']==2]['step_length'], 'go')
    #plt.plot (data.loc[data['range']==3]['step_length'], 'yo')
    #plt.plot (data.loc[data['range']==4]['step_length'], 'bo')
    #plt.ylabel('step_length')

    #plt.plot (data.loc[data['range']==1]['pace'], 'ro')
    #plt.plot (data.loc[data['range']==2]['pace'], 'go')
    #plt.plot (data.loc[data['range']==3]['pace'], 'yo')
    #plt.plot (data.loc[data['range']==4]['pace'], 'bo')
    #plt.ylabel('pace')
    
    #plt.show()
    X = data[['pace','step_length']]
    y = data['range']
    x_predict =[[0.85,60]]
    classification_model(X,y, x_predict)

    #X2 = get_pca(X)
    #clusters = get_clusters(X)
    #plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=20)
    #plt.savefig('clusters.png')

    #df = pd.DataFrame({
    #    'cluster': clusters,
    #    'range': y,
    #})
    #counts = pd.crosstab(df['range'], df['cluster'])
    #print(counts)
