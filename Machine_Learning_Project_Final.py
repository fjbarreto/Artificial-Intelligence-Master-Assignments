####RETO KAGGLE: House Prices Advanced Regression Techniques####
##Frank Barreto. A905386.##
##Machine Learning. Master de Inteligencia Artificial##
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
SEED=30
plt.rcParams['figure.figsize'] = [10, 8]

import warnings
warnings.filterwarnings('ignore')

#Cargamos los datos.

train_df=pd.read_csv('train.csv', sep=',')
test_df=pd.read_csv('test.csv', sep=',')

y_train=train_df['SalePrice']
ID_test=test_df['Id']

#Transformamos la variable Sale Price.

y_train=np.log(y_train)

#Definimos una funcion capaz de dividir la base datos luego del preprocesado.

def dividir_df(df):
    train_df=df.iloc[0:1460,:]
    test_df=df.iloc[1460:2919,:]
    return train_df,test_df

#Definimos una funcion que con ayuda de un modelo de XGBoost nos regrese un RMLSE 
#aproximado como indicador de si estamos modificando los datos correctamente.

def score(X, y, model=XGBRegressor(random_state=SEED)):

    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes

    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

#Juntamos los data sets para explorar los datos.

df_exp=pd.concat([train_df, test_df], axis=0)
df_exp=df_exp.drop(['GarageYrBlt', 'SalePrice'], axis=1)

#Con la siguiente funcion vemos el tipo de variable de nuestros features y comprobamos si son correctos.
#En el caso de MSSubClass lo debemos convertir en categorica, ya que representa distintas clases.

df_exp.info()
df_exp['MSSubClass'] = df_exp.astype("category")

#Estudiamos los NaNs.

null_sum = df_exp.isnull().sum()

#Nos quitamos FireplaceQu y Alley ya que tienen mas de 1000 espacios en blanco y la informacion de su NaN esta expresada en la variable Fireplaces. 
#Ademas se observa que utilities solo tiene una clasificacion, la eliminamos.

df_exp=df_exp.drop(['FireplaceQu','Utilities','Alley', 'Id'], axis=1)

#Los valores en blanco de las siguientes variables las sustituiremos por la moda de la distribbucion de su columna por vecindario.

mode_nan = ["GarageArea", "LotFrontage", "MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "KitchenQual", "Functional"]
df_exp[mode_nan] = df_exp.groupby("Neighborhood")[mode_nan].transform(lambda x: x.fillna(x.mode()[0]))

#Transformamos los NaNs de las siguientes variables en None. De esta manera el modelo sera capaz de relacionar esta clase con el precio de venta.

none = ["PoolQC", "MiscFeature", "Fence", "GarageCond", "GarageQual", "GarageFinish", "GarageType", "BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"]
df_exp[none] = df_exp[none].fillna("None")

#Los NaNs de las siguientes variables numericas significan 0.

ceros = [ "MasVnrArea", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageCars"]
df_exp[ceros] = df_exp[ceros].fillna(0)

#Verifiquemos que solo la varibable SalePrice puede tener NaNs.

null_sum_pos=df_exp.isnull().sum()

#Hacemos feature engineering con algunas variables.

df_exp['SF']=df_exp['GrLivArea']+df_exp['TotalBsmtSF']
df_exp['PorchSF']=df_exp['OpenPorchSF']+df_exp['EnclosedPorch']+df_exp['3SsnPorch']+df_exp['ScreenPorch']
df_exp['Age']=df_exp['YearBuilt']-df_exp['YrSold']

#Drop the recipe features.

df_exp=df_exp.drop(['GrLivArea','TotalBsmtSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'],axis=1)

#Hacemos el mapping de las siguientes variables que representan ratings.

ratingFeatures = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond']

ratings= { 'Ex': 5, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0, 'Null': 0 }

for col in ratingFeatures:
    df_exp[col] = df_exp[col].map(ratings)
    
#Transformamos las siguientes variables en categoricas.
    
for col in df_exp.select_dtypes(include=['object', 'category']).columns:
    df_exp[col] = pd.Categorical(df_exp[col])
    df_exp[col] = df_exp[col].cat.codes

#Escalada y normalizacion de los datos.

num_col = df_exp.select_dtypes(exclude=["object","category"]).columns
scaler = StandardScaler()
df_exp[num_col] =scaler.fit_transform(df_exp[num_col])

#Binarizacion de las categorias. 

df_exp=pd.get_dummies(df_exp)

#Probamos un score de referencia con la funcion definida arriba. Obtenemos uno muy bajo con el training set de 0.001098.

train,test=dividir_df(df_exp)
score=score(train,y_train)
print(score)

#Utilizaremos la libreria XGBregressor de la libreria XGBOOST, para construir el modelo.
#Este algoritmo se conoce como uno de los mas rapidos y eficientes en el mundo del machine learning.
#Hacemos un Grid search para optimizar sus parametros con el grid que se muestra abajo. 
#Se ha comentado este codigo para evitar que se ejecutan las 270 fits que toma conseguir
#los hiperparametros adecuados.

#grid = { 'max_depth': [3,4,5],
         #  'learning_rate': [0.01, 0.05, 0.1],
         #  'n_estimators': [1000, 1500, 2000],
         #  'colsample_bytree': [0.3, 0.7]}

#xgbr = XGBRegressor(seed = SEED)
#clf = GridSearchCV(estimator=xgbr, param_grid=grid,scoring='neg_mean_squared_error', verbose=1)
#clf.fit(train, y_train)
#print("Best parameters:", clf.best_params_)
#print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

#Ahora que tenemos los mejores parametros hacemos el modelo final para hacer 
#las predicciones con el test set.

modelo_final=XGBRegressor(colsample_bytree= 0.3, learning_rate= 0.1, max_depth= 3, n_estimators= 1000)
modelo_final=modelo_final.fit(train,y_train)
y_pred=modelo_final.predict(test)

#Creamos el submission para probar su resultado en Kaggle.

submission = pd.DataFrame({'Id': ID_test, 'SalePrice':np.exp(y_pred)})

submission.to_csv("submission_final.csv", index=False)

#Con esta metodologia se ha conseguido  un score de 0.13601.