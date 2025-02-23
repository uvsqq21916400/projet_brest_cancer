# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:24:38 2025

@author: flavi
"""
# prétraitement des données
import pandas as pd
import numpy as np 

df=pd.read_csv("C:/Users/flavi/Downloads/breast-cancer-dataset.csv")

df.head()
df.info()

# prétraitement des données
print(df.isnull().sum(),df.duplicated().sum()) #aucune valeurs manquantes et aucune lignes en doublon

print(df.dtypes)     

#conversion des variables catégorielles en numériques


for col in ['Year', 'Tumor Size (cm)', 'Inv-Nodes', 'Breast', 'Metastasis', 'Breast Quadrant', 'History', 'Diagnosis Result']:
    print(f"Valeurs uniques de '{col}':")
    print(df[col].unique())
    print("\n")

    
# on constate qu'il y'a des "#" comme valeurs uniques dans toutes les colonnes excepté "Diagnosis Result"
# et toutes les colonnes excepté les colonnes "Breast", 'Breast Quadrant' et 'Diagnosis Result' 
# sont sous forme de nombres cachés en texte

# Remplacer les '#' par des valeurs NaN
df.replace('#', np.nan, inplace=True)
print(df.isnull().sum())

# il y'a peu de valeurs manquantes donc supprimons les NaN
df.dropna(inplace=True)  
print(df.isnull().sum())  

# les modèles de Machine Learning ne traitent que des nombres donc convertissons les variables object en float
num_cols = ['Year', 'Tumor Size (cm)', 'Inv-Nodes', 'Metastasis', 'History']

df[num_cols] = df[num_cols].astype(float)

print(df.dtypes)

# verifions la dispersion des variables afin de normaliser ou non 
print(df.describe())

