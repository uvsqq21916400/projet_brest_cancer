# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:24:38 2025

@author: flavi
"""

import pandas as pd
df=pd.read_csv("C:/Users/flavi/Downloads/breast-cancer-dataset.csv")

df.head()
df.info()

# prétraitement des données
for col in df.columns:
 valeurs_manquantes=df[col].isnull().sum()
 print(f"La colonne '{col}' contient {valeurs_manquantes} valeur(s) manquante(s).")

 

