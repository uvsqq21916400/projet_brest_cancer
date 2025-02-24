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

# verifions la dispersion des variables afin de normaliser ou non 
print(df.describe())

# standardisation
scaler = StandardScaler()

cols_to_scale = ['Age', 'Tumor Size (cm)', 'Inv-Nodes']

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print(df[['Age', 'Tumor Size (cm)', 'Inv-Nodes']].describe())


# Visualisons la répartition des diagnostics (bénin vs malin)
plt.figure(figsize=(6,4))
sns.countplot(x=df["Diagnosis Result"], palette="viridis")
plt.title("Répartition des diagnostics (Bénin vs Malin)")
plt.xlabel("Type de Cancer")
plt.ylabel("Nombre de patients")
plt.show()

# On observe beaucoup plus de patient (une bonne centaine) avec un cancer bénini,
# que de patient atteint d'un cancer malin mais le nombre reste tout de même important (environ 90)

# analysons comment des variables comme l'âge, la taille de la tumeur et les ganglions envahis 
# influencent le diagnostic (bénin/malin)

# Regardons l'influence de l'âge sur les diagnostics
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Diagnosis Result"], y=df["Age"], palette="magma")
plt.title("Répartition de l'âge selon le diagnostic")
plt.xlabel("Type de Cancer")
plt.ylabel("Âge des patients")
plt.show()

# On observe que l'âge médian chez les patients atteint de d'un cancer malin est plus âgé
# que ceux atteint d'un cancer bénin, de plus la distribution est moins large ,
# ce qui montre une plus grande dispersion d'âge chez les patients atteint d'un cancer bénin
# en effet il y'a des âges beaucoup plus jeunes que la mediane :  
# certains patients ont un âge beaucoup plus élevé ou plus bas que la majorité des autres


# Regardons l'influence de de la taille de la tumeur sur les diagnostics
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Diagnosis Result"], y=df["Tumor Size (cm)"], palette="coolwarm")
plt.title("Taille de la tumeur en fonction du diagnostic")
plt.xlabel("Type de Cancer")
plt.ylabel("Taille de la tumeur (cm)")
plt.show()

# Les tailles de tumeur à diagnostic malin sont évidemmant plus grande que ceux bénin ,
# on note tout de même des outliers dans les deux diagnostics montrant des tailles de tumeurs 
# anormalement grande pour des cancers malins


# Regardons l'influence des ganglions envahis sur les diagnostics 
# ganglions envahis :enflure causée par une accumulation de lymphe dans les tissus mous dit lymphœdème, qui est
# une augmentation durable du volume d'un bras ou d'une jambe liée à une accumulation de lymphe
# (=liquide blanchâtre ou jaunâtre qui recueille certains déchets, des bactéries et des cellules endommagées provenant de l'intérieur des tissus du corps afin qu'ils puissent être évacués du corps ou détruits)
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Diagnosis Result"], y=df["Inv-Nodes"], palette="viridis")
plt.title("Nombre de ganglions envahis en fonction du diagnostic")
plt.xlabel("Type de Cancer")
plt.ylabel("Nombre de ganglions envahis")
plt.show()

# On observe que la plupart des patients bénins ont peu ou pas de ganglions envahis, cependant
# la médiane est plus élevée pour les cancers malins,
# ce qui suggère une forte corrélation entre le nombre de ganglions touchés et la malignité du cancer


# Analysons les corrélations entre toutes les variables 
# pour voir lesquelles ont le plus d’impact sur le diagnostic

# Affichons la matrice de corrélation
# Sélectionner uniquement les colonnes numériques
df_numeric = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,8))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation entre les variables numériques")
plt.show()


# Préparons le modèle pour le machine learning :
    
# Nous allons utiliser la régression logistique, efficace pour notre dataset et problématique, 
# en effet c'est un modèle de classification qui permet de prédire une classe binaire et donc performant pour 
# des classes bien séparées (ici cancer bénin ou malin)
X = df[['Age', 'Tumor Size (cm)', 'Inv-Nodes', 'Metastasis']]
y = df['Diagnosis Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dimensions de X_train :", X_train.shape)
print("Dimensions de X_test  :", X_test.shape)
print("Dimensions de y_train :", y_train.shape)
print("Dimensions de y_test  :", y_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")
print("\nRapport de classification :\n", classification_report(y_test, y_pred))

# Les résultats confirment nos prédictions sur l'efficacité du modèle sur nos données;
# optimisons le modèle de régression logistique pour améliorer ses performances, notamment sur la détection des cancers malins;
# en effet notre modèle rate encore trop de cancers malins (recall = 0.78).

model_optimized = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.5)

model_optimized.fit(X_train, y_train)

y_pred_optimized = model_optimized.predict(X_test)

accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"Précision du modèle optimisé : {accuracy_optimized:.2f}")
print("\nRapport de classification (modèle optimisé) :\n", 
      classification_report(y_test, y_pred_optimized))

# Le modèle est plus performant pour détecter les cancers malins


# Analysons maintenantl'importance des variables nous permet de comprendre quelles caractéristiques 
# influencent le plus le diagnostic du cancer
coefficients = model_optimized.coef_[0]

feature_importance = pd.DataFrame({'Variable': X.columns, 'Coefficient': coefficients})

feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

print(feature_importance[['Variable', 'Coefficient']])

# Le facteur le plus influent est le nombre de ganglions envahis (Inv-Nodes).
# Si cette variable est élevée, le risque de cancer malin est plus grand
# Cela confirme que la propagation aux ganglions lymphatiques est un critère clé en oncologie

