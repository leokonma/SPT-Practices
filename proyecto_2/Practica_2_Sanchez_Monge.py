import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

# ---------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------
os.chdir(r"C:\Users\leodo\OneDrive\Escritorio\special topics\Practicas\proyecto_2") 
titanic_raw = pd.read_csv("Titanic_2 (1).csv")
meteo_raw = pd.read_csv("condiciones_meteorologicas.csv")

print("\nDimensiones Titanic:", titanic_raw.shape)
titanic_raw.info()

# ---------------------------------------------------------------------
# 2. Eliminación de variables poco relevantes
# ---------------------------------------------------------------------
print("Columnas originales:", list(titanic_raw.columns))

titanic = titanic_raw.copy()
titanic.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Barco"], inplace=True)

titanic.describe()
titanic.info()

# ---------------------------------------------------------------------
# 3. Valores perdidos
# ---------------------------------------------------------------------
missing_pct = titanic.isnull().mean() * 100
print("\nPorcentaje de NaN por variable (%):")
print(missing_pct)

n_rows = len(titanic)
rows_comp = len(titanic.dropna())
pct_remaining = rows_comp / n_rows * 100
pct_lost = 100 - pct_remaining
print(f"Porcentaje de registros que se conservarían: {pct_remaining:.2f}%")
print(f"Porcentaje de registros que se perderían: {pct_lost:.2f}%")

# ---------------------------------------------------------------------
# 3.1 Imputación simple
# ---------------------------------------------------------------------
num_var = titanic.select_dtypes(include=["int64", "float64"]).columns
cat_var = titanic.select_dtypes(include=["object"]).columns

print("\nColumnas numéricas:", list(num_var))
print("Columnas categóricas:", list(cat_var))

imp_numeric = SimpleImputer(strategy="median")
imp_categorical = SimpleImputer(strategy="most_frequent")

titanic_test_1 = titanic.copy()
titanic_test_1[num_var] = imp_numeric.fit_transform(titanic_test_1[num_var])
titanic_test_1[cat_var] = imp_categorical.fit_transform(titanic_test_1[cat_var])

print("\n¿Quedan NaN tras imputación simple?:",
      titanic_test_1.isnull().any().any())
print("\nResumen numérico imputación simple:")
print(titanic_test_1[num_var].describe())

# ---------------------------------------------------------------------
# 3.2 Imputación iterativa
# ---------------------------------------------------------------------
iter_imputer = IterativeImputer(random_state=42)
titanic_test_2 = titanic.copy()
titanic_test_2[num_var] = iter_imputer.fit_transform(titanic_test_2[num_var])
titanic_test_2[cat_var] = imp_categorical.fit_transform(titanic_test_2[cat_var])

print("\n¿Quedan NaN tras imputación iterativa?:",
      titanic_test_2.isnull().any().any())

print("\nComparación medias de Edad:")
print("Edad (simple):   ", f"{titanic_test_1['Age'].mean():.2f}")
print("Edad (iterativa):", f"{titanic_test_2['Age'].mean():.2f}")

# ---------------------------------------------------------------------
# 4. Correcciones ortográficas
# ---------------------------------------------------------------------
print(titanic_test_1["Sex"].value_counts())

def correct_categorical_typos(series, valid_values, cutoff=0.7):
    corrected = []
    for value in series.astype(str):
        matches = difflib.get_close_matches(value, valid_values, n=1, cutoff=cutoff)
        corrected.append(matches[0] if matches else value)
    return pd.Series(corrected, index=series.index)

titanic = titanic_test_1.copy()

titanic["Sex"] = correct_categorical_typos(titanic["Sex"], ["male", "female"], cutoff=0.6)
print("\nSex corregido:")
print(titanic["Sex"].value_counts())

titanic["Embarked"] = correct_categorical_typos(titanic["Embarked"], ["C", "Q", "S"], cutoff=0.6)
print("\nEmbarked corregido:")
print(titanic["Embarked"].value_counts())

# ---------------------------------------------------------------------
# 5. Outliers
# ---------------------------------------------------------------------
num_var = titanic.select_dtypes(include=["int64", "float64"]).columns
outliers_cols = ["Age", "Fare"]

print("\nResumen variables outliers:")
print(titanic[outliers_cols].describe())

# 5.1 Eliminar Age negativa
titanic = titanic[titanic["Age"] >= 0]

# 5.2 Reemplazar Fare negativa por media por clase
fare_means = titanic[titanic["Fare"] >= 0].groupby("Pclass")["Fare"].mean()
for cls, mean_value in fare_means.items():
    mask = (titanic["Pclass"] == cls) & (titanic["Fare"] < 0)
    titanic.loc[mask, "Fare"] = mean_value

# Boxplot original
plt.figure(figsize=(8, 5))
titanic[outliers_cols].boxplot()
plt.show()

# Método IQR
def iqr(df, columns):
    df_w = df.copy()
    for col in columns:
        q1 = df_w[col].quantile(0.25)
        q3 = df_w[col].quantile(0.75)
        i = q3 - q1
        lower = q1 - 1.5 * i
        upper = q3 + 1.5 * i
        df_w[col] = np.clip(df_w[col], lower, upper)
    return df_w

no_outliers = iqr(titanic, outliers_cols)

# Boxplot tras IQR
plt.figure(figsize=(8, 5))
no_outliers[outliers_cols].boxplot()
plt.show()


# =============================================================================
# 5. FILTRADO Y AGREGACIONES
# =============================================================================
print("\n=== 5) Filtrado, duplicados y agregaciones ===")

df = no_outliers.copy()

# 5.1 Comprobar duplicados
num_duplicados = df.duplicated().sum()
print(f"\nNúmero de registros duplicados en el dataset: {num_duplicados}")

# En este dataset concreto, no hay duplicados. Si los hubiera, podríamos eliminarlos así:
# df = df.drop_duplicates()

# 5.2 Subconjuntos por sexo
df_male = df[df["Sex"] == "male"].copy()
df_female = df[df["Sex"] == "female"].copy()

# Estadísticas de edad
age_stats_male = df_male.groupby("Pclass")["Age"].agg(["mean", "min", "max"]).reset_index()
age_stats_female = df_female.groupby("Pclass")["Age"].agg(["mean", "min", "max"]).reset_index()

print("\nEdad male por clase:")
print(age_stats_male)
print("\nEdad female por clase:")
print(age_stats_female)

# Estadísticas de Fare
fare_stats_male = df_male.groupby("Pclass")["Fare"].agg(["mean", "std"]).reset_index()
fare_stats_female = df_female.groupby("Pclass")["Fare"].agg(["mean", "std"]).reset_index()

print("\nFare male por clase:")
print(fare_stats_male)
print("\nFare female por clase:")
print(fare_stats_female)

df_total = pd.concat([df_male, df_female], axis=0)

group_cols = ["Pclass", "Sex"]
agg_surv = df_total.groupby(group_cols).agg(
    Total_pasajeros=("Survived", "count"),
    Sobrevivientes=("Survived", "sum")
).reset_index()
agg_surv["%_Supervivencia"] = agg_surv["Sobrevivientes"] / agg_surv["Total_pasajeros"] * 100

print("\nSupervivencia por clase y sexo:")
print(agg_surv)

# ---------------------------------------------------------------------
# 7. AgeGroup + survival
# ---------------------------------------------------------------------
bins = [0, 12, 18, 35, 60]
labels = ["Niño (0-12)", "Adolescente (13-18)", "Joven (19-35)", "Adulto (36-60)"]
titanic["AgeGroup"] = pd.cut(titanic["Age"], bins=bins, labels=labels, right=True)

agegroup_surv = titanic.groupby(["AgeGroup", "Pclass"]).agg(
    Total=("Survived", "count"),
    Sobrevivientes=("Survived", "sum")
).reset_index()
agegroup_surv["%_Superv"] = agegroup_surv["Sobrevivientes"] / agegroup_surv["Total"] * 100

print("\nSupervivencia por AgeGroup y Pclass:")
print(agegroup_surv)

# ---------------------------------------------------------------------
# 8. Uniones
# ---------------------------------------------------------------------
meteo = meteo_raw.copy()

titanic["Barco"] = titanic_raw["Barco"]
meteo = meteo[["Nombre_Barco", "Clima", "TemperaturaMedia"]].copy()

titanic["Barco_upper"] = titanic["Barco"].str.upper()
meteo["Nombre_Barco_upper"] = meteo["Nombre_Barco"].str.upper()

df_merged = titanic.merge(
    meteo,
    left_on="Barco_upper",
    right_on="Nombre_Barco_upper",
    how="left"
)

print(df_merged[["Barco", "Nombre_Barco", "Clima", "TemperaturaMedia"]].head())

# ---------------------------------------------------------------------
# 9. Exportación final
# ---------------------------------------------------------------------
final_cols_order = [
    "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "AgeGroup", "Barco", "Clima", "TemperaturaMedia"
]
final_cols_order = [c for c in final_cols_order if c in df_merged.columns]

df_final = df_merged[final_cols_order].copy()

output_name = "Titanic_2_clean.csv"
df_final.to_csv(output_name, index=False, encoding="utf-8")

print(f"\nArchivo final guardado como: {output_name}")
print("Dimensiones finales:", df_final.shape)
