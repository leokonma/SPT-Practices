
"""
Práctica 2 – Bloque 2
Autores: Leonardo Sánchez Castillo y [Nombre de la pareja]

Objetivo:
Pipeline ETL completo sobre el dataset Titanic_2:
    1) Identificación de variables poco relevantes
    2) Análisis y tratamiento de valores perdidos (2 métodos de imputación)
    3) Corrección de errores tipográficos en variables categóricas
    4) Análisis y tratamiento de outliers
    5) Filtrado, agregaciones y análisis de supervivencia
    6) Unión con condiciones meteorológicas
    7) Exportación de la base final y reflexión (comentada)
"""

# =============================================================================
# 0. IMPORTACIÓN DE LIBRERÍAS Y CARGA DE DATOS
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import difflib

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


# --- Opcional: fijar estilo de gráficos
sns.set(style="whitegrid")


# --- Carga de datos (prueba dos nombres de archivo por seguridad)
def load_titanic_data():
    possible_names = ["Titanic_2.csv", "Titanic__2.csv", "Titanic_2 (1).csv"]
    for name in possible_names:
        if os.path.exists(name):
            print(f"✅ Cargando dataset Titanic desde: {name}")
            return pd.read_csv(name)
    raise FileNotFoundError(
        "No se encontró el archivo Titanic_2.csv / Titanic__2.csv / Titanic_2 (1).csv "
        "Asegúrate de que el nombre coincide con el del script."
    )


def load_meteo_data():
    name = "condiciones_meteorologicas.csv"
    if os.path.exists(name):
        print(f"✅ Cargando dataset meteorológico desde: {name}")
        return pd.read_csv(name)
    else:
        raise FileNotFoundError(
            "No se encontró condiciones_meteorologicas.csv en el directorio de trabajo."
        )


titanic_raw = load_titanic_data()
meteo_raw = load_meteo_data()

print("\n--- Vista inicial del Titanic ---")
print(titanic_raw.head())
print("\nDimensiones Titanic:", titanic_raw.shape)


# =============================================================================
# 1. VARIABLES POCO RELEVANTES
# =============================================================================
"""
Aquí identificamos variables que probablemente:
    - No aportan valor predictivo
    - Son identificadores puros
    - Tienen demasiados missing o ruido contextual
"""

print("\n=== 1) Variables poco relevantes ===")
print("Columnas originales:", list(titanic_raw.columns))

# Justificación (como comentarios en código):
# - PassengerId: identificador puro -> no aporta información sobre la supervivencia.
# - Name: contiene información útil pero muy compleja de explotar (texto libre).
#         Para un primer modelo supervisado suele tratarse como no relevante directamente,
#         salvo que se extraigan títulos, apellidos, etc. (no es el foco aquí).
# - Ticket: identificador de billete, difícil de interpretar.
# - Cabin: > 70% de NaN en este dataset; podría ser útil pero la calidad es muy mala.
# - Barco: en este dataset sólo toma el valor "Titanic" para todos los registros,
#          por lo que no tiene variación y no aporta información predictiva.

variables_poco_relevantes = ["PassengerId", "Name", "Ticket", "Cabin", "Barco"]
print("Variables consideradas poco relevantes o problemáticas:", variables_poco_relevantes)

titanic = titanic_raw.copy()


# =============================================================================
# 2. VALORES PERDIDOS
# =============================================================================
print("\n=== 2) Análisis de valores perdidos ===")

# Porcentaje de valores perdidos por variable
missing_pct = titanic.isnull().mean() * 100
print("\nPorcentaje de NaN por variable (%):")
print(missing_pct)

# Evaluar eliminar registros con ANY NaN
total_rows = len(titanic)
rows_complete = len(titanic.dropna())
pct_remaining = rows_complete / total_rows * 100
pct_lost = 100 - pct_remaining

print(f"\nSi eliminamos todas las filas con algún NaN quedarían "
      f"{rows_complete}/{total_rows} registros ({pct_remaining:.2f}%)")
print(f"Se perdería aproximadamente el {pct_lost:.2f}% de los registros.")

# Decisión:
# Perder casi un 80% de las observaciones no es razonable; preferimos imputar
# y eliminar sólo aquellas columnas con demasiados NaN (por ejemplo Cabin).

# Eliminamos las columnas con alta proporción de NaN y algunas poco relevantes
cols_to_drop_for_missing = ["Cabin"]  # muy > 70% NaN
cols_to_drop_for_relevance = ["PassengerId", "Name", "Ticket", "Barco"]

titanic_reduced = titanic.drop(columns=cols_to_drop_for_missing + cols_to_drop_for_relevance)
print("\nColumnas tras eliminar Cabin y variables poco relevantes:")
print(list(titanic_reduced.columns))

# --- 2.1 Imputación basada en estadísticos (media/mediana/moda) ---

# Separamos numéricas y categóricas
num_cols = titanic_reduced.select_dtypes(include=["int64", "float64"]).columns
cat_cols = titanic_reduced.select_dtypes(include=["object"]).columns

print("\nColumnas numéricas:", list(num_cols))
print("Columnas categóricas:", list(cat_cols))

# Usaremos mediana para las numéricas (menos sensible a outliers)
simple_imputer_num = SimpleImputer(strategy="median")
simple_imputer_cat = SimpleImputer(strategy="most_frequent")

titanic_simple = titanic_reduced.copy()

titanic_simple[num_cols] = simple_imputer_num.fit_transform(titanic_simple[num_cols])
titanic_simple[cat_cols] = simple_imputer_cat.fit_transform(titanic_simple[cat_cols])

print("\n¿Quedan NaN tras la imputación simple?",
      titanic_simple.isnull().any().any())

print("\nResumen numérico tras imputación simple:")
print(titanic_simple[num_cols].describe())


# --- 2.2 Imputación iterativa (MICE / IterativeImputer) ---

# Aplicamos IterativeImputer sólo sobre las numéricas
iter_imputer = IterativeImputer(random_state=42)
titanic_iter = titanic_reduced.copy()

titanic_iter[num_cols] = iter_imputer.fit_transform(titanic_iter[num_cols])
titanic_iter[cat_cols] = simple_imputer_cat.fit_transform(titanic_iter[cat_cols])

print("\n¿Quedan NaN tras la imputación iterativa?",
      titanic_iter.isnull().any().any())

print("\nComparación rápida de medias (Edad) simple vs iterativa:")
print("Edad (simple):   ", titanic_simple["Age"].mean())
print("Edad (iterativa):", titanic_iter["Age"].mean())

# Comentario (en código):
# Para este dataset pequeño, la imputación basada en estadísticos (mediana para Age
# y moda para Embarked) es suficiente y más interpretable. La imputación iterativa
# puede ser útil cuando hay muchas variables correlacionadas, pero añade complejidad.
# En el resto del análisis seguiremos trabajando con titanic_simple.


# =============================================================================
# 3. CORRECCIONES ORTOGRÁFICAS (variables categóricas / texto)
# =============================================================================
print("\n=== 3) Correcciones ortográficas ===")

titanic_clean = titanic_simple.copy()

# Vemos los valores de Sex antes de corregir
print("\nValores únicos de Sex ANTES de corregir:")
print(titanic_clean["Sex"].value_counts())

# Algunos valores con errores: "femalee", "malee", "mal", "femle"
# Usamos difflib.get_close_matches para corregir hacia ['male', 'female']

def correct_categorical_typos(series, valid_values, cutoff=0.7):
    """
    Corrige errores tipográficos en una serie categórica
    aproximando cada valor a la opción más cercana dentro de valid_values.
    """
    corrected = []
    for value in series.astype(str):
        matches = difflib.get_close_matches(value, valid_values, n=1, cutoff=cutoff)
        if matches:
            corrected.append(matches[0])
        else:
            corrected.append(value)
    return pd.Series(corrected, index=series.index)


# Corregimos Sex
titanic_clean["Sex"] = correct_categorical_typos(
    titanic_clean["Sex"],
    valid_values=["male", "female"],
    cutoff=0.6
)

print("\nValores únicos de Sex DESPUÉS de corregir:")
print(titanic_clean["Sex"].value_counts())

# Corregimos Embarked por si hubiese errores (C, Q, S)
titanic_clean["Embarked"] = correct_categorical_typos(
    titanic_clean["Embarked"],
    valid_values=["C", "Q", "S"],
    cutoff=0.6
)

print("\nValores únicos de Embarked tras corrección:")
print(titanic_clean["Embarked"].value_counts())


# =============================================================================
# 4. VALORES EXTREMOS (OUTLIERS)
# =============================================================================
print("\n=== 4) Análisis de outliers ===")

numeric_cols_for_outliers = ["Age", "SibSp", "Parch", "Fare"]

print("\nResumen inicial de variables numéricas clave:")
print(titanic_clean[numeric_cols_for_outliers].describe())

# Gráfico recomendado: boxplot por variable numérica
plt.figure(figsize=(8, 5))
titanic_clean[numeric_cols_for_outliers].boxplot()
plt.title("Boxplots de variables numéricas (Age, SibSp, Parch, Fare)")
plt.tight_layout()
plt.show()

# A partir de los boxplots sabemos:
# - Age: algunos valores altos (hasta 80), pero plausibles.
# - SibSp y Parch: pocos valores extremos (familias muy grandes).
# - Fare: valores muy altos (512) que pueden influir mucho en el análisis estadístico.

# Decisión de tratamiento:
# En lugar de eliminar filas (perderíamos observaciones), aplicamos una
# "winsorización" con el criterio IQR para limitar el impacto de outliers.

def winsorize_iqr(df, columns):
    df_w = df.copy()
    for col in columns:
        q1 = df_w[col].quantile(0.25)
        q3 = df_w[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        # Recortamos los valores extremos al rango [lower, upper]
        df_w[col] = np.clip(df_w[col], lower, upper)
        print(f"\n[{col}] - límites IQR: ({lower:.2f}, {upper:.2f})")
    return df_w


titanic_no_out = winsorize_iqr(titanic_clean, numeric_cols_for_outliers)

print("\nResumen tras winsorización:")
print(titanic_no_out[numeric_cols_for_outliers].describe())

plt.figure(figsize=(8, 5))
titanic_no_out[numeric_cols_for_outliers].boxplot()
plt.title("Boxplots tras tratamiento de outliers")
plt.tight_layout()
plt.show()


# =============================================================================
# 5. FILTRADO Y AGREGACIONES
# =============================================================================
print("\n=== 5) Filtrado, duplicados y agregaciones ===")

df = titanic_no_out.copy()

# 5.1 Comprobar duplicados
num_duplicados = df.duplicated().sum()
print(f"\nNúmero de registros duplicados en el dataset: {num_duplicados}")

# En este dataset concreto, no hay duplicados. Si los hubiera, podríamos eliminarlos así:
# df = df.drop_duplicates()

# 5.2 Subconjuntos por sexo
df_male = df[df["Sex"] == "male"].copy()
df_female = df[df["Sex"] == "female"].copy()

print("\nDimensiones df_male:", df_male.shape)
print("Dimensiones df_female:", df_female.shape)

# 5.3 Agregaciones por Pclass para cada subconjunto

# Edad: media, mínima y máxima
age_stats_male = df_male.groupby("Pclass")["Age"].agg(["mean", "min", "max"]).reset_index()
age_stats_female = df_female.groupby("Pclass")["Age"].agg(["mean", "min", "max"]).reset_index()

print("\nEdad (male) por Pclass:")
print(age_stats_male)
print("\nEdad (female) por Pclass:")
print(age_stats_female)

# Fare: media y desviación estándar
fare_stats_male = df_male.groupby("Pclass")["Fare"].agg(["mean", "std"]).reset_index()
fare_stats_female = df_female.groupby("Pclass")["Fare"].agg(["mean", "std"]).reset_index()

print("\nTarifa (male) por Pclass:")
print(fare_stats_male)
print("\nTarifa (female) por Pclass:")
print(fare_stats_female)

# Comentario (en código):
# - En general, las tarifas medias son mayores en 1ª clase que en 3ª, tanto en hombres como en mujeres.
# - Esto refleja la fuerte relación entre Pclass y el poder adquisitivo.
# - También se observan diferencias de edad entre clases (por ejemplo, pasajeros algo más mayores en 1ª).

# 5.4 Unión de subconjuntos en df_total
df_total = pd.concat([df_male, df_female], axis=0)
print("\nDimensiones df_total (male + female):", df_total.shape)

# 5.5 Agregación por Pclass y Sex: total pasajeros, sobrevivientes, % supervivencia

group_cols = ["Pclass", "Sex"]

agg_surv = df_total.groupby(group_cols).agg(
    Total_pasajeros=("Survived", "count"),
    Sobrevivientes=("Survived", "sum")
).reset_index()

agg_surv["%_Supervivencia"] = agg_surv["Sobrevivientes"] / agg_surv["Total_pasajeros"] * 100

print("\nTabla resumen supervivencia por Pclass y Sex:")
print(agg_surv)

# Comentario:
# - Normalmente se observa mayor % de supervivencia en mujeres que en hombres
#   dentro de la misma clase.
# - También suele haber una clara ventaja en 1ª clase frente a 3ª.


# 5.6 Filtro y agregación adicional "interesante"
# Ejemplo: analizar supervivencia por grupos de edad y clase

# Creamos una variable AgeGroup
bins = [0, 12, 18, 35, 60, 120]
labels = ["Niño (0-12)", "Adolescente (13-18)", "Joven (19-35)", "Adulto (36-60)", "Mayor (60+)"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

agegroup_surv = df.groupby(["AgeGroup", "Pclass"]).agg(
    Total=("Survived", "count"),
    Sobrevivientes=("Survived", "sum")
).reset_index()
agegroup_surv["%_Superv"] = agegroup_surv["Sobrevivientes"] / agegroup_surv["Total"] * 100

print("\nSupervivencia por AgeGroup y Pclass:")
print(agegroup_surv.sort_values(["AgeGroup", "Pclass"]))

# Este análisis aporta información sobre qué grupos de edad y clase
# concentran mayor supervivencia (por ejemplo, niños en 1ª clase).


# =============================================================================
# 6. UNIONES CON CONDICIONES METEOROLÓGICAS
# =============================================================================
print("\n=== 6) Unión con condiciones meteorológicas ===")

meteo = meteo_raw.copy()

print("\nDataset meteorológico:")
print(meteo.head())

# En titanic_original, la columna es 'Barco' (borrada en titanic_reduced).
# La recuperamos desde el dataset original para poder hacer el merge.
df["Barco"] = titanic_raw["Barco"]

# Preparamos claves de unión homogeneizando mayúsculas
meteo_small = meteo[["Nombre_Barco", "Clima", "TemperaturaMedia"]].copy()
meteo_small["Barco_upper"] = meteo_small["Nombre_Barco"].str.upper()
df["Barco_upper"] = df["Barco"].str.upper()

df_merged = pd.merge(
    df,
    meteo_small[["Barco_upper", "Clima", "TemperaturaMedia"]],
    on="Barco_upper",
    how="left"
)

# Comprobamos unión
print("\nComprobación de unión (columnas añadidas):")
print(df_merged[["Barco", "Clima", "TemperaturaMedia"]].head())

# Comentario:
# En este caso todos los pasajeros pertenecen al mismo barco ("Titanic"),
# por lo que Clima y TemperaturaMedia no introducen variación entre observaciones.
# No aportarán poder predictivo adicional para un modelo de supervivencia,
# pero sí contextualizan el entorno del viaje.


# =============================================================================
# 7. EXPORTACIÓN FINAL Y REFLEXIÓN
# =============================================================================
print("\n=== 7) Exportación de la base final ===")

final_cols_order = [
    "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "AgeGroup", "Barco", "Clima", "TemperaturaMedia"
]

# Nos aseguramos de que sólo exportamos columnas que existan
final_cols_order = [c for c in final_cols_order if c in df_merged.columns]

df_final = df_merged[final_cols_order].copy()

output_name = "Titanic_2_clean.csv"
df_final.to_csv(output_name, index=False, encoding="utf-8")
print(f"\n✅ Archivo final guardado como: {output_name}")
print("Dimensiones de la base final:", df_final.shape)


"""
Reflexión final (máx. 15 líneas):

Principales problemas de calidad de datos:
- Presencia de valores perdidos, especialmente en Age (~20%) y de forma masiva en Cabin (~77%).
- Errores tipográficos en variables categóricas como Sex ("femalee", "mal", etc.).
- Valores extremos en variables numéricas como Fare, SibSp o Parch que pueden distorsionar estadísticas.
- Variables poco informativas o con exceso de ruido (PassengerId, Name, Ticket, Cabin).

Decisiones tomadas:
- Eliminamos Cabin por su alto porcentaje de NaN y descartamos identificadores puros (PassengerId, Ticket, Name).
- Imputamos valores faltantes con mediana (numéricas) y moda (categóricas), y contrastamos con un método iterativo.
- Corregimos categorías mediante similitud de texto (difflib) en lugar de hacerlo manualmente.
- Tratamos outliers con winsorización basada en el IQR para mantener el tamaño muestral.
- Construimos agregaciones por Pclass, Sex y grupos de edad para entender patrones de supervivencia.
- Unimos información meteorológica para completar el contexto del viaje, aunque sin impacto predictivo real al no variar entre pasajeros.

Impacto en un modelo predictivo:
- La imputación permite aprovechar la mayoría de observaciones sin introducir sesgos extremos,
  aunque siempre existe el riesgo de infraestimar la incertidumbre.
- La corrección de typos evita crear categorías artificiales y mejora la interpretación de coeficientes.
- El tratamiento de outliers reduce la influencia de casos atípicos en modelos sensibles (por ejemplo, regresiones).
- En conjunto, las decisiones de limpieza deberían mejorar la estabilidad y precisión de un modelo de supervivencia,
  manteniendo un equilibrio razonable entre calidad de datos y tamaño muestral.
"""

if __name__ == "__main__":
    print("\n✅ Ejecución del script completada.")
