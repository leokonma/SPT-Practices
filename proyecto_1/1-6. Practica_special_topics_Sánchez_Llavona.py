import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\leodo\OneDrive\Escritorio\special topics\proyecto_1")

# =========================
# 0) Carga de datos
# =========================
df = pd.read_csv("titanic_1.csv")
df


# =========================
# 1) Revisión inicial
# =========================

#1.1: ¿Cuántos registros de pasajeros tiene el dataset y cuántas variables?
df.describe()

#1.2: ¿Qué tipo de datos tiene cada variable?
df.info()





# =========================
# 2) Estadísticas descriptivas iniciales 
# =========================

#2.1
df["Embarked"].value_counts()
df["Sex"].value_counts()

#2.2 
df["Embarked"].describe()
df["Sex"].describe()

#2.3: Dime las medias de todas las variables numéricas.
for _ in df.select_dtypes(include='number'):
    if _ != "PassengerId":
        print(f"Mean of {_}: {df[_].mean():.2f}")



# =========================
# 3) Distribución de la variable objetivo 
# =========================

#3.1: Transforma la variable Survived para que contenga las etiquetas: 0 → "No", 1 → "Sí".
df["Survived"] = df["Survived"].replace({0: "No", 1: "Sí"})


#3.2: Cuenta cuántos pasajeros sobrevivieron y cuántos no.
df["Survived"].value_counts()

#3.3:  Representa gráficamente esta distribución en un gráfico de barras.
counts = df["Survived"].value_counts()
colors = plt.cm.viridis([0.2, 0.8])
plt.bar(counts.index, counts.values, color=colors)
plt.xlabel("Survived")
plt.ylabel("Freq")
plt.title("Variable Survived distribution ")
plt.grid(axis='y', linestyle='-', alpha=0.7)
    # Mostrar valores encima de las barras
for i, v in enumerate(counts.values):
    plt.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.show()


# =========================
# 4) Enriquecimiento de la base de datos
# =========================


# 4.1: Crea una nueva columna LastName con el apellido del pasajero.
df[["Surname", "Rest"]] = df["Name"].str.split(",", n=1, expand=True)

# 4.2: Crea una nueva columna FirstName con el nombre propio del pasajero.
df["FirstName"] = df["Rest"].str.extract(r'\.\s*(.*)')

# 4.3: Extrae el título (Mr, Mrs, Miss, etc.) en una nueva columna Title. 
df["Title"] = df["Rest"].str.extract(r'([A-Za-z]+)\.')

# 4.4: Crea una columna FamilySize = SibSp + Parch + 1. 
df["Surname"] = df["Surname"].str.strip()
df["Title"] = df["Title"].str.strip()
df["FirstName"] = df["FirstName"].str.strip()
df.drop(columns=["Rest"], inplace=True)
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1


#4.5: A partir de ella, genera la variable IsAlone (1 si FamilySize == 1, 0 en caso contrario).
df["IsAlone"] = 0
df.loc[df["Family_Size"] == 1, "IsAlone"] = 1

#4.6: si tiene acompañante ya que su probabilidad de sobrevivir puede aumentar


# =========================
# 5) Enriquecimiento II
# =========================


#5.1: Calcula la edad de los pasajeros en una nueva columna Age, teniendo en cuenta la fecha de
# nacimiento y la fecha del suceso (10 de abril de 1912).

fecha_titanic = pd.Timestamp("1912-04-15")  # día del hundimiento
df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
df["Age"] = (fecha_titanic - df["BirthDate"]).dt.days / 365.25

#5.2: Ordena el dataset por Age de manera ascendente y descendiente
df.sort_values(by="Age", ascending=True, inplace=True)
print(df["Age"].head())
print(df["Name"].head())
df.sort_values(by="Age", ascending=False, inplace=True)
print(df["Age"].head())
print(df["Name"].head())


#5.3: ¿En qué mes se concentran más nacimientos?
import pandas as pd
df["BirthMonth"] = df["BirthDate"].dt.month_name()
max_month = df["BirthMonth"].value_counts().idxmax()
max_count = df["BirthMonth"].value_counts().max()
print(f"Mes con más nacimientos: {max_month} ({max_count} registros)")




#5.4: Crea una variable categórica AgeGroup
df["AgeGroup"] = ""
df.loc[(df["Age"] > 0) & (df["Age"] < 12), "AgeGroup"] = "Niño"
df.loc[(df["Age"] >= 12) & (df["Age"] < 18), "AgeGroup"] = "Adolescente"
df.loc[(df["Age"] >= 18) & (df["Age"] < 35), "AgeGroup"] = "Jovenes"
df.loc[(df["Age"] >= 35) & (df["Age"] < 60), "AgeGroup"] = "Adultos"
df.loc[df["Age"] >= 60, "AgeGroup"] = "Adulto mayor"


#5.5 . Cambia los valores de la variable Pclass: 1→1st, 2→ 2nd, 3→ 3rd.
df["Pclass"] = df["Pclass"].replace({1: "1 St", 2: "2 St",3:"3 St"})






# =========================
# 6) Análisis de las variables
# =========================


#6.1: Haz una tabla de contingencia entre Sex y Survived
contingency = pd.crosstab(df["Sex"], df["Survived"])
print(contingency)

#6.2: Haz un histograma de Age con 3 intervalos
plt.hist(df["Age"].dropna(), bins=3, edgecolor="black", color=plt.cm.viridis(0.6))
plt.title("Distribución de la Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

    #histograma con 25 intervalos 
plt.hist(df["Age"].dropna(), bins=25, edgecolor="black", color=plt.cm.viridis(0.6))
plt.title("Distribución de la Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


#6.3: Grafico de Barras para Embarked
counts = df["Embarked"].value_counts(dropna=False)  
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
plt.bar(counts.index.astype(str), counts.values, color=colors, edgecolor='black')
plt.title("Distribución de los Puertos de Embarque (Embarked)")
plt.xlabel("Puerto de Embarque")
plt.ylabel("Número de Pasajeros")
plt.xticks(rotation=0, ha="center")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(counts.values):
    plt.text(i, v + 2, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()


#6.4: Gráfico de barras para AgeGroup
counts = df["AgeGroup"].value_counts()
plt.bar(counts.index, counts.values, color=plt.cm.viridis(0.6), edgecolor='black')
plt.title("Distribución de los grupos de edad")
plt.xlabel("AgeGroup")
plt.ylabel("Frecuencia")

    # Mostrar los números encima de cada barra
for i, v in enumerate(counts.values):
    plt.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


#6.5: Haz un gráfico de dispersión Age y Fare, coloreando los puntos por la variable Survived. 
colors = df["Survived"].map({"No": "red", "Sí": "green"})
plt.scatter(df["Age"], df["Fare"], c=colors, alpha=0.6, edgecolor='black', linewidth=0.3)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Relación entre Age y Fare \n (verde = sobrevivió, rojo = no)")
plt.tight_layout()
plt.show()

#6.6: Calcula correlación de Pearson entre Age y Fare
corr = df["Age"].corr(df["Fare"], method="pearson")
print(f"Coeficiente de correlación de Pearson (Age vs Fare): {corr:.3f}")

#6.7: Si quisieras ver si existe una relación significativa entre Sex y Survived: CHI SQUARED
from scipy import stats
from scipy.stats import chi2_contingency
contingencia = pd.crosstab(df['Survived'], df['Sex'])
print("Tabla de contingencia:")
print(contingencia)

chi2_stat, p_val, dof, expected = chi2_contingency(contingencia)
print("\nChi2:", chi2_stat)
print("p-valor:", p_val)

if p_val < 0.05:
    print("Existe dependencia significativa entre Survived y Sex")
else:
    print("No hay evidencia de dependencia entre Survived y Sex")


#6.8: Si quieres analizar si la tarifa (Fare) varía significativamente entre los distintos puertos de
#embarque (Embarked): ANOVA(ONE WAY)
anova_results = stats.f_oneway(
    df[df['Embarked'] == 'C']['Fare'], 
    df[df['Embarked'] == 'S']['Fare'], 
    df[df['Embarked'] == 'Q']['Fare'],
)

print(f"F-statistic: {anova_results.statistic}")
print(f"P-value: {anova_results.pvalue}")

if p_val < 0.05:
    print("Existe dependencia significativa entre Fare y Embarked")
else:
    print("No hay evidencia de dependencia entre Fare y Embarked")



#6.9: Si quieres analizar si la tarifa (Fare) varía de forma significativa entre las clases (Pclass): ANOVA(ONE WAY)
anova_results2 = stats.f_oneway(
    df[df['Pclass'] == '1 St']['Fare'], 
    df[df['Pclass'] == '2 St']['Fare'], 
    df[df['Pclass'] == '3 St']['Fare'],
)

print(f"F-statistic: {anova_results2.statistic}")
print(f"P-value: {anova_results2.pvalue}")

if p_val < 0.05:
    print("Existe dependencia significativa entre Fare y Pclass")
else:
    print("No hay evidencia de dependencia entre Fare y Pclass")

# =========================
# 7) Análisis libre
# =========================

#7.1 Analisis PROBIT and LOGIT. Documento Aparte








