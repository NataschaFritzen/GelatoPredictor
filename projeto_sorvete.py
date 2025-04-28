# projeto_sorvete.py

# 1. Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

# 2. Criar os dados (Temperatura vs Vendas)
dados = {
    "Temperatura": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    "Vendas": [200, 220, 240, 265, 280, 300, 320, 340, 360, 390, 400]
}

df = pd.DataFrame(dados)

# 3. Treinar o modelo
X = df[['Temperatura']]
y = df['Vendas']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Fazer uma previsão
temperatura_dia = 26  # Exemplo: temperatura hoje
venda_prevista = modelo.predict([[temperatura_dia]])
print(f"Se a temperatura for {temperatura_dia}°C, a previsão de vendas é {venda_prevista[0]:.0f} sorvetes.")

# 5. Plotar o gráfico
plt.scatter(X, y, color='blue')
plt.plot(X, modelo.predict(X), color='red')
plt.title('Temperatura vs Vendas de Sorvete')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas')
plt.show()

# 6. Registrar o modelo no MLflow
mlflow.set_experiment("Previsao_Vendas_Sorvete")

with mlflow.start_run():
    mlflow.sklearn.log_model(modelo, "modelo_vendas_sorvete")
    mlflow.log_param("temperatura_dia", temperatura_dia)
    mlflow.log_metric("venda_prevista", venda_prevista[0])

print("Modelo registrado no MLflow!")
