import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Simulação de dados de transações financeiras
np.random.seed(42)
n_samples = 5000
data = {
    'valor_transacao': np.random.uniform(1, 10000, n_samples),
    'hora_transacao': np.random.randint(0, 24, n_samples),
    'pais_origem': np.random.choice(['EUA', 'Brasil', 'Alemanha', 'China', 'Índia'], n_samples),
    'pais_destino': np.random.choice(['EUA', 'Brasil', 'Alemanha', 'China', 'Índia'], n_samples),
    'cartao_virtual': np.random.choice([0, 1], n_samples),
    'fraude': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # Apenas 5% são fraudes
}

df = pd.DataFrame(data)

# Conversão de variáveis categóricas em numéricas
df = pd.get_dummies(df, columns=['pais_origem', 'pais_destino'])

# Separação entre atributos e rótulos
X = df.drop(columns=['fraude'])
y = df['fraude']

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de aprendizado de máquina
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação do modelo
y_pred = modelo.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))

# Função para prever fraude em uma nova transação
def prever_fraude(transacao):
    transacao_df = pd.DataFrame([transacao])
    transacao_df = pd.get_dummies(transacao_df)
    transacao_df = transacao_df.reindex(columns=X.columns, fill_value=0)
    return modelo.predict(transacao_df)[0]

# Testando a função de previsão
nova_transacao = {
    'valor_transacao': 5000,
    'hora_transacao': 14,
    'pais_origem': 'Brasil',
    'pais_destino': 'EUA',
    'cartao_virtual': 1
}
print("Fraude detectada:" if prever_fraude(nova_transacao) else "Transação segura")
