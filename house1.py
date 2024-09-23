import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Carregando o dataset
housing = fetch_california_housing()

# Criando o DataFrame com os dados e os nomes das colunas
casas = pd.DataFrame(housing.data, columns=housing.feature_names)

# Adicionando a coluna alvo (o preço médio das casas)
casas['MedHouseVal'] = housing.target

# Exibindo as primeiras linhas do DataFrame
print(casas.head())

# Plotando o gráfico de dispersão com a reta de regressão
plt.figure(figsize=(10, 6))
sns.regplot(x=casas['MedInc'], y=casas['MedHouseVal'], scatter_kws={'alpha':0.6}, line_kws={"color":"red"})

# Adicionando título e rótulos aos eixos
plt.title('Comparação entre Renda Mediana e Preço Médio das Casas na Califórnia')
plt.xlabel('Renda Mediana (MedInc)')
plt.ylabel('Preço Médio das Casas (MedHouseVal)')

# Exibindo o gráfico
plt.show()
