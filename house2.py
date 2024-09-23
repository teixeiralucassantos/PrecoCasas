import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Para utilizar este recurso experimental, precisamos solicitá-lo explicitamente:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

N_SPLITS = 5

rng = np.random.RandomState(0)

X_completo, y_completo = fetch_california_housing(return_X_y=True)
# ~2k amostras são suficientes para o propósito do exemplo.
# Remova as duas linhas a seguir para uma execução mais lenta com barras de erro diferentes.
X_completo = X_completo[::10]
y_completo = y_completo[::10]
n_amostras, n_features = X_completo.shape

# Estimando a pontuação no conjunto de dados completo, sem valores ausentes
estimador_br = BayesianRidge()
pontuacao_dados_completos = pd.DataFrame(
    cross_val_score(
        estimador_br, X_completo, y_completo, scoring="neg_mean_squared_error", cv=N_SPLITS
    ),
    columns=["Dados Completos"],
)

# Adicionando um único valor ausente a cada linha
X_ausentes = X_completo.copy()
y_ausentes = y_completo
amostras_ausentes = np.arange(n_amostras)
features_ausentes = rng.choice(n_features, n_amostras, replace=True)
X_ausentes[amostras_ausentes, features_ausentes] = np.nan

# Estimando a pontuação após imputação (estratégias média e mediana)
pontuacao_imputador_simples = pd.DataFrame()
for estrategia in ("mean", "median"):
    estimador = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy=estrategia), estimador_br
    )
    pontuacao_imputador_simples[estrategia] = cross_val_score(
        estimador, X_ausentes, y_ausentes, scoring="neg_mean_squared_error", cv=N_SPLITS
    )

# Estimando a pontuação após imputação iterativa dos valores ausentes
# com diferentes estimadores
estimadores = [
    BayesianRidge(),
    RandomForestRegressor(
        # Ajustamos os hiperparâmetros do RandomForestRegressor para obter um desempenho
        # preditivo razoável em um tempo de execução restrito.
        n_estimators=4,
        max_depth=10,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=0,
    ),
    make_pipeline(
        Nystroem(kernel="polynomial", degree=2, random_state=0), Ridge(alpha=1e3)
    ),
    KNeighborsRegressor(n_neighbors=15),
]
pontuacao_imputador_iterativo = pd.DataFrame()
# O imputador iterativo é sensível à tolerância e
# dependente do estimador usado internamente.
# Ajustamos a tolerância para manter esta execução do exemplo com recursos computacionais
# limitados, sem alterar muito os resultados em comparação com a manutenção do
# valor padrão mais restrito para o parâmetro de tolerância.
tolerancias = (1e-3, 1e-1, 1e-1, 1e-2)
for estimador_imputacao, tol in zip(estimadores, tolerancias):
    estimador = make_pipeline(
        IterativeImputer(
            random_state=0, estimator=estimador_imputacao, max_iter=25, tol=tol
        ),
        estimador_br,
    )
    pontuacao_imputador_iterativo[estimador_imputacao.__class__.__name__] = cross_val_score(
        estimador, X_ausentes, y_ausentes, scoring="neg_mean_squared_error", cv=N_SPLITS
    )

pontuacoes = pd.concat(
    [pontuacao_dados_completos, pontuacao_imputador_simples, pontuacao_imputador_iterativo],
    keys=["Original", "Imputador Simples", "Imputador Iterativo"],
    axis=1,
)

# plotando os resultados da habitação na Califórnia
fig, ax = plt.subplots(figsize=(13, 6))
medias = -pontuacoes.mean()
erros = pontuacoes.std()
medias.plot.barh(xerr=erros, ax=ax)
ax.set_title("Regressão da Habitação na Califórnia com Diferentes Métodos de Imputação")
ax.set_xlabel("MSE (quanto menor, melhor)")
ax.set_yticks(np.arange(medias.shape[0]))
ax.set_yticklabels([" w/ ".join(label) for label in medias.index.tolist()])
plt.tight_layout(pad=1)
plt.show()
