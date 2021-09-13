import pandas
import pandas as pd

arquivo = pd.read_csv('C:/Users/mathe/OneDrive/Área de Trabalho/Machine Learning/wine_dataset.csv') # pegar arquivo

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

#print(arquivo)

y = arquivo['style'] # recebe a coluna style
x = arquivo.drop('style', axis=1) # recebe o resto dos dados

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3) # definimos o X e Y de teino e teste da maquina
                                                            # test_size pega 30% de todas as amostras para teste

# algoritmo machine learning
from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier() # criando o modelo
modelo.fit(x_treino, y_treino) # treina o algoritmo

#imprimindo os resultados
resultado = modelo.score(x_teste, y_teste) # pega os dados do x_teste com y_teste e o algoritmo preve o resultado
print('Acerto de:', resultado)

#print(y_teste[400:410])
#print('-------------------------------------------------------------------------------')
#print(x_teste[400:410])

print('=============================================================================')

#previsoes = modelo.predict(x_teste[400:403])

#print('previsao:',previsoes)


#print(x)
#print('-----------------')
#print(y)

