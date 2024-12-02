import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import joblib

# Importação
df = pd.read_csv("translated1400.csv")

# Combinar 'subject' e 'body' em uma única coluna
df['combined_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

# Mapear a coluna de rótulo para valores numéricos
df['label'] = df['label'].map({1: 1, 0: -1})

# Garantir que a coluna seja string e substituir NaN por string vazia
df['combined_text'] = df['combined_text'].fillna('').astype(str)

# Baixar stopwords (se necessário)
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Pré-processamento de texto
def preprocess_text(text):
    text = text.lower()  # Convertendo para minúsculas
    text = re.sub(r'\W', ' ', text)  # Removendo caracteres especiais
    text = re.sub(r'\s+', ' ', text)  # Removendo espaços extras
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Removendo stopwords
    return text

df['cleaned_text'] = df['combined_text'].apply(preprocess_text)

# Verificar os resultados
print(df[['combined_text', 'cleaned_text']].head())

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Encoder para os rótulos
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Dados de treino e validação em formato DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Parâmetros do modelo
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}

# Inicializar dicionário para armazenar resultados
evals_result = {}

# Treinar o modelo
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=True
)

# Resultados de treino e validação
train_loss = evals_result['train']['logloss']
val_loss = evals_result['validation']['logloss']

# Plotar o gráfico
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'r', label='Erro de Treinamento')
plt.plot(epochs, val_loss, 'b', label='Erro de Validação')
plt.title('Erro de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Log Loss')
plt.legend()
plt.show()

# Avaliar o modelo
y_pred = model.predict(xgb.DMatrix(X_test))
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_binary))

# Salvar o modelo
joblib.dump(model, 'modelo_phishing.joblib')
print("Modelo salvo com sucesso!")
