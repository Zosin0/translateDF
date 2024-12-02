import pandas as pd

# Caminhos dos arquivos traduzidos
body_files = ["Ling_body - 1.txt", "Ling_body - 2.txt"]
subject_files = ["Ling_subject - 1.txt", "Ling_subject - 2.txt"]
label_files = ["Ling_label - 1.txt", "Ling_label - 2.txt"]

# Função para carregar e concatenar os arquivos
def load_and_combine(files):
    combined_data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            combined_data.extend(f.read().strip().split("\n"))
    return combined_data

# Carregar os dados
bodies = load_and_combine(body_files)
subjects = load_and_combine(subject_files)
labels = load_and_combine(label_files)

# Verificar se todos os arquivos têm o mesmo número de linhas
if not (len(bodies) == len(subjects) == len(labels)):
    raise ValueError("Os arquivos não têm o mesmo número de linhas!")

# Criar o DataFrame
data = {
    "subject": subjects,
    "body": bodies,
    "label": labels
}
df = pd.DataFrame(data)

# Adicionar o cabeçalho e salvar o CSV
output_csv_path = "eu_amo_minha_vida.csv"
df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Dataset traduzido salvo como {output_csv_path}")

# Exibir informações do dataset criado
print(df.info())
print(df.head())
