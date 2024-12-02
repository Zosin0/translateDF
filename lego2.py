import pandas as pd

# Caminhos dos novos arquivos para adicionar
new_body_files = ["slices/pt3/b1.txt", "slices/pt3/b2.txt", "slices/pt3/b3.txt", "slices/pt3/b4.txt", "slices/pt3/b5.txt"]
new_subject_file = "slices/pt3/subject.txt"
new_label_file = "slices/pt3/label.txt"

# Função para carregar e concatenar os arquivos
def load_and_combine(files):
    combined_data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            combined_data.extend(f.read().strip().split("\n"))
    return combined_data

# Carregar os novos dados
new_bodies = load_and_combine(new_body_files)
new_subjects = load_and_combine([new_subject_file])
new_labels = load_and_combine([new_label_file])

# Verificar se todos os arquivos têm o mesmo número de linhas
if not (len(new_bodies) == len(new_subjects) == len(new_labels)):
    raise ValueError("Os arquivos não têm o mesmo número de linhas!")

# Criar DataFrame com os novos dados
new_data = {
    "subject": new_subjects,
    "body": new_bodies,
    "label": new_labels
}
new_df = pd.DataFrame(new_data)

# Carregar o dataset existente
existing_file_path = "translated1400.csv"  # Substitua pelo caminho correto
existing_dataset = pd.read_csv(existing_file_path)

# Concatenar os novos dados ao dataset existente
updated_dataset = pd.concat([existing_dataset, new_df], ignore_index=True)

# Salvar o dataset atualizado
updated_csv_path = "translated2800.csv"
updated_dataset.to_csv(updated_csv_path, index=False, encoding="utf-8")

print(f"Dataset atualizado salvo como {updated_csv_path}")
