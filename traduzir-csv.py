import pandas as pd

dataset = pd.read_csv('datasets\Ling.csv')


dataset['subject'] = dataset['subject'].fillna("").astype(str)
dataset['label'] = dataset['label'].astype(str)
dataset['body'] = dataset['body'].fillna("").astype(str)

subject_txt_path = "english slices\Ling_subject.txt"
with open(subject_txt_path, "w", encoding="utf-8") as subject_file:
    for line in dataset['subject']:
        subject_file.write(line + "\n")

label_txt_path = "english slices\Ling_label.txt"
with open(label_txt_path, "w", encoding="utf-8") as label_file:
    for line in dataset['label']:
        label_file.write(line + "\n")


body_txt_path = "english slices\Ling_body.txt"
with open(body_txt_path, "w", encoding="utf-8") as body_file:
    for line in dataset['body']:
        body_file.write(line + "\n")


print(f"Arquivos para tradução criados")