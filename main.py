import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent

csv_path = project_root / "trainLabels.csv"

df = pd.read_csv(csv_path)

def sample_per_class(df, n, seed=42):
    return df.groupby('level', group_keys=False).apply(lambda x: x.sample(n=n, random_state=seed))

train_df = sample_per_class(df, 708, seed=42)

test_samples = []
for cls in range(5):
    n = 3000 if cls == 0 else 708
    cls_samples = df[df['level'] == cls].sample(n=n, random_state=100)
    test_samples.append(cls_samples)

test_df = pd.concat(test_samples)

train_df.to_csv(project_root / "train_split.csv", index=False)
test_df.to_csv(project_root / "test_split.csv", index=False)

print("Divisão concluída.")
print("Treino:", train_df['level'].value_counts().to_dict())
print("Teste:", test_df['level'].value_counts().to_dict())
