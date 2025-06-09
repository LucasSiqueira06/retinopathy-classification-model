from pathlib import Path

# Testar se uma imagem do CSV realmente existe
img_path = Path("resized_train/resized_train/10_left.jpeg")
print(img_path.exists())