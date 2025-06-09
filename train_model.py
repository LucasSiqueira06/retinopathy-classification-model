import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 📁 Caminhos
project_root = Path(__file__).resolve().parent
image_dir = project_root / "resized_train"
train_csv = project_root / "train_split.csv"
test_csv = project_root / "test_split.csv"

# ⚙️ Parâmetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 📄 Carregar os CSVs
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

for df in [train_df, test_df]:
    df['filename'] = df['image'] + ".jpeg"
    df['class'] = df['level'].astype(str)

# 🔄 Data Augmentation para treino
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

# 🔀 Geradores
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = test_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=image_dir,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================
# MODELO 1: CNN DO ZERO
# =========================
def build_custom_cnn():
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n🔧 Treinando CNN personalizada do zero...\n")
custom_model = build_custom_cnn()
custom_history = custom_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# =========================
# MODELO 2: TRANSFER LEARNING (MobileNetV2)
# =========================
def build_mobilenet():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(5, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n🔧 Treinando MobileNetV2 (Transfer Learning)...\n")
mobilenet_model = build_mobilenet()
mobilenet_history = mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# =========================
# AVALIAÇÃO
# =========================
def evaluate_model(model, name):
    print(f"\n📊 Avaliação do modelo: {name}")
    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(test_gen.labels, y_pred_prob, multi_class='ovr'))

    return y_true, y_pred, y_pred_prob

# Avaliar os dois modelos
evaluate_model(custom_model, "CNN Personalizada")
evaluate_model(mobilenet_model, "MobileNetV2 (Transfer Learning)")

# =========================
# GRÁFICOS DE TREINAMENTO
# =========================
def plot_history(hist, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Acc')
    plt.plot(hist.history['val_accuracy'], label='Val Acc')
    plt.title(f'Acurácia - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Val Loss')
    plt.title(f'Perda - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(custom_history, 'CNN Personalizada')
plot_history(mobilenet_history, 'MobileNetV2')
