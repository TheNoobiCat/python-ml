import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from transformers import BertTokenizerFast, TFBertForSequenceClassification

fake = pd.read_csv('dataset/Fake.csv')
true = pd.read_csv('dataset/True.csv')

fake["label"] = 0
true["label"] = 1

fake = fake[["text", "label"]]
true = true[["text", "label"]]

dataset = pd.concat([fake, true], ignore_index=True)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

x = dataset["text"].tolist()
y = dataset["label"].tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Tokenizing data...")
x_train = tokenizer(x_train, truncation=True, padding=True)
x_test = tokenizer(x_test, truncation=True, padding=True)
print("Tokenizing data complete!")

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, weight_decay=0.01)
print("Compiling model...")
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print("Model compiled!")

print("Creating datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train)).batch(2)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(x_test), y_test)).batch(2)
print("Datasets created!")

print("Training model...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=3
)
print("Model trained!")

model.save_pretrained('model')