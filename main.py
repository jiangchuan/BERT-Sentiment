import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizerFast, TFBertForSequenceClassification, create_optimizer
import matplotlib.pyplot as plt

BATCH_SIZE = 8

dataset_id = 'imdb'
dataset = load_dataset(dataset_id)  # load_dataset is a function of datasets library of HuggingFace

dataset

pd.DataFrame({'text': dataset['train'][:5]['text'], 'label': dataset['train'][:5]['label']})

# Load pre-trained BERT model
model_id = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_id)

test_input_1 = 'The Weather of Today is Gréat! zwp'
test_input_2 = 'How are you doing?'
inputs = [test_input_1, test_input_2]

print(tokenizer.tokenize(inputs))

output = tokenizer(inputs, padding=True, truncation=True, max_length=128)
print(output)

# [CLS] token marks the beginning of the input sequence,
# [SEP] token marks the end of the input sequence.
# [PAD] tokens ensure that the sequence has the same length as other sequences in the batch.
tokenizer.decode(output['input_ids'][0])
tokenizer.decode(output['input_ids'][1])


def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, )


tokenized_dataset = dataset.map(preprocess_function, batched=True)
print(tokenized_dataset)

pd.DataFrame({
    'text': tokenized_dataset['train'][:5]['text'],
    'label': tokenized_dataset['train'][:5]['label'],
    'input_ids': tokenized_dataset['train'][:5]['input_ids'],
    'token_type_ids': tokenized_dataset['train'][:5]['token_type_ids'],
    'attention_mask': tokenized_dataset['train'][:5]['attention_mask']
})

# Train Dataset
tf_train_dataset = tokenized_dataset['train'].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
)

# Test Dataset
tf_val_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    # collate_fn=data_collator
)


def swap_positions(dataset):
    return {'input_ids': dataset['input_ids'],
            'token_type_ids': dataset['token_type_ids'],
            'attention_mask': dataset['attention_mask'], }, dataset['label']


# The .prefetch() method is used to asynchronously fetch batches of data from the dataset while the model is training on the current batch. This can help reduce data loading times and keep the GPU or CPU more utilized.
# tf.data.AUTOTUNE is a constant that indicates TensorFlow should automatically choose the number of batches to prefetch based on available resources and the execution context. This ensures optimal performance without manual tuning.
tf_train_dataset = tf_train_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)
tf_val_dataset = tf_val_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)

for i in tf_train_dataset.take(1):
    print(i)

# Modelling using TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
model.summary()

### Training
num_epochs = 1
# num_epochs = 3
batches_per_epoch = len(tokenized_dataset["train"]) // BATCH_SIZE
total_train_steps = int(batches_per_epoch * num_epochs)

# Optimizer
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

# Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy'], )

# Fit the model
history = model.fit(
    tf_train_dataset.take(1000),
    validation_data=tf_val_dataset,
    epochs=num_epochs)

# Plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot the accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Test
inputs = tokenizer([
    "this movie looks very interesting, i love the fact that the actors do a great job in showing how people lived in the 18th century, which wasn't very good at all. But atleast this movie recreates this scenes! ",
    "very good start, but movie started becoming uninteresting at some point though initially i thought it would have been much more fun. There was too much background noise, but later on towards the middle of the movie, my favorite character got in and he did a great job, so over "
], padding=True, return_tensors="tf")

logits = model(**inputs).logits
print(logits)

# Apply softmax to logits to get probabilities
probabilities = tf.nn.softmax(logits, axis=-1)

# Print the probabilities
print(probabilities)

#
##
#
##
#
##
#
##
#
##
#
##
#
##
#
##
#
##
#
##
#
##
#
##
#
