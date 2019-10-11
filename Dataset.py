from collections import Counter
import tensorflow as tf
import tensorflow_datasets as tfds
import os


if __name__ == "__main__":
    def labeler(example, index):
        return example, tf.cast(index, tf.int64)


    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label


    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


    with open("Objective_dataset.txt", encoding="utf-8") as f:
        for line in f.readlines():
            nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "")\
                .replace("\"", "").replace("\n", "")
            with open('Treated_Objective_Dataset.txt', mode='a+', encoding="utf-8") as f2:
                f2.write(nline + '\n')

    labeled_data_sets = []

    lines_dataset = tf.data.TextLineDataset('Treated_Objective_Dataset.txt')
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, 1))  # forma o map com o objetivo e o label 1
    labeled_data_sets.append(labeled_dataset)

    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    TAKE_SIZE = 5000

    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

    for ex in all_labeled_data.take(5):
        print(ex)

    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    example_text = next(iter(all_labeled_data))[0].numpy()
    print(example_text)

    encoded_example = encoder.encode(example_text)
    print(encoded_example)

    all_encoded_data = all_labeled_data.map(encode_map_fn)

    train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

    test_data = all_encoded_data.take(TAKE_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

    vocab_size += 1

    # training

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(88000, 16))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

    fitModel = model.fit(train_data, epochs=40, batch_size=512, validation_data=test_data, verbose=1)
