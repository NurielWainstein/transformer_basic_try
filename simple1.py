import torch
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch import nn

data = ['01/10', '02/10', '03/10', '04/10', '05/10', '06/10', '07/10', '08/10', '09/10', '10/10', '11/10', '12/10', '01/11', '02/11', '03/11', '04/11', '05/11', '06/11', '07/11', '08/11', '09/11', '10/11', '11/11', '11/11']
data2 = ['01/01/10', '01/02/10', '01/03/10', '01/04/10', '01/05/10', '01/06/10', '01/07/10', '01/08/10', '01/09/10', '01/10/10', '01/11/10', '01/12/10', '01/01/11', '01/02/11', '01/03/11', '01/04/11', '01/05/11', '01/06/11', '01/07/11', '01/08/11', '01/09/11', '01/10/11', '01/11/11', '01/11/11']
def preprocess_data(src):

    word_embeddings = np.zeros((1, 128))

    for i, char in enumerate(src):
        if char == "!":
            input = 10
        elif char == ".":
            input = 11
        elif char == ",":
            input = 12
        elif char == "-":
            input = 13
        elif char == "+":
            input = 14
        elif char == "/":
            input = 15
        elif char == "\\":
            input = 16
        elif char == " ":
            input = 17
        else:
            input = char
        word_embeddings[0][i] = input
    return torch.from_numpy(word_embeddings)

def replace_non_numeric_chars(string):
    new_string = ""
    for char in string:
        if char.isdigit() or char in "-/+.\\|, ":  # check if char is numeric or in allowed set
            new_string += char
        else:
            new_string += "!"
    return new_string

def add_zeros(string):
    while len(string) < 128:
        string += '0'
    return string

input_data = []
for d1, d2 in zip(data, data2):
    input_data.append((preprocess_data(add_zeros(replace_non_numeric_chars(d1))),preprocess_data(add_zeros(replace_non_numeric_chars(d2)))))

src = torch.zeros((1, 24, 128))
tgt = torch.zeros((1, 24, 128))
for i in range(len(input_data)):
    src[0][i] = input_data[i][0][0]
    tgt[0][i] = input_data[i][1][0]

embedding = nn.Embedding(num_embeddings=18, embedding_dim=128)
norm = nn.LayerNorm(128)
encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4,dropout=0.2, batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
model = nn.Transformer(d_model=128)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def embeding1(tensor):
    embeddings = []
    my_words = {}
    for vector in tensor:
        vector = torch.tensor(vector).to(torch.int64).squeeze()
        embeddings.append(embedding(vector.unsqueeze(0)))
        for i, emb in enumerate(embeddings[-1][0]):
            my_words[vector[i].item()] = emb
    return torch.cat(embeddings, dim=0), my_words

def embeding2(tensor):
    embeddings = []
    for vector in tensor:
        vector = torch.tensor(vector).to(torch.int64).squeeze()
        embeddings.append(embedding(vector.unsqueeze(0)))
    return torch.cat(embeddings, dim=0)

def decoder(embeddings_dict, input_tensor):
    """
    Given a dictionary of word embeddings and an input tensor,
    finds the most similar embedding in the dictionary and returns the corresponding key.

    Args:
    - embeddings_dict (dict): a dictionary mapping words to their embeddings (tensors of length 128)
    - input_tensor (tensor): a tensor of shape (128,) representing the input embedding

    Returns:
    - most_similar_word (str): the key in the dictionary that corresponds to the most similar embedding
    """

    # Convert the embeddings dictionary to a tensor of shape (num_words, 128)
    embeddings_tensor = torch.stack(list(embeddings_dict.values()))

    # Compute the cosine similarity between the input tensor and all embeddings in the dictionary
    similarities = torch.nn.functional.cosine_similarity(embeddings_tensor, input_tensor, dim=1)

    # Find the index of the most similar embedding
    most_similar_idx = similarities.argmax().item()

    # Find the corresponding word key in the dictionary
    most_similar_word = list(embeddings_dict.keys())[most_similar_idx]

    return most_similar_word

class MyDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        input_value = self.src[idx]
        expected_output = self.tgt[idx]
        return input_value, expected_output


my_dataset = MyDataset(src[0], tgt[0])
batch_size = 4
my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(10):
    epoch_loss = 0
    for i, batch in enumerate(my_dataloader):
        inputs, expected_outputs = batch
        # Forward pass
        target_seq = embeding2(expected_outputs)
        encoder_input, embedings_dict = embeding1(inputs)
        target_seq = norm(target_seq)
        mask = torch.zeros(4, 128)
        for batch in range(4):
            for emb in range(128):
                if (encoder_input[batch][emb] == embedings_dict[0]).all():
                    mask[batch][emb] = True
                else:
                    mask[batch][emb] = False
        out_encoder = transformer_encoder(encoder_input, src_key_padding_mask=mask)
        output = model(encoder_input, out_encoder)

        # # Calculate loss and backpropagate
        # for i, batch in enumerate(output):
        #     for j, vector in enumerate(batch):
        #         output[i][j] = decoder(embedings_dict, vector)

        loss = criterion(output, target_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Epoch {}, Loss: {}'.format(epoch + 1, epoch_loss))

test_string = "11/21"
with torch.no_grad():
    encoder_input = preprocess_data(add_zeros(replace_non_numeric_chars(test_string)))
    encoder_input, embedings_dict = embeding1(encoder_input)
    encoder_input = norm(encoder_input)
    out_encoder = transformer_encoder(encoder_input)
    output = model(encoder_input, out_encoder)

    output_string = ""
    for i in range(128):
        char = output[0][i]
        input = decoder(embedings_dict, char)
        output_string = output_string + str(input)

print(output_string)