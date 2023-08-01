import copy

import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import nltk
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer

torch.set_printoptions(threshold=10_000)
import random
import calendar
import string

def generate_samples_mm_yy(num_samples):
    input_data = []
    expected_output = []

    for _ in range(num_samples):
        # Generate random characters before and after the date
        chars_before = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=random.randint(0, 10)))
        chars_after = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=random.randint(0, 10)))

        # Generate random month and year
        month = random.randint(1, 12)
        year = random.randint(0, 99)

        # Get the last day of the month
        last_day = calendar.monthrange(2000 + year, month)[1]

        # Generate the input and output strings
        input_string = f"{chars_before}{month:02d}/{year:02d}{chars_after}"
        output_string = f"01/{month:02d}/{year:02d}-{last_day:02d}/{month:02d}/{year:02d}"

        input_data.append(input_string)
        expected_output.append(output_string)

    return input_data, expected_output

def generate_samples_dd_mm_yy(num_samples):
    input_data = []
    expected_output = []

    for _ in range(num_samples):
        # Generate random characters before and after the date
        chars_before = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=random.randint(0, 10)))
        chars_after = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=random.randint(0, 10)))

        # Generate random start and end dates
        start_day = random.randint(1, 28)
        start_month = random.randint(1, 12)
        start_year = random.randint(2000, 2099)

        end_day = random.randint(start_day, 28)
        end_month = random.randint(start_month, 12)
        end_year = random.randint(start_year, 2099)

        # Generate the input and output strings
        input_string = f"{chars_before}{start_day:02d}/{start_month:02d}/{start_year:04d}-{end_day:02d}/{end_month:02d}/{end_year:04d}{chars_after}"
        output_string = f"{start_day:02d}/{start_month:02d}/{start_year:04d}-{end_day:02d}/{end_month:02d}/{end_year:04d}"

        input_data.append(input_string)
        expected_output.append(output_string)

    return input_data, expected_output


def generate_samples_point_sep(num_samples):
    input_data = []
    expected_output = []

    for _ in range(num_samples):
        # Generate random characters before and after the dates
        chars_before = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=random.randint(0, 10)))
        chars_after = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=random.randint(0, 10)))

        # Generate random start and end dates
        start_day = random.randint(1, 28)
        start_month = random.randint(1, 12)
        start_year = random.randint(2000, 2099)

        end_day = random.randint(1, 28)
        end_month = random.randint(1, 12)
        end_year = random.randint(2000, 2099)

        # Generate the input and output strings
        input_string = f"{chars_before} {start_day:02d}.{start_month:02d}.{start_year:02d} {end_day:02d}.{end_month:02d}.{end_year:02d} {chars_after}"
        output_string = f"{start_day:02d}/{start_month:02d}/{start_year:04d}-{end_day:02d}/{end_month:02d}/{end_year:04d}"

        input_data.append(input_string)
        expected_output.append(output_string)

    return input_data, expected_output

# Generate 10,000 samples
#input_data_type1, expected_output_type1 = generate_samples_mm_yy(1000)
input_data_type2, expected_output_type2 = generate_samples_dd_mm_yy(20000)
# input_data_type3, expected_output_type3 = generate_samples_point_sep(10000)

input_data = input_data_type2
target_data = expected_output_type2


def create_testing_data(input_data, target_data):
    length = len(input_data)
    test_input = []
    test_target = []
    for i in range(int(length*0.1)):
        length = len(input_data)
        num = random.randint(0, length-1)
        test_input.append(input_data[num])
        test_target.append(target_data[num])
        del input_data[num]
        del target_data[num]
    return test_input, test_target


test_input, test_target = create_testing_data(input_data, target_data)

print("done_generating_data")

tokenizer_regex = RegexpTokenizer('\d+|/+|\.+|-+')
def tokenizer(input_src, output_tgt):
    words_ids = {}

    words = []
    data = zip(input_src, output_tgt)
    for s1, s2 in data:

        # input_words = tokenizer_regex.tokenize(s1)
        input_words = tokenizer_regex.tokenize(s1)
        output_words = tokenizer_regex.tokenize(s2)
        words.extend(input_words)
        words.extend(output_words)

    id = 0
    for word in words:
        if word not in words_ids.keys():
            words_ids[word] = id
            id += 1

    words_ids["<sos>"] = id
    words_ids["<eos>"] = id+1
    words_ids["<pad>"] = id+2
    words_ids["<unk>"] = id+3

    return words_ids


words_ids = tokenizer(input_data, target_data)

def convert_data_to_matrices(words_ids, data):
    for index_in_data, sentence in enumerate(data):
        empty_sentence = torch.full(size=(1, 64), fill_value=words_ids["<pad>"])

        empty_sentence[0][0] = words_ids["<sos>"]
        for index_in_word, word in enumerate(tokenizer_regex.tokenize(sentence)):
            if sentence in words_ids:
                empty_sentence[0][index_in_word+1] = words_ids[word]
            else:
                empty_sentence[0][index_in_word + 1] = words_ids[word]
        empty_sentence[0][index_in_word+2] = words_ids["<eos>"]

        data[index_in_data] = empty_sentence

convert_data_to_matrices(words_ids, input_data)
convert_data_to_matrices(words_ids, target_data)


def create_masking(data):
    masks = torch.zeros(len(data), 64)
    for i ,seq in enumerate(data[0]):
        mask = torch.zeros(64)
        padding_start = False
        for j, item in enumerate(seq):
            if item.item() == words_ids["<pad>"] or padding_start:
                padding_start = True
                mask[j] = True
            else:
                mask[j] = False
        masks[i] = mask
    return masks


input_data_masks = create_masking(input_data)
target_data_masks = create_masking(target_data)

input_data = torch.cat(input_data)
target_data = torch.cat(target_data)


class MyDataset(Dataset):
    def __init__(self, src, tgt, src_mask, tgt_mask):
        self.src = src
        self.tgt = tgt
        self.src_mask = src_mask
        self.tgt_mask = tgt_mask

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        input_value = self.src[idx]
        expected_output = self.tgt[idx]
        src_mask = self.src_mask[idx]
        tgt_mask = self.tgt_mask[idx]

        return input_value, expected_output, src_mask, tgt_mask


my_dataset = MyDataset(input_data, target_data, input_data_masks, target_data_masks)
batch_size = 4
my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=len(words_ids), d_model=64, nhead=4, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )
        self.softmax = nn.Softmax(-1)


    def forward(self, src, mask):

        input_seq = torch.tensor(src).to(torch.int64)

        encoder_input = self.embedding(input_seq)
        encoder_input = self.pos_encoder(encoder_input)

        out_encoder = self.transformer_encoder(encoder_input , src_key_padding_mask=mask )
        out_encoder = self.dropout(out_encoder)

        output = self.decoder(out_encoder)
        output = self.softmax(output)

        return output


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = TransformerModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

for epoch in range(10):
    epoch_loss = 0
    counter = 1
    last_step = len(my_dataloader)
    for batch in my_dataloader:
        if counter == last_step:
            continue
        counter += 1
        inputs, expected_outputs, input_mask, expected_outputs_mask = batch
        inputs = torch.transpose(inputs, 0, 1)

        inputs = inputs.to(device)
        input_mask = input_mask.to(device)

        encoded_target_data = torch.zeros(batch_size, 64, len(words_ids))
        for i, seq in enumerate(expected_outputs):
            for j, word in enumerate(seq):
                encoding = torch.zeros(len(words_ids))
                encoding[word] = 1
                encoded_target_data[i][j] = encoding.unsqueeze(0)

        encoded_target_data = encoded_target_data.to(device)

        encoded_target_data = torch.transpose(encoded_target_data, 0, 1)

        # expected_outputs=expected_outputs.float()
        # inputs=inputs.float()

        # forward pass
        output = model.forward(inputs, input_mask)
        # Calculate loss and backpropagate
        loss = criterion(output, encoded_target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Epoch {}, Loss: {}'.format(epoch + 1, epoch_loss))
torch.save(model.state_dict(), "transformer4_model.pt")




model.load_state_dict(torch.load("transformer4_model.pt"))
model.eval()

x = copy.copy(test_input)
test_data = test_input

with torch.no_grad():
    convert_data_to_matrices(words_ids, test_data)
    convert_data_to_matrices(words_ids, test_target)
    test_mask = create_masking(test_data)
    test_matrics = torch.cat(test_data)
    test_matrics_target = torch.cat(test_target)

    test_matrics = test_matrics.to(device)
    test_mask = test_mask.to(device)
    test_matrics_target = test_matrics_target.to(device)

    test_matrics_target = torch.transpose(test_matrics_target, 0, 1)
    test_matrics = torch.transpose(test_matrics, 0, 1)
    output = model(test_matrics , test_mask )
    output = output.argmax(-1).squeeze()

    def calculate_recall(output, expected_output):
        # Initialize an array to store recall values for each class
        recall_values = np.zeros(len(words_ids))

        # Loop through each class
        for c in words_ids.values():
            total_recal_for_class = 0
            for output_batch, batch_expected in zip(output, expected_output):
                # Find the indices of all instances of class c in the expected output
                true_indices = np.where(output_batch == c)[0]

                # Find the indices of all instances of class c in the output
                predicted_indices = np.where(batch_expected == c)[0]

                # Calculate the number of true positives
                true_positives = len(np.intersect1d(true_indices, predicted_indices))

                # Calculate the total number of positives
                total_positives = len(true_indices)

                # Calculate the recall for this class
                if total_positives == 0 and true_positives == 0:
                    recall = 1
                elif total_positives == 0:
                    recall = 0
                else:
                    recall = true_positives / total_positives

                # Store the recall value in the recall_values array
                total_recal_for_class += recall

            recall_values[c] = total_recal_for_class/expected_output.size()[0]
        # Return the recall values for all classes
        mean = sum(recall_values) / len(recall_values)
        return mean


    def calculate_Accuracy(output, expected_output):
        count = 0
        for batch_output, batch_expected in zip(output, expected_output):
            if (batch_output == batch_expected).all():
                count += 1
        return count/len(output)


    def calculate_precision(output, expected_output):
        # Initialize an array to store precision values for each class
        precision_values = np.zeros(len(words_ids))

        # Loop through each class
        for c in words_ids.values():
            total_precision_for_class = 0
            for output_batch, batch_expected in zip(output, expected_output):
                # Find the indices of all instances of class c in the expected output
                true_indices = np.where(output_batch == c)[0]

                # Find the indices of all instances of class c in the output
                predicted_indices = np.where(batch_expected == c)[0]

                # Calculate the number of true positives
                true_positives = len(np.intersect1d(true_indices, predicted_indices))

                # Calculate the total number of predicted positives
                total_predicted_positives = len(predicted_indices)

                # Calculate the precision for this class
                if total_predicted_positives == 0 and true_positives == 0:
                    precision = 1
                elif total_predicted_positives == 0:
                    precision = 0
                else:
                    precision = true_positives / total_predicted_positives

                # Store the precision value in the precision_values array
                total_precision_for_class += precision

            precision_values[c] = total_precision_for_class / expected_output.size()[0]
        # Return the precision values for all classes
        mean = sum(precision_values) / len(precision_values)
        return mean


    accuracy = calculate_Accuracy(torch.transpose(output, 0, 1).cpu(), torch.transpose(test_matrics_target, 0, 1).cpu())
    recal = calculate_recall(torch.transpose(output, 0, 1).cpu(), torch.transpose(test_matrics_target, 0, 1).cpu())
    precition = calculate_precision(torch.transpose(output, 0, 1).cpu(), torch.transpose(test_matrics_target, 0, 1).cpu())
    print(f"Training set size - {len(input_data)}")
    print(f"Testing set size - {int(len(input_data)*0.1)}")
    print(f'{accuracy} - accuracy')
    # print(f'{recal} - recall')
    # print(f'{precition} - precision')
    # print(f'{(2*precition*recal)/precition+recal} - F1')

    for index, i in enumerate(torch.split(output, 1, dim=-1)):
        output_string = ''
        for j in i.squeeze():
            for word in words_ids.keys():
                if j.item() == words_ids[word]:
                    output_string += word
        print(output_string)