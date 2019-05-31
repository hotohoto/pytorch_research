#%%

import math
import os
import os.path
import pprint
import random
import string
import tarfile
import time
import unicodedata
import urllib.request
import zipfile
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import typer.etl.downloader as downloader

#%%

writer = SummaryWriter()

#%%: Set seeds
SEED = 82
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#%%

def prepare_dataset():
    all_dataset_path = []

    zip_file_path = os.path.join("data", "movie_dialogs.zip")
    downloader.maybe_download(
        "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
        zip_file_path,
    )
    path = downloader.maybe_unzip(zip_file_path)
    all_dataset_path.append(
        os.path.join(path, "cornell movie-dialogs corpus", "movie_lines.txt")
    )

    def bundle_dataset(files):
        def filter (x):
            cols = x.split(" +++$+++ ")
            if cols and len(cols) >= 5:
                return cols[4]
            else:
                return ""

        _files = [
            {
                "name": os.path.split(f)[1],
                "path": f,
                "size": os.stat(f).st_size,
                "handle": None,
                "filter": filter,
            }
            for f in files
        ]
        total_size = reduce((lambda x, y: x + y), map((lambda x: x["size"]), _files))
        return {"files": _files, "total_size": total_size}

    all_dataset_bundle = bundle_dataset(all_dataset_path)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(all_dataset_bundle)
    return all_dataset_bundle


all_dataset = prepare_dataset()

#%%

end_letter = "$"
_all_letters = "abcdefghijklmnopqrstuvwxyz .,'"
all_letters = _all_letters + end_letter
n_letters = len(all_letters)


#%%


def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def indexToLetter(index):
    return all_letters[index]


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def textToTensor(text):
    tensor = torch.zeros(len(text), 1, n_letters)
    for li, letter in enumerate(text):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def textToIndexTensor(text):
    tensor = torch.zeros(len(text), 1, dtype=torch.long)
    for li, letter in enumerate(text):
        tensor[li][0] = letterToIndex(letter)
    return tensor


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
# O'Néàl => O'Neal
def transform(s):
    s = s.lower()
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in _all_letters
    )
    s = " ".join(s.split())
    s = s.replace(" .", ".").replace(" ,", ",")
    return s


print(transform("O'Néàl\":    , , ."))


#%%


print(all_letters, n_letters)
print(letterToIndex("a"))
print(letterToTensor("b"))
print(textToTensor("ab"))
print(textToIndexTensor("ab"))


#%%
def next_inputs(bundled_dataset):
    """Fetch next input text

    Returns:
    None: if there is nothing to fetch
    [str]: Fetched sentences. No control symbol included in any sentences.
    """

    files = bundled_dataset["files"]
    total_size = bundled_dataset["total_size"]

    file_idx = np.random.choice(len(files), p=[f["size"] / total_size for f in files])

    if not files[file_idx]["handle"]:
        files[file_idx]["handle"] = open(files[file_idx]["path"], encoding="utf-8")

    max_lines_to_check = 50
    min_length = 100

    text = []
    total_length = 0

    for i in range(max_lines_to_check):
        try:
            line = files[file_idx]["handle"].readline()
        except:
            print("! {}".format(files[file_idx]["handle"].tell()))
            continue
        if not line:
            files[file_idx]["handle"].close()
            files[file_idx]["handle"] = None
            if text:
                return text
            else:
                return None
        if line:
            line = files[file_idx]["filter"](line).strip()
            if line:
                line = transform(line).strip()
                if line:
                    text.append(line)
                    total_length += len(line)
                    if total_length > min_length:
                        break
    if text:
        return text
    else:
        return None

def close_dataset_files(bundled_dataset):
    files = bundled_dataset["files"]

    for f in files:
        if f["handle"] is not None:
            f["handle"].close()
            f["handle"] = None


print(next_inputs(all_dataset))
print(next_inputs(all_dataset))
print(next_inputs(all_dataset))
print(next_inputs(all_dataset))
close_dataset_files(all_dataset)

#%%

class LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        print("LSTM init()", (input_size + output_size) * output_size * 4)
        super(LSTM, self).__init__()

        self.wf = nn.Linear(input_size + output_size, output_size)
        self.wi = nn.Linear(input_size + output_size, output_size)
        self.wc = nn.Linear(input_size + output_size, output_size)
        self.wo = nn.Linear(input_size + output_size, output_size)

    def forward(self, input, output_in, context_in):
        input_combined = torch.cat((input, output_in), 1)
        f = torch.sigmoid(self.wf(input_combined))
        i = torch.sigmoid(self.wi(input_combined))
        c = torch.tanh(self.wc(input_combined))
        o = torch.sigmoid(self.wo(input_combined))

        ic = torch.mul(i, c)

        context_out = torch.mul(context_in, f) + ic
        output_out = torch.mul(torch.tanh(context_out), o)

        return context_out, output_out


class MyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MyNet, self).__init__()

        self.hidden_size = hidden_size

        self.lstm1 = LSTM(n_letters, hidden_size)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = LSTM(hidden_size, n_letters)
        self.dropout2 = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, output_in, hidden_in, context_in):
        context1_out, hidden_out = self.lstm1(
            input, hidden_in, context_in[0][: self.hidden_size]
        )
        hidden_out = self.dropout1(hidden_out)
        context2_out, output = self.lstm2(
            hidden_out, output_in, context_in[0][self.hidden_size :]
        )
        output_ = self.dropout2(output)
        output_out = self.softmax(output_)
        return output_out, hidden_out, torch.cat((context1_out, context2_out), 1)


hidden_size = 512
my_net = MyNet(n_letters, n_letters, hidden_size)


#%%


learning_rate = (
    0.005
)  # If you set this too high, it might explode. If too low, it might not learn
n_iters = 200000
print_every = 1000
plot_every = 200
current_loss = 0
all_losses = []


#%%


optimizer = optim.Adam(my_net.parameters(), lr=learning_rate)


def train(text):
    sentences = end_letter.join(text) + end_letter
    context = torch.zeros(1, hidden_size + n_letters)  # memory
    hidden = torch.zeros(1, hidden_size)
    output = torch.zeros(1, n_letters)
    loss = torch.zeros(1)
    text_tensor = textToTensor(sentences)
    text_index_tensor = textToIndexTensor(sentences)
    outputs = ""
    n_iteration = len(sentences) - 1

    my_net.zero_grad()

    for i in range(n_iteration):
        output, hidden, context = my_net(text_tensor[i], output, hidden, context)
        _, idx = output.max(1)
        c = all_letters[idx[0].data]
        outputs += c
        l = F.nll_loss(output, text_index_tensor[i + 1])
        loss += l

    loss = loss / n_iteration
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    #     for p in lstm.parameters():
    #         p.data.add_(-learning_rate, p.grad.data)
    optimizer.step()

    return outputs, loss.item()


#%%:


def sample(context=torch.zeros(1, hidden_size + n_letters)):
    output_text = ""
    hidden = torch.zeros(1, hidden_size)
    output = torch.zeros(1, n_letters)
    input_ = torch.zeros(1, n_letters)

    n_max_len = 500

    for i in range(n_max_len):
        tmp_output, hidden, context = my_net(input_, output, hidden, context)
        val, idx = torch.max(tmp_output, 1)
        c = all_letters[idx[0].data]
        output_text += c
        if c == end_letter:
            break

        output = tmp_output
        input_ = textToTensor(c)[0]

    return output_text, context


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


start = time.time()


#%%


current_loss = 0
for i in range(1, n_iters + 1):
    text = next_inputs(all_dataset)
    if not text:
        continue
    outputs, loss = train(text)
    current_loss += loss

    # Print iter number, loss, name and guess
    if i % print_every == 0:
        print(
            "[%d] %d%% (%s) %.4f" % (i, i / n_iters * 100, timeSince(start), loss)
        )
        print("- TO BE: %s" % text)
        print("- AS IS: %s" % outputs)
        sampled_text = sample()[0]
        print("- SAMPLE: %s" % sampled_text)
        print("- LOSS: %s" % current_loss)
        writer.add_text('data/ground_truth', str(text), i)
        writer.add_text('data/outputs', outputs, i)
        writer.add_text('data/sample', sampled_text, i)

    # Add current loss avg to list of losses
    if i % plot_every == 0:
        average_loss = current_loss / plot_every
        all_losses.append(average_loss)
        writer.add_scalar('data/loss', average_loss, i)
        current_loss = 0

close_dataset_files(all_dataset)

#%%: Sampling


context = torch.zeros(1, hidden_size + n_letters)
for i in range(10):
    output_text, context = sample(context=context)
    print(output_text)

#%%: Save result as a file

model_name = "text_gen_model_{}.pt".format(int(time.time()))
model_path = os.path.join("data", model_name)
torch.save(my_net.state_dict(), model_path)


model_symlink_path = os.path.join("data", "text_gen_model.pt")
try:
    os.remove(model_symlink_path)
except:
    pass
os.symlink(model_name, model_symlink_path)

writer.close()

# References
# - https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
# - https://github.com/lanpa/tensorboardX
# - http://karpathy.github.io/2015/05/21/rnn-effectiveness/

