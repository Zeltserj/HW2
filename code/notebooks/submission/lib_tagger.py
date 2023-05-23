import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, Optional
from numpy import loadtxt
import matplotlib.pyplot as plt
import os 
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

EMBEDDING_DIM = 50

def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def mask_outside(pred:torch.TensorType, y:torch.TensorType, noisy_tag):
    # mask outside
    outside_tag_predicted_and_true = torch.logical_not(torch.logical_and(torch.eq(pred.argmax(1), y), torch.eq(y, noisy_tag)))
    return pred[outside_tag_predicted_and_true], y[outside_tag_predicted_and_true]

def train(dataloader, model, loss_fn, optimizer,  device, noisy_tag: Optional[str] = None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct, overall = 0, 0, 0
    model.train()
    for batch, (window, tag) in enumerate(dataloader):
        X  = window.to(device)
        y = tag.to(device)

        # Compute prediction error
        pred = model(X)
        if noisy_tag:
            masked_pred, masked_y = mask_outside(pred, y, noisy_tag)
        else:
            masked_pred, masked_y = pred, y
        loss = loss_fn(masked_pred,masked_y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss_fn(pred, y).item()
        if batch % 1000 == 0:
            logger.info(f"batch: {batch}\tpreds: {pred.argmax(1)}\ty: {y}")
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        overall += len(y)

    train_loss /= num_batches
    correct /= overall 
    logger.info(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

def write_test_results(path, model, data_loader, index_to_tag, index_to_word,sub_word):
    with open(path, 'w') as out_file:
        for X, _ in data_loader:
            pred = model(X)
            tag = index_to_tag[pred.argmax(1).item()]
            if sub_word:
                out_file.write(f"{index_to_word[X[0][2][1].item()]} {tag}\n")
            else:
                out_file.write(f"{index_to_word[X[0][2].item()]} {tag}\n")
        out_file.write('\n')            

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    with torch.no_grad():
        test_loss, correct, overall = 0,0,0
        for X, y in dataloader:
            X  = X.to(device)
            y = y.to(device)

            # Compute prediction error
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            overall += len(y)

        test_loss /= num_batches
        correct /= overall
        logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss


def save_loss_plot(epochs, losses, save_path):
    """
    Saves a plot of accuracy as a function of epochs.

    Args:
        epochs (list): List of epoch values.
        loss (list): List of accuracy values.
        save_path (str): Path to save the plot.
    """
    plt.plot(torch.arange(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation set loss as a function of epochs')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_accuracy_plot(epochs, accuracy, save_path):
    """
    Saves a plot of accuracy as a function of epochs.

    Args:
        epochs (list): List of epoch values.
        accuracy (list): List of accuracy values.
        save_path (str): Path to save the plot.
    """
    plt.plot(torch.arange(epochs), accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a Function of Epochs')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()



def write_plots_and_parameters(model, accs, losses, num_epochs, learning_rate, output_path, pre_trained):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    param_text =[
    f"Model configuration: {model}",
    f"Trained for {num_epochs} epochs",
    f"Using pretraiend embeddings: {pre_trained}",
    f"With learning rate {learning_rate}",
    f"Acheived final accuracy on dev set of {accs[-1]}"
    f"With loss {losses[-1]}",
    ]
    with open(f"{output_path}/parameters.txt", '+w') as f:
        f.write('\n'.join(param_text))

    save_accuracy_plot(num_epochs, accs, f"{output_path}/accuracy_plot.png")
    save_loss_plot(num_epochs, losses, f"{output_path}/loss_plot.png")


class WindowTagger(nn.Module):
    def __init__(self, vocab, in_dim, hid_dim,out_dim, pretrained_embeddings_path, sub_word=False, cnn=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(len(vocab), EMBEDDING_DIM)
        self.sub_word = sub_word
        self.vocab = vocab
        self.cnn=cnn
        if self.sub_word:
            self.prefix_embedding = nn.Embedding(len(vocab.prefixes), EMBEDDING_DIM)
            self.suffix_embedding = nn.Embedding(len(vocab.suffixes), EMBEDDING_DIM)
        if vocab.last_pretrained_idx:
            vectors = loadtxt(pretrained_embeddings_path)
            self.embedding.weight.data[:vocab.last_pretrained_idx] = torch.from_numpy(vectors)
        if self.cnn:
            self.char_embeddings = nn.Embedding(len(vocab.chars))

        self.mlp = nn.Sequential(
            nn.Linear(EMBEDDING_DIM*in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, out_dim)
        )
    
    def forward(self, x):
        if self.sub_word:
            pref = self.prefix_embedding(x[:, :, 0])
            word =  self.embedding(x[:, :, 1]) 
            suf = self.suffix_embedding(x[:, :, 2])
            embeddings = pref + word + suf
        else:
            embeddings = self.embedding(x)
        embeddings = embeddings.view(-1, embeddings.size(1)*embeddings.size(2))
        logits = self.mlp(embeddings)
        probs = nn.Softmax(1)(logits)
        return probs

        