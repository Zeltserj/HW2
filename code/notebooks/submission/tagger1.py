import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
from lib_tagger import WindowTagger, get_device, train, test, write_test_results, write_plots_and_parameters
from vocab import Vocab, TagData
import logging
import click

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


@click.command()
@click.option("--train-path", type=str, help="Path to the training data")
@click.option("--dev-path", type=str, help="Path to the development data")
@click.option("--test-path", type=str, default=None, help="Path to the test data")
@click.option("--num-epochs", type=int, default=10, help="Number of epochs to train")
@click.option("--batch-size", type=int, default=8, help="Batch size")
@click.option("--learning-rate", type=float, default=0.03, help="Learning rate")
@click.option(
    "--task",
    type=click.Choice(["pos", "ner"]),
    default="pos",
    help='Task type ("pos" or "ner")',
)
@click.option("--hidden-dim", type=int, default=2000, help="Hidden dimension size")
@click.option(
    "--output-path",
    type=str,
    default="test.out",
    help="Path to write the test results to",
)
@click.option(
    "--pretrained-weights-path",
    type=str,
    default=None,
    help="Path for file containing weights of pre-trained words",
)
@click.option(
    "--pretrained-words-path",
    type=str,
    default=None,
    help="Path for file containing pre-trained words",
)
@click.option(
    "--results-path",
    type=str,
    default=None,
    help="Path of directory to save plots and parameters",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="Which device to use. Can be either cpu or cuda. If not given, the program will decide according to availbality",
)
@click.option(
    "--sub-word",
    type=bool,
    default=False,
    is_flag=True,
    help="Whether to use sub word units or not",
)
def main(
    train_path: str,
    dev_path: str,
    test_path: Optional[str] = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    task: str = "pos",
    hidden_dim=2000,
    output_path: str = "test.out",
    pretrained_weights_path: Optional[str] = None,
    pretrained_words_path: Optional[str] = None,
    results_path: Optional[str] = None,
    device: str = None,
    sub_word: bool = False,
):
    """
    Trains a neural network model on a specified task using the given dataset.

        Args:
            train_path (str): Path to the training data.
            dev_path (str): Path to the development data.
            test_path (Optional[str]): Path to the test data. Default is None.
            num_epochs (int): Number of epochs to train. Default is 10.
            batch_size (int): Batch size. Default is 8.
            learning_rate (float): Learning rate. Default is 0.03.
            task (str): Task type. Must be one of ["pos", "ner"]. Default is "pos".
            hidden_dim (int): Hidden dimension size. Default is 2000.
            output_path (str): Path to write the test results to. Default is "test.out".
            pretrained_weights_path (Optional[str]): Path for the file containing weights of pre-trained words. Default is None.
            pretrained_words_path (Optional[str]): Path for the file containing pre-trained words. Default is None.
            device (str): Which device to use. Can be "cpu" or "cuda". If not given, the program will decide according to availability. Default is None.
            sub_word (bool): If set, data will be broken down to sub word units. Default is false
        """

    if not device:
        device = get_device()
    seperator = " " if task == "pos" else "\t"

    logger.info(f"Using device {device}")
    logger.info("Fetching vocabulary and tags from %s", train_path)

    vocab = Vocab(train_path,pretrained_word_path=pretrained_words_path,seperator=seperator, sub_word=sub_word)
    logger.info(
        f"Vocabulary contains {len(vocab.words)} words. There are {len(vocab.tags)} possible tags"
    )
    noisy_tag = vocab.T2I[vocab.OUTSIDE] if task == "ner" else None

    logger.info("Initializing training and validation datasets...")
    train_dataset = TagData(train_path, vocab, seperator,sub_word=sub_word)
    dev_dataset = TagData(dev_path, vocab, seperator,sub_word=sub_word)

    train_loader = DataLoader(train_dataset, batch_size)
    dev_loader = DataLoader(dev_dataset, batch_size)

    logger.info("Done initializaing training and dev datasets")
    if test_path:
        logger.info("Initializing test data...")
        test_dataset = TagData(test_path, vocab, separator=seperator, test=True,sub_word=sub_word)
        test_dataloader = DataLoader(test_dataset)

    logger.info("Initializing mode and optimizer")
    model = WindowTagger(
        vocab,
        in_dim=5,
        hid_dim=hidden_dim,
        out_dim=len(vocab.tags),
        pretrained_embeddings_path=pretrained_weights_path,
        sub_word=sub_word
    )
    logger.info(f"model:\n{model}")
    logger.info("Initalizing optimizer")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    accs, losses = [], []
    for t in range(num_epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            noisy_tag=noisy_tag,
        )
        acc, loss = test(dev_loader, model, loss_fn, device)
        accs.append(acc)
        losses.append(loss)
    logger.info(f"Training done! Final Accuracy: {acc}, loss: {loss}")

    if test_path:
        write_test_results(
            output_path, model, test_dataloader, vocab.I2T, vocab.I2W, seperator
        )
    if results_path:
        write_plots_and_parameters(model, accs, losses,num_epochs,learning_rate,results_path,pre_trained=True if pretrained_weights_path else False)

if __name__ == "__main__":
    main()
