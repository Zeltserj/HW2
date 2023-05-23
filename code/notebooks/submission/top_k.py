import numpy as np
import click


def gen_vocab(path):
    vocab = {}
    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            vocab[line.strip()] = i
    rev_vocab = {v: k for k, v in vocab.items()}
    return vocab, rev_vocab


def get_vecotrs(path):
    return np.loadtxt("/Users/zeltserj/Documents/MSc/TextDL/HW2/wordVector.txt")


def cosine_similarity(word, other):
    nominator = np.dot(word, other)
    denominator = np.sqrt(np.dot(word, word)) * np.sqrt(np.dot(other, other))
    return nominator / denominator


def most_similar(word, vectors, vocab, k):
    out = []
    w_vec = vectors[vocab[word]]
    similarities = []
    for i, v in enumerate(vectors):
        similarities.append((i, cosine_similarity(w_vec, v)))
    out = sorted(similarities, key=lambda x: x[1], reverse=True)[1 : k + 1]
    return out


@click.command()
@click.option("--embeddings-path", "-e", type=str, default="wordVector.txt")
@click.option("--vocab-path", "-v", type=str, default="vocab.txt")
@click.option(
    "--words",
    "-w",
    type=str,
    multiple=True,
    default=["dog", "england", "john", "explode", "office"],
)
@click.option("--num-words", "-k", type=int, default=5)
def main(words, embeddings_path, vocab_path, num_words):
    vocab, rev_vocab = gen_vocab(vocab_path)
    vectors = get_vecotrs(embeddings_path)
    print(words)
    for w in words:
        print(f"5 most similar words to {w}:\n")
        sim = most_similar(w, vectors, vocab, num_words)
        for idx, dist in sim:
            print(f"\t{rev_vocab[idx]} with distance: {dist}\n")
        print("\n")


if __name__ == "__main__":
    main()
