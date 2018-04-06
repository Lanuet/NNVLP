from datetime import datetime

import theano
import network
import numpy as np
import os
from utils2 import ObjectDict, get_name, reversed_dict
from glob import glob
import config


def load_data(folder, domain):
    data = ObjectDict()
    data.num_data = None
    for path in glob("%s/%s_*.npy" % (folder, domain)):
        name = get_name(path)
        name = name.replace(domain, "")[1:]
        data[name] = np.load(path)
        if name == "mask":
            data[name] = data[name].astype(theano.config.floatX)
        if data.num_data is None:
            data.num_data = data[name].shape[0]
        else:
            assert len(data[name]) == data.num_data
    return data

def main():
    print("Loading embeddings...")
    data = ObjectDict()
    for path in glob("embedding/*.npy"):
        name = get_name(path)
        data[name] = np.load(path).astype(theano.config.floatX)
    encoder = ObjectDict.load("embedding/encoder.json")
    data.max_sen_len = encoder.max_sen_len
    data.max_word_len = encoder.max_word_len
    data.num_labels = len(encoder.ner2idx)
    label_decoder = reversed_dict(encoder.ner2idx)
    del encoder

    print('Building model...')
    ner_model = network.build_model(data)

    print("Loading test & dev data")
    test_dir = "data/ner/test"
    test_domain = "Doi_song"
    test_data = load_data(test_dir, test_domain)
    dev_dir = "data/ner/dev"
    dev_domain = "Doi_song"
    dev_data = load_data(dev_dir, dev_domain)

    train_dir = "data/ner/train"
    train_domain = ["Doi_song"]
    os.makedirs("output/ner", exist_ok=True)
    for domain in train_domain:
        start_time = datetime.now()
        print("Loading data domain %s" % domain)
        train_data = load_data(train_dir, domain)

        print('Training model...')
        ner_model.fit([train_data.words, train_data.pos, train_data.chars], train_data.ner, batch_size=config.batch_size, epochs=config.num_epochs)
        end_time = datetime.now()
        print("Running time:")
        print(end_time - start_time)


if __name__ == '__main__':
    main()
