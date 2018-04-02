import utils2 as utils

UNIQUE_PREV_WORDS = True


def unique_prev_words(prev_words):
    return {k: list(set(v)) for k, v in prev_words.items()}


def prev_words_count(prev_words):
    return {k: len(v) for k, v in prev_words.items()}


# lấy các từ đứng trước các entity
def word_previous_entity(path):
    print(path)
    sentences = utils.parse(utils.read(path), ["\n\n", "\n", "\t"])
    output = {k: [] for k in ["ORG", "LOC", "PER", "MIC"]}
    for s in sentences:
        for w, prev_w in zip(s[1:], s[:-1]):
            if w[-2] != "O" and prev_w[-2] == 'O' and prev_w[0] not in utils.punctuations:
                tag = w[-2].split("-")[-1]
                output[tag].append(prev_w[0])
    if UNIQUE_PREV_WORDS:
        output = unique_prev_words(output)
    return output


def update(prev_words, sentences, min_count=0):
    """

    :param prev_words: prev_words.json hiện tại
    :param sentences:   [
                            [(w1, t1), (w2, t2), ...],   # sentence 1
                            ,...
                        ]
    :param min_count:
    :return: prev_words mới
    """
    counters = {}
    prev_words = {k: v[:] for k, v in prev_words.items()}  # copy
    for sen in sentences:
        for (word, tag), (prev_word, prev_tag) in zip(sen[1:], sen[:-1]):
            if tag != 'O' and prev_tag != 'O' and prev_word not in utils.punctuations:
                _tag = tag.split("-")[-1]
                if _tag not in counters:
                    counters[_tag] = {}
                if prev_word not in counters[_tag]:
                    counters[_tag][prev_word] = 0
                counters[_tag][prev_word] += 1
    for tag, words_count in counters.items():
        for word, count in words_count.items():
            if count > min_count:
                prev_words[tag].append(word)
    if UNIQUE_PREV_WORDS:
        prev_words = unique_prev_words(prev_words)
    return prev_words


def main():
    words = word_previous_entity("data/ner/-Doi_song_train.muc")
    print(prev_words_count(words))
    utils.json_dump("data/ner/prev_words.json", words)


def main2():
    prev_words = utils.json_load("data/ner/prev_words.json")
    test_data = utils.parse(utils.read("data/ner/Doi_song_test.muc"), ["\n\n", "\n", "\t"])
    test_data = [[(w[0], w[-2]) for w in s] for s in test_data]
    new_prev_words = update(prev_words, test_data)
    print("old: " + str(prev_words_count(prev_words)))
    print("new: " + str(prev_words_count(new_prev_words)))


if __name__ == "__main__":
    main()
    # main2()