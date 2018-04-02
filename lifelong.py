import utils2 as utils

UNIQUE_PREV_WORDS = True


def unique_prev_words(prev_words):
    return {k: list(set(v)) for k, v in prev_words.items()}


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


def update(prev_words, labeled_words, min_count=0):
    """

    :param prev_words: prev_words.json hiện tại
    :param labeled_words: [(word1, ner_tag1), (word2, ner_tag2), ...]
    :param min_count:
    :return: prev_words mới
    """
    counters = {}
    for word, tag, prev_word, prev_tag in zip(labeled_words[1:], labeled_words[:-1]):
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
    utils.json_dump("data/ner/prev_words.json", words)


if __name__ == "__main__":
    main()
