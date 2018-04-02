import utils2 as utils


def save_prev_words(prev_words):
    prev_words = {k: list(v) for k, v in prev_words.items()}
    utils.json_dump("data/ner/prev_words.json", prev_words)


# lấy các từ đứng trước các entity
def word_previous_entity(path):
    print(path)
    sentences = utils.parse(utils.read(path), ["\n\n", "\n", "\t"])
    output = {}
    for s in sentences:
        for w, prev_w in zip(s[1:], s[:-1]):
            if w[-2] != "O" and prev_w[-2] == 'O' and prev_w[0] not in utils.punctuations:
                tag = w[-2].split("-")[-1]
                if tag not in output:
                    output[tag] = set()
                output[tag].add(prev_w[0])
    return output


def main():
    words = word_previous_entity("data/ner/-Doi_song_train.muc")
    save_prev_words(words)


def update(prev_words, labeled_words, min_count=0):
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
                if tag not in prev_words:
                    prev_words[tag] = set()
                prev_words[tag].add(word)
    return prev_words


if __name__ == "__main__":
    main()