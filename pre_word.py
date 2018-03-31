import glob

def parse(content, separator):
    if len(separator) == 0:
        return content.strip()
    else:
        return [parse(c, separator[1:]) for c in content.strip().split(separator[0])]

# lấy các từ đứng trước các entity
def word_previous_entity(path):
    print(path)
    with open(path, "r", encoding="utf8") as f:
        sentences = parse(f.read(), ["\n\n", "\n", "\t"])
    words = {}
    for s in sentences:
        for i, w in enumerate(s):
            if w[-2] != "O" and i > 0 and s[i-1][-2] == 'O' and s[i-1][0] not in [",", ".", '“', '”', ":", "'", "(", ")", "–", "-", '"', "/"]:
                tag = w[-2][-3:]
                if tag not in words:
                    words[tag] = []
                words[tag].append(s[i-1][0])
    return words

def main():
    words = word_previous_entity("data/ner/ner_train.muc")
    words = set(words)
    for tag in words:
        with open("data/ner/prev_%s.txt" % tag, "w", encoding="utf8") as f:
            f.write("\n".join(words[tag]))


if __name__ == "__main__":
    main()