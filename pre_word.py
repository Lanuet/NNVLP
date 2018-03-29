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
    words = []
    for s in sentences:
        for i, w in enumerate(s):
            if w[-2] != "O" and i > 0 and s[i-1][-2] == 'O':
                words.append(s[i-1][0])
    new_path = path + ".prev"
    with open(new_path, "w", encoding="utf8") as f:
        f.write("\n".join(words))
    return words

def main():
    files = glob.glob("data/ner/**.txt") + glob.glob("data/ner/**.muc")
    words = []
    for f in files:
        words += word_previous_entity(f)
    words = set(words)
    with open("data/ner/total.prev", "w", encoding="utf8") as f:
        f.write("\n".join(words))


if __name__ == "__main__":
    main()