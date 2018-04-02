import json

def parse(content, separator):
    if len(separator) == 0:
        return content.strip()
    else:
        return [parse(c, separator[1:]) for c in content.strip().split(separator[0])]


punctuations = ["~", "`", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", "{", "[", "}", "]", "|",
                "\\", ":", ";", "'", '"', "<", ",", ">", ".", "?", "/"] \
               + ['“', '”', "–"]  # ký tự không có trên bàn phím


def write(path, string):
    with open(path, "w", encoding="utf8") as f:
        f.write(string)


def read(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()


def json_dump(path, data):
    json.dump(data, open(path, "w", encoding="utf8"), ensure_ascii=False)


def json_load(path):
    return json.load(open(path, "r", encoding="utf8"))
