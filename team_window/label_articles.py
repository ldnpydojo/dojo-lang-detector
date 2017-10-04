import sys
from grade import get_languages
import json

from collections import Counter, defaultdict


def letter_count(body):
    return Counter(body)

def normalise_count(counts):
    results = {}
    for char, count in counts.items():
        results[char] = count / sum(counts.values())
    return results


def agg(counters):
    return sum(counters, Counter())


def make_model(train_fn):
    lang_to_corpuses = defaultdict(list)

    for line in open(train_fn):
        data = json.loads(line)
        c = letter_count(data['text'])
        lang_to_corpuses[data['lang']].append(c)

    # print(lang_to_corpuses)
    normalised_model = {
        lang: normalise_count(agg(corpses))
        for lang, corpses in lang_to_corpuses.items()
    }

    # print(normalised_model)
    return normalised_model

def pythag(v1, v2):
    keys = set(v1) | set(v2)
    ds = []
    for k in keys:
        x1 = v1.get(k, 0)
        x2 = v2.get(k, 0)
        d = abs(x1 - x2) ** 2
        ds.append(d)
    return sum(ds)

def find_lang(test_text, model):
    vec = normalise_count(letter_count(test_text))
    ds = [[lang, pythag(vec, m)] for lang, m in model.items()]
    return min(ds, key=lambda lang_d: lang_d[1])[0]

def lang_me(test_fn, model):
    # languages = get_languages()
    for line in open(test_fn):
        data = json.loads(line)
        ex_num = data['example']
        text = data['text']
        lang = find_lang(text, model)
        # lang_name = languages[lang]
        # print(ex_num, lang, lang_name, repr(text))
        # print()
        # print()
        yield json.dumps({'example': ex_num, 'lang': lang})


def main():
    """
    python team_window/label_articles.py train_100.json test_100.json
    python team_window/label_articles.py train_200.json test_200.json
    """

    train_data_fn, test_data_fn = sys.argv[1:]

    model = make_model(train_data_fn)
    for answer in lang_me(test_data_fn, model):
        print(answer)


if __name__ == '__main__':
    main()
