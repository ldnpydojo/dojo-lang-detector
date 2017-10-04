from collections import Counter

from label_articles import normalise_count, letter_count

def test_letter_count():
    body = 'qwertye'
    expected_counts = {
        'q': 1,
        'w': 1,
        'e': 2,
        'r': 1,
        't': 1,
        'y': 1,
    }
    assert letter_count(body) == Counter(expected_counts)


def near_to(a, b, d):
    return abs(b - a) < d

def test_normalise_count():
    denormalised_counts = Counter({
        'q': 1,
        'w': 1,
        'e': 2,
        'r': 1,
        't': 1,
        'y': 1,
    })

    result = normalise_count(denormalised_counts)
    print(result)
    assert near_to(result['q'], 1 / 7.0, 0.1)
    assert near_to(result['e'], 2 / 7.0, 0.1)
    assert near_to(sum(result.values()), 1, 0.1)

