import argparse
import json
import random


def get_languages():
    return json.load(open('languages.json'))


def parse_test_file(test_file):
    return {
        d['example']: d['text']
        for d in (json.loads(l) for l in open(test_file))
    }


def parse_answer_file(answer_file):
    return {
        d['example']: d['lang']
        for d in (json.loads(l) for l in open(answer_file))
    }


def grade(reference_answer_file, team_answer_file, test_file):
    reference_answers = parse_answer_file(reference_answer_file)
    team_answers = parse_answer_file(team_answer_file)
    test_data = parse_test_file(test_file)
    languages = get_languages()

    max_score = len(reference_answers)

    correct_answers = []
    incorrect_answers = []
    blank_answers = []
    for example, answer in team_answers.items():
        test_text = test_data[example]
        correct_answer = reference_answers[example]
        if answer is None:
            blank_answers.append((answer, correct_answer, test_text))
        elif answer == correct_answer:
            correct_answers.append((answer, correct_answer, test_text))
        else:
            incorrect_answers.append((answer, correct_answer, test_text))

    score = len(correct_answers) - len(incorrect_answers)
    print("Final score {}/{}!".format(score, max_score))
    print("Correct answers: {}!".format(len(correct_answers)))
    print("Incorrect answers: {}!".format(len(incorrect_answers)))
    print("Blank answers: {}!".format(len(blank_answers)))

    print("Examples of correct answers from team:")
    for answer, correct_answer, text in random.sample(correct_answers, 5):
        print("({}) {}".format(languages[answer], text))
        print()

    print("Examples of incorrect answers from team:")
    for answer, correct_answer, text in random.sample(incorrect_answers, 5):
        try:
            print("(team answer: {}, correct answer: {}) {}".format(languages[answer], languages[correct_answer], text))
            print()
        except KeyError:
            print("team answer: {} is not a recognised language code".format(answer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grade dojo lang detector')
    parser.add_argument('reference_answer_file', type=str)
    parser.add_argument('team_answer_file', type=str)
    parser.add_argument('test_file', type=str, default='')

    args = parser.parse_args()
    print(args.reference_answer_file)
    print(args.team_answer_file)
    grade(
        args.reference_answer_file,
        args.team_answer_file,
        args.test_file,
    )

