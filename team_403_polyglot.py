#! /usr/bin/env python
# This works temporarily, and then google gets grumpy,..
# ...
# ...yes, really. Textblob backs onto google translate for language detection.
import json
import random
import sys
from polyglot.detect import Detector


def main():
    for line in sys.stdin:
        d = json.loads(line)
        try:
            detect = Detector(text=d['text'])
            result = detect.language.code
        except Exception:
            # ISO STANDARD UNKNOWN LANGUAGE
            result = 'sr'
        print(json.dumps({
            'example': d['example'],
            'lang': result,
        }))


if __name__ == "__main__":
    main()
