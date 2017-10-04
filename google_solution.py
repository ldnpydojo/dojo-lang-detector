import json
import sys
import os
import requests

KEY = os.environ.get('API_KEY')
GOOGLE_URL = 'https://translation.googleapis.com/language/translate/v2/detect'

def get_language(text):
    resp = requests.post(GOOGLE_URL, data={'q': text}, params={'key': KEY})
    return resp.json()['data']

def main():
    filename = sys.argv[1]
    for line in open(filename, encoding='utf8'):
        d = json.loads(line)
        detection = get_language(d['text'])
        print(json.dumps({
            'example': d['example'],
            'lang': detection['detections'][0][0]['language'],
        }))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
