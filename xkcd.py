#!/usr/bin/env python3

import sys
import json
import urllib.request
from pathlib import Path


def get(url, as_json=False):
    print('GET', url)
    with urllib.request.urlopen(url) as f:
        if as_json:
            return json.load(f)
        return f.read()


def get_xkcd(num, dest_dir):
    info = get(f'https://xkcd.com/{num}/info.0.json', as_json=True)
    title = info['safe_title']
    img_url = info['img']
    ext = Path(img_url).suffix
    dest = dest_dir / f'XKCD #{num}: {title}{ext}'
    print(f'Writing {num} to "{dest}"')
    dest.write_bytes(get(img_url))


if __name__ == '__main__':
    dest_dir = Path('img')
    for num in sys.argv[1:]:
        get_xkcd(num, dest_dir)
