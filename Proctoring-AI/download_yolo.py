import urllib.request
import os

def weights_download(out='models/yolov3.weights'):
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    headers = {'User-Agent': 'Mozilla/5.0'}

    req = urllib.request.Request(url, headers=headers)

    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

    with urllib.request.urlopen(req) as response, open(out, 'wb') as out_file:
        out_file.write(response.read())

    print(f"\nDownloaded weights to: {out}")

weights_download()
