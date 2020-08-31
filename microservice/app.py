#!flask/bin/python
from flask import Flask, jsonify, request

import shutil
import requests
import os


def download_img (image_url, save_dir, file_name):
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(file_name,'wb') as f:
            #shutil.copyfileobj(r.raw, f)
            f.write(r.content)

        print('Image sucessfully Downloaded: ',file_name)
    else:
        print('Image Couldn\'t be retreived')

app = Flask(__name__)

@app.route('/', methods=['POST'])
def create_task():
    if request.method == 'POST':
        file = request.json
        save_dir = "save_dir"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        for f in file:
            file_name = os.path.join(save_dir, f["filename"])
            image_url = f["url"]
            #file_name = file_name.split(".")[0] + ".zip"
            download_img (image_url, save_dir, file_name)
        print ("download image finished")
        print ("zipping dir ...")
        shutil.make_archive(save_dir,'zip',save_dir)
        shutil.rmtree(save_dir)
        print("zipping dir finished")
        return jsonify({"message":"download successful"})

if __name__ == '__main__':
    app.run(debug=True)
