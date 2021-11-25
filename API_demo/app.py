# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from flask import Flask, jsonify, request, redirect

import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
#
from image_captioning import *
import json

import urllib


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set({'jpg', 'jpeg','png'})

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


############################################################################

client_id = "##############" # 개발자센터에서 발급받은 Client ID 값
client_secret = "############" # 개발자센터에서 발급받은 Client Secret 값


#############################################################################


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    img=request.args.get('img') #/?img=


    checkpoint_path = "../checkpoints/train"
    print(checkpoint_path)

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)

    latest = tf.train.latest_checkpoint(checkpoint_path)
    print(latest)

    # Restore latest checkpoint
    ckpt.restore('../checkpoints/train/ckpt-9')

    image_extension=img[-4:]
    print(img)
    image_path=tf.keras.utils.get_file("image1"+image_extension,origin=img)
    print(image_path)

    # Evaluate the image
    result, attention_plot = evaluate(image_path)
    if result:
        os.remove(image_path)

        for i in result:
            if i == "<end>":
                result.remove(i)
            else:
                pass

        result_string = ' '.join(result)

        # result
        print(result_string)


        data = "source=en&target=ko&text=" + result_string
        url = "https://openapi.naver.com/v1/papago/n2mt"
        requests = urllib.request.Request(url)
        requests.add_header("X-Naver-Client-Id",client_id)
        requests.add_header("X-Naver-Client-Secret",client_secret)
        response = urllib.request.urlopen(requests, data=data.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            print(response_body.decode('utf-8'))
        else:
            print("Error Code:" + rescode)
        # Return caption
        return response_body.decode('utf-8')
        # return jsonify(caption=result_string);




@app.route('/caption', methods=['POST'])
def caption():
    checkpoint_path = "../checkpoints/train"
    print(checkpoint_path)

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)

    latest = tf.train.latest_checkpoint(checkpoint_path)
    print(latest)

    # Restore latest checkpoint
    ckpt.restore('../checkpoints/train/ckpt-13')

    if request.method == 'POST':
        print(request.files)
        print(request.get_data())
        params=json.loads(request.get_data())
        print(params)

        # if 'file' not in params["files"]:
        #     return "no file"
        file = params
        if file == '':
            return "no selected file"
        if file and allowed_file(file["filename"]):
            #filename = secure_filename(file["filename"])
            #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #file.save(file_path)

            file_path=file["filename"]
            image_extension=file_path[-4:]
            image_path=tf.keras.utils.get_file("image1"+image_extension,origin=file_path)
            print(image_path)

            # Evaluate the image
            result, attention_plot = evaluate(image_path)
            if result:
                os.remove(image_path)

                for i in result:
                    if i == "<end>":
                        result.remove(i)
                    else:
                        pass

                result_string = ' '.join(result)

                # result
                print(result_string)


                # Return caption
                return jsonify(caption=result_string)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8888)
