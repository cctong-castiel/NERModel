import numpy as np 
import pandas as pd 
import json
import os
import shutil
import config
import logging
from scripts.func import mark_sent, SentenceGetter, get_digest
from scripts.model import *
from handler.awshandler import *
from handler.ziphelper import *
from flask import Flask, request

# s3 config
aws_config = config.s3
accessKey = aws_config['aws_access_key_id']
secretKey = aws_config['aws_secret_access_key']
region = aws_config['region']
bucket = aws_config['bucket']

app = Flask(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(filename="train.log", filemode='a', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@app.route("/train", methods=['POST'])
def train():
    try:

        """Train NER model
        Input: json_link
        Output: model file and upload to S3"""
        # post request
        json_link = str(request.get_json(force=True)['json_link'])

        # variables
        logging.info("get variables")
        model_file_name = os.path.basename(json_link).split(".")[0]
        mdir = os.path.join(config.MODELS_DIR, model_file_name)
        logging.info(f"mdir is: {mdir}")
        hashword = None
        zip_type = ".tar.gz"

        # check if mdir exist
        if os.path.exists(mdir):
            shutil.rmtree(mdir)
            os.makedirs(mdir)
            logging.info(f"mdir path in if path: {mdir}")
        else: 
            os.makedirs(mdir)
            logging.info(f"mdir path in else path: {mdir}")
            logging.info(os.path.exists(mdir))

        # wget json link
        logging.info("get json link")
        logging.info(f"wget -P {mdir}/ {json_link}")
        os.system(f"wget -P {mdir}/ {json_link}")
        with open(os.path.join(mdir,model_file_name+'.json'), 'r') as json_f:
            array_text = json.load(json_f)

        # train model
        logging.info("train model")
        model_path = os.path.join(mdir, model_file_name)
        word2idx, tag2idx = mtrain(array_text, model_path)

        # remove s3_link json file
        os.remove(os.path.join(mdir,model_file_name+'.json'))
        
        # save word2idx and tag2idx
        logging.info("output word2idx.json and tag2idx.json")
        with open(os.path.join(mdir, model_file_name+'_word2idx.json'), 'w') as json_f:
            json.dump(word2idx, json_f)
        with open(os.path.join(mdir, model_file_name+'_tag2idx.json'), 'w') as json_f:
            json.dump(tag2idx, json_f)
        
        # zip word2idx.json, tag2idx.json and model files
        logging.info("zip file")
        zip_helper = Ziphelper(mdir, config.MODELS_DIR, model_file_name, zip_type, "")
        zip_helper.compressor()

        # hash
        logging.info("hashing")
        hashword = get_digest(os.path.join(config.MODELS_DIR, model_file_name + zip_type))
        logging.info(f"The directory before hash: {model_file_name}{zip_type}")
        logging.info(f"The hashword is {hashword}")

        # upload s3
        logging.info("upload to s3")
        local_path = os.path.join(config.MODELS_DIR, model_file_name + zip_type)
        m_folder = os.path.basename(os.path.normpath(config.MODELS_DIR))
        s3_path = os.path.join(m_folder, model_file_name)
        aws_handler = AWSHandler(accessKey, secretKey, region, bucket)
        aws_handler.upload_2S3(s3_path, local_path)

        return json.dumps({'hash':hashword})
    except Exception as e:
        logging.error(f"Error message is {e}.")
        return json.dumps({'error': f'Error message is {e}'})

@app.route("/run", methods=['POST'])
def run():
    try:

        """Run NER model
        Input: json_link
        Output: an array of target entities"""
        # post request
        json_link = str(request.get_json(force=True)['json_link'])
        model_file_hash = str(request.get_json(force=True)['model_file_hash'])
        s3_link = str(request.get_json(force=True)['s3_link'])

        # variables
        logging.info("get variables")
        model_file_name = os.path.basename(s3_link)
        zip_type = ".tar.gz"
        mdir = os.path.join(config.MODELS_DIR, model_file_name)
        logging.info(f"mdir is: {mdir}")

        # wget json link
        logging.info("get json link")
        logging.info(f"wget -P {mdir}/ {json_link}")
        os.system(f"wget -P {mdir}/ {json_link}")
        with open(os.path.join(mdir,model_file_name+'.json'), 'r') as json_f:
            array_text = json.load(json_f)

        # aws_handler
        local_path = os.path.join(config.MODELS_DIR, model_file_name + zip_type)
        m_folder = os.path.basename(os.path.normpath(config.MODELS_DIR))
        s3_path = os.path.join(m_folder, model_file_name)
        aws_handler = AWSHandler(accessKey, secretKey, region, bucket)

        # check if files exist
        if not os.path.exists(mdir):
            logging.info(f"mdir {mdir} not exists")
            os.makedirs(mdir)
            aws_handler.download_fromS3(s3_path, local_path)
        
        # check hash
        hashword = get_digest(os.path.join(config.MODELS_DIR, model_file_name + zip_type))
        if hashword != model_file_hash:
            logging.info(f"hashword {model_file_hash} not match")
            aws_handler.download_fromS3(s3_path, local_path)
        
        # unzip files
        logging.info("unzip files")
        zip_helper = Ziphelper(config.MODELS_DIR, config.MODELS_DIR, model_file_name, zip_type, "")
        zip_helper.decompressor()

        # run models
        logging.info("load word2idx.json and tag2idx.json")
        with open(os.path.join(mdir, model_file_name+'_word2idx.json'), 'r') as json_f:
            word2idx = json.load(json_f)
        with open(os.path.join(mdir, model_file_name+'_tag2idx.json'), 'r') as json_f:
            tag2idx = json.load(json_f)
        model_path = os.path.join(mdir, model_file_name)
        l_join_ent, s_join_ent = mrun(array_text, model_path, word2idx, tag2idx)

        # return dictionary count on entity tag
        logging.info('create dictionary count on entity tag')
        dict_tag_cnt = dict((x, l_join_ent.count(x)) for x in s_join_ent)

        return json.dumps(dict_tag_cnt)
    except Exception as e:
        logging.error(f"Error message is {e}.")
        return json.dumps({'error': f'Error message is {e}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=config.port, debug=True)


    





