# NERModel
- need to copy config.template.py and create config.py

### Train
- call /train
```
{
    "json_link": "https://abc.s3.amazonaws.com/folder/file.json"
}
```
- return:
  - hash: hash string of the zip model directory
  
### Run
- call /run
```
{
    "json_link": "https://abc.s3.amazonaws.com/folder/file.json",
    "model_file_hash": "hash",
    "s3_link": "https://bucket/model_folder/model_name"
}
```
- return: a dictionary of entity count
# NERModel
