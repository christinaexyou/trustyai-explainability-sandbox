# common
import requests
import json
import torch

# third party
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

# load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")

# tokenize sample text
text = """
        i went to see this movie with a bunch of friends one night.
        I didn't really hear much about it. So I wasn't expecting anything.
        But after I saw it, I really liked it. Nicolas Cage and the rest of
        the cast were very good. But I do have to say Giovanni Ribisi's acting
        performace did need a little perking up. But such a small flaw, it
        could be overrided. <br /><br />Gone In 60 Seconds is about a retired
        car thief who must boost 60 rare and exotic cars in one night to save
        his brother's life. The movie is in no way predictable. So the ending
        should be a suprise. Think it's just another, fast car driving movie?
        Well you are partially right. There is much more to it. Everyone should
        take a look at this movie.
       """

model_inputs = tokenizer(text, return_tensors="pt")

# specify the model name and version
deployed_model_name = "distilbert1"
rest_url = "http://modelmesh-serving:8008"

# set the inference URL based on the Triton server's address
url =  f"{rest_url}/v2/models/{deployed_model_name}/infer"

# payload with input params
payload = {
    "inputs": [
        {
            "name": "INPUT__0",  # what you named input in config.pbtxt
            "datatype": "INT64",
            "shape": [1, 181],
            "data": model_inputs['input_ids'].tolist(),
        },

        {
            "name": "INPUT__1",  # what you named input in config.pbtxt
            "datatype": "INT64",
            "shape": [1, 181],
            "data": model_inputs['input_ids'].tolist(),
        },

        {
            "name": "INPUT__2",  # what you named input in config.pbtxt
            "datatype": "FP32",
            "shape": [1, 181, 768],
            "data": torch.zeros_like(model.base_model.embeddings.word_embeddings(model_inputs['input_ids'])).tolist(),
        },

    ]
}

# sample invoke
response = requests.post(url, data=json.dumps(payload))
if response.status_code == 200:
    result = json.loads(response.content)
    print(result)
else:
    print(response.status_code, response.content)