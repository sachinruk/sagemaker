import json
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def model_fn(model_dir: str) -> nn.Module:
    model = models.Model([784, 100, 100, 10])
    model.load_state_dict(torch.load(pathlib.Path(model_dir) / "model.ckpt"))
    
    return model.to(device).eval()

def input_fn(request_body, request_content_type) -> torch.FloatTensor:
    assert request_content_type=='application/json'
    data = json.loads(request_body)['inputs']
    data = torch.FloatTensor(data) / 255.0 # bs x num_channel x width x height
    return data.reshape(len(data), -1).to(device)

def predict_fn(input_object, model):
    with torch.inference_mode():
        probabilities = F.softmax(model(input_object), dim=-1)
    return probabilities

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    prob, idx = predictions.topk(1)
    return json.dumps(
        {
            "probability": prob.cpu().tolist(),
            "prediction": idx.cpu().tolist()
        }
    )
