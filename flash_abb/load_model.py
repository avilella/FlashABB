import os, subprocess, json, argparse,requests
from yaml import load, Loader
import torch

list_of_models = {
    "flash-abb":["https://zenodo.org/api/records/15920210/draft/files/rope_model.ckpt/content", "model.pt"],
}
flash_abb_models = ["flash-abb"]


def load_model(model_to_use="flash-abb", random_init=False, device='cpu'):

    if model_to_use in flash_abb_models:
        flabb, hparams = fetch_flash_abb(
            model_to_use, 
            random_init=random_init, 
            device=device
        )
    else: 
        assert False, f"The selected model to use ({model_to_use}) does not exist.\
        Please select a valid model."   

    return flabb, hparams


def download_model(model_to_use="flash-abb"):
    """
    If not already downloaded, download model inside environment.
    """

    local_model_folder = os.path.join(os.path.dirname(__file__), "model-weights-{}".format(model_to_use))
    os.makedirs(local_model_folder, exist_ok=True)

    file_w_weights, file_model = list_of_models[model_to_use] # modify list of models

    if not os.path.isfile(os.path.join(local_model_folder, file_model)):
        print("Downloading model ...")
        # tmp_file = os.path.join(local_model_folder, "tmp.tar.gz")

        # with open(tmp_file,'wb') as f: f.write(requests.get(file_w_weights).content)

        # subprocess.run(["tar", "-zxvf", tmp_file, "-C", local_model_folder], check = True) 
        # os.remove(tmp_file)
        model_path = os.path.join(local_model_folder, file_model)
        with open(model_path,'wb') as f: f.write(requests.get(file_w_weights).content)


    return local_model_folder


def fetch_flash_abb(model_to_use, random_init=False, device='cpu'):

    from .model.flash_abb import FlashABB

    local_model_folder = download_model(model_to_use)

    with open(os.path.join(local_model_folder, 'params.yaml'), 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**load(f, Loader=Loader)).model

    flabb = FlashABB(hparams)
    if not random_init:
        ckpt = torch.load(
            os.path.join(local_model_folder, 'model.pt'),
            map_location=torch.device(device),
            weights_only=False,
        )
        flabb.load_state_dict(
            ckpt['state_dict']
        )

    return flabb, hparams
