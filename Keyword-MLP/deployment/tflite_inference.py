
from argparse import ArgumentParser
from config_parser import get_config
from utils.misc import get_model
from utils.dataset import GoogleSpeechDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import glob
import json
import numpy as np
import tensorflow as tf
import time

def get_preds(interpreter, dataloader, input_details, output_details) -> list:
    """Performs inference."""
    start_time = time.time()
    preds_list = []
    infer_list = []
    for data in tqdm(dataloader):
        interpreter.set_tensor(input_details[0]['index'], data)
        for i in range(11):
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            infer_list.append(inference_time)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        preds = np.argmax(output_data, axis=1).tolist()
        preds_list.extend(preds)
    end_time = time.time()
    print(f'Average time:{sum(infer_list)/len(infer_list)*1000}')
    print(f"Inference time: {infer_list} seconds")
    for element in infer_list:
        print(f'{element:.15f}')
    return preds_list

def main(args):
    config = get_config(args.conf)
    model = get_model(config["hparams"]["model"])

    ######################
    # setup data
    ######################
    if os.path.isdir(args.inp):
        data_list = glob.glob(os.path.join(args.inp, "*.wav"))
    elif os.path.isfile(args.inp):
        data_list = [args.inp]

    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=None,
        audio_settings=config["hparams"]["audio"],
        aug_settings=None,
        cache=0
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ######################
    # convert to TFLite
    ######################
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # tflite_path = os.path.join(args.out, "model.tflite")
    # with open(tflite_path, 'wb') as f:
    #     f.write(tflite_model)
    # print(f"Exported TFLite model to {tflite_path}")

    tflite_path = 'converted_model_fp16.tflite'
    ######################
    # run inference
    ######################
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = get_preds(interpreter, dataloader, input_details, output_details)

    ######################
    # save predictions
    ######################
    if args.lmap:
        with open(args.lmap, "r") as f:
            label_map = json.load(f)
        preds = list(map(lambda a: label_map[str(a)], preds))
    
    pred_dict = dict(zip(data_list, preds))
    
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "preds.json")

    with open(out_path, "w+") as f:
        json.dump(pred_dict, f)

    print(f"Saved preds to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True, help="Path to config file. Will be used only to construct model and process audio.")
    parser.add_argument("--inp", type=str, required=True, help="Path to input. Can be a path to a .wav file, or a path to a folder containing .wav files.")
    parser.add_argument("--out", type=str, default="./", help="Path to output folder. Predictions will be stored in {out}/preds.json.")
    parser.add_argument("--lmap", type=str, default=None, help="Path to label_map.json. If not provided, will save predictions as class indices instead of class names.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for batch inference.")
    
    args = parser.parse_args()

    assert os.path.exists(args.inp), f"Could not find input {args.inp}"

    main(args)