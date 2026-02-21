import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import argparse
import os

from time import time

from custom_dataset import MSCDataset
from kernel import WaveformAugment,Functions
from training import Training
from optimization import Quatization

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keyword Spotter - CLI configuration"
    )

    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--frame-length", type=float, default=0.032, help="Frame length in seconds")
    parser.add_argument("--frame-step", type=float, default=0.032, help="Frame step in seconds")
    parser.add_argument("--n-mels", type=int, default=20)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--f-min", type=int, default=40)
    parser.add_argument("--f-max", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--model-width-mult", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--train-batch-size", type=int, default=64)

    return parser


def args_to_cfg(args: argparse.Namespace) -> dict:
    return {
        'sampling_rate': args.sampling_rate,
        'frame_length_in_s': args.frame_length,
        'frame_step_in_s': args.frame_step,
        'n_mels': args.n_mels,
        'n_mfcc': args.n_mfcc,
        'f_min': args.f_min,
        'f_max': args.f_max,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'model_width_mult': args.model_width_mult,
        'seed': args.seed,
        'train_steps': args.train_steps,
        'train_batch_size': args.train_batch_size,
    }

def loader(train_ds,val_ds,test_ds,CFG):
    train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=CFG['train_batch_size'],
    shuffle=True,
    generator=torch.Generator().manual_seed(CFG["seed"]),
    num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=100,
        shuffle=False,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=100,
        shuffle=False,
        num_workers=0
    )

    return (train_loader,val_loader,test_loader)

def store(transform_to_export,model,train_ds,test_acc,CFG):
    timestamp = int(time())
    saved_model_dir = './saved_models/'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    print(f'Model Timestamp: {timestamp}')

    torch.onnx.export(
        transform_to_export,  # model to export
        torch.randn(1, 1, 16000),  # inputs of the model,
        f'{saved_model_dir}/{timestamp}_frontend.onnx',  # filename of the ONNX model
        input_names=['input'], # input name in the ONNX model
        dynamo=True,
        optimize=True,
        report=False,
        external_data=False,
    )
    torch.onnx.export(
        model,  # model to export
        train_ds[0]['x'].unsqueeze(0),  # inputs of the model,
        f'{saved_model_dir}/{timestamp}_model.onnx',  # filename of the ONNX model
        input_names=['input'], # input name in the ONNX model
        dynamo=True,
        optimize=True,
        report=False,
        external_data=False,
    )

    output_dict = {
    'timestamp': timestamp,
    **CFG,
    'test_accuracy': test_acc
    }

    df = pd.DataFrame([output_dict])

    output_path='./saved_models/results.csv'
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

    return timestamp

def main():
    parser = build_parser()
    args = parser.parse_args()
    CFG = args_to_cfg(args)

    CLASSES = ['stop','up']
    train_dir = "Datasets/msc-train"
    val_dir =  "Datasets/msc-val"
    test_dir  = "Datasets/msc-test"

    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    random.seed(CFG['seed'])

    mfcc_transform = Functions.build_mfcc(CFG)

    train_transform = nn.Sequential(WaveformAugment(sr=16000, 
                                                    p=0.20,
                                                    max_shift_ms=30,
                                                    snr_db_min=35, 
                                                    snr_db_max=55,
                                                    gain_min=0.97, 
                                                    gain_max=1.03),
                                                    mfcc_transform
                                    )

    eval_transform = mfcc_transform

    train_ds = MSCDataset(train_dir, classes=CLASSES, transform=train_transform)
    val_ds   = MSCDataset(val_dir,   classes=CLASSES, transform=eval_transform)
    test_ds  = MSCDataset(test_dir,  classes=CLASSES, transform=eval_transform)

    train_loader,val_loader,test_loader = loader(train_ds,val_ds,test_ds,CFG)
    trainer = Training(train_loader,val_loader)

    model,best_state = trainer.train(CFG)
    model.load_state_dict(best_state)

    test_acc = Functions.eval_acc(model,test_loader)
    print(f'Test Accuracy based on val_acc: {test_acc:.2f}%')

    pass_corr = Functions.pass_rate(model, test_loader, thr=0.999)
    print("pass@0.999 (correct):", pass_corr)

    model.load_state_dict(best_state)
    model.eval()
    transform_to_export = mfcc_transform
    transform_to_export.eval()
    timestamp = store(transform_to_export,model,train_ds,test_acc,CFG)
    print(timestamp)

    quantizer = Quatization()
    quantizer.quantization(CLASSES, test_dir, val_dir, timestamp=timestamp)

if __name__ == "__main__":
    main()