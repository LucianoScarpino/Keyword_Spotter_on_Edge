import os
import torch
import onnx
import onnxruntime as ort
import numpy as np

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    StaticQuantConfig,
    quantize,
)

from kernel import Functions
from custom_dataset import MSCDataset

class DataReader(CalibrationDataReader):
    def __init__(self, dataset,ort_frontend):
        self.dataset = dataset
        self.enum_data = None
        self.ort_frontend = ort_frontend

        self.datasize = len(self.dataset)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.dataset)

        x = next(self.enum_data, None)

        if x is None:
            return None

        x = x['x']
        x = x.numpy().astype(np.float32)
        x = np.expand_dims(x, 0)
        x = self.ort_frontend.run(None, {'input': x})[0]
        x = {'input': x}

        return x

    def rewind(self):
        self.enum_data = None

class Quatization(object):
    def __init__(self):
        pass
    
    def quantization(self,CLASSES,test_dir,val_dir,timestamp):
        test_ds = MSCDataset(test_dir, CLASSES, torch.nn.Identity())
        MODEL_NAME = timestamp

        frontend_float32_file = f'./saved_models/{MODEL_NAME}_frontend.onnx'
        model_float32_file = f'./saved_models/{MODEL_NAME}_model.onnx'
        ort_frontend = ort.InferenceSession(frontend_float32_file)
        ort_model = ort.InferenceSession(model_float32_file)

        true_count = 0.0
        for sample in test_ds:
            inputs = sample['x']
            label = sample['label']
            inputs = inputs.numpy()
            inputs = np.expand_dims(inputs, 0)
            features = ort_frontend.run(None, {'input': inputs})[0]
            outputs = ort_model.run(None,  {'input': features})[0]
            prediction = np.argmax(outputs, axis=-1).item()
            true_count += prediction == label

        float32_accuracy = true_count / len(test_ds) * 100
        frontend_size = os.path.getsize(frontend_float32_file)
        model_float32_size = os.path.getsize(model_float32_file)
        total_float32_size = frontend_size + model_float32_size

        print(f'Float32 Accuracy: {float32_accuracy:.2f}%')
        print(f'Float32 Frontend Size: {frontend_size / 2**10:.1f}KB')
        print(f'Float32 Model Size: {model_float32_size / 2**10:.1f}KB')
        print(f'Float32 Total Size: {total_float32_size / 2**10:.1f}KB')

        calibration_ds = MSCDataset(val_dir, transform=torch.nn.Identity(), classes=CLASSES)
        data_reader = DataReader(calibration_ds,ort_frontend)

        conf = StaticQuantConfig(
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,                       #QDQ = quantize,dequantize,quantize
        #calibrate_method=CalibrationMethod.MinMax ,
        calibrate_method=CalibrationMethod.Entropy ,        #Less susciptible to outliers
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        )

        model_int8_file = f'./saved_models/{timestamp}_model_INT8.onnx'
        quantize(model_float32_file, model_int8_file, conf)

        ort_model_int8 = ort.InferenceSession(model_int8_file)

        true_quant_count = 0.0
        for sample in test_ds:
            inputs = sample['x']
            label = sample['label']
            inputs = inputs.numpy()
            inputs = np.expand_dims(inputs, 0)
            features = ort_frontend.run(None, {'input': inputs})[0]
            outputs = ort_model_int8.run(None,  {'input': features})[0]
            prediction = np.argmax(outputs, axis=-1).item()
            true_quant_count += prediction == label

        int8_accuracy = true_quant_count / len(test_ds) * 100
        frontend_size = os.path.getsize(frontend_float32_file)
        model_int8_size = os.path.getsize(model_int8_file)
        total_int8_size = frontend_size + model_int8_size

        print(f'INT8 Accuracy: {int8_accuracy:.2f}%')
        print(f'Float32 Frontend Size: {frontend_size / 2**10:.1f}KB')
        print(f'INT8 Model Size: {model_int8_size / 2**10:.1f}KB')
        print(f'INT8 Total Size: {total_int8_size / 2**10:.1f}KB')

        pass_f32_onnx  = Functions.pass_rate_onnx(test_ds, ort_frontend, ort_model, thr=0.999)
        pass_int8_onnx = Functions.pass_rate_onnx(test_ds, ort_frontend, ort_model_int8, thr=0.999)

        print("pass@0.999 ONNX float32:", pass_f32_onnx)
        print("pass@0.999 ONNX int8:", pass_int8_onnx)