import argparse
import time
import sys
import collections
import numpy as np
import redis
import onnxruntime as ort
import torch
import torchaudio.transforms as T
import sounddevice as sd
import board
import adafruit_dht

conf = {
    'SAMPLE_RATE': 48000,   
    'CHANNELS': 1,          
    'DOWNSAMPLE_RATE': 16000, 
    'WINDOW_DURATION': 1.0,
    'INFERENCE_INTERVAL': 1.0, 
    'DHT_INTERVAL': 5.0,
    'CONFIDENCE_THRESHOLD': 0.999
}

BUFFER_SIZE = int(conf['SAMPLE_RATE'] * conf['WINDOW_DURATION'])

class SmartHygrometer:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.data_collection_enabled = False # Initial state disabled
        self.audio_buffer = collections.deque(maxlen=BUFFER_SIZE)
        
        # Initialize Redis 
        try:
            self.r = redis.Redis(
                host=args.host,
                port=args.port,
                username=args.user,
                password=args.password,
                decode_responses=True # Adjust based on Lab2 requirements
            )
            self.r.ping()
            print("Connected to Redis Cloud.")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            sys.exit(1)

        # Initialize DHT11 
        self.dht_device = adafruit_dht.DHT11(board.D4)
        print("DHT-11 Sensor initialized.")

        # Load ONNX Models
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.frontend_sess = ort.InferenceSession("Group12_frontend.onnx", sess_options=so, providers=["CPUExecutionProvider"])
            self.model_sess = ort.InferenceSession("Group12_model.onnx", sess_options=so, providers=["CPUExecutionProvider"])
            print("ONNX models loaded.")
        except Exception as e:
            print(f"Error loading ONNX models: {e}")
            sys.exit(1)

        # Resampler for pipeline
        self.resampler = T.Resample(conf['SAMPLE_RATE'], conf['DOWNSAMPLE_RATE'])

    def audio_callback(self, indata, frames, time_info, status):
        """Callback to continuously record audio"""
        if status:
            print(f"Audio status: {status}")
        self.audio_buffer.extend(indata[:, 0].tolist())

    def process_audio(self):
        """KWS Pipeline: int16@48k -> float32 -> resample@16k -> pad/crop 16000 -> ONNX frontend -> ONNX model"""
        if len(self.audio_buffer) < BUFFER_SIZE:
            return None

        audio_i16 = np.asarray(self.audio_buffer, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        x = torch.from_numpy(audio_f32).unsqueeze(0)  # (1, 48000)
        x16 = self.resampler(x)

        target_len = 16000
        cur_len = x16.shape[-1]
        if cur_len < target_len:
            x16 = torch.nn.functional.pad(x16, (0, target_len - cur_len))
        elif cur_len > target_len:
            x16 = x16[..., :target_len]

        input_np = x16.unsqueeze(1).contiguous().cpu().numpy().astype(np.float32)  # (1,1,16000)

        frontend_input_name = self.frontend_sess.get_inputs()[0].name
        feats = self.frontend_sess.run(None, {frontend_input_name: input_np})[0]

        model_input_name = self.model_sess.get_inputs()[0].name
        logits = self.model_sess.run(None, {model_input_name: feats})[0]  # (1,2)
        probs = self.softmax(logits)[0]  # (2,)

        return probs

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def handle_command(self, probs):
        """Control Logic """
        # Mapping: 0 -> "stop", 1 -> "up"         
        stop_prob = probs[0]
        up_prob = probs[1]
        
        predicted_idx = np.argmax(probs)
        top_prob = probs[predicted_idx]

        # Logic
        if predicted_idx == 1 and up_prob > conf['CONFIDENCE_THRESHOLD']:
            if not self.data_collection_enabled:
                print(f"Command 'UP' detected ({up_prob:.4f}). Enabling data collection.")
                self.data_collection_enabled = True
        elif predicted_idx == 0 and stop_prob > conf['CONFIDENCE_THRESHOLD']:
            if self.data_collection_enabled:
                print(f"Command 'STOP' detected ({stop_prob:.4f}). Stopping data collection.")
                self.data_collection_enabled = False
        else:
            # Remain in current state if prob <= 99.9%
            pass

    def read_and_upload_sensor(self):
        """Data collection and upload """
        try:
            # Measure temp and humidity
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            
            if temperature is not None and humidity is not None:
                timestamp = int(time.time())
                print(f"Sending data -> Temp: {temperature}C, Hum: {humidity}%")
                
                self.r.ts().add("temperature", timestamp, temperature)
                self.r.ts().add("humidity", timestamp, humidity)
                
            else:
                print("Failed to retrieve data from humidity sensor")
        except RuntimeError as error:
            # retry next time
            print(error.args[0])
        except Exception as error:
            self.dht_device.exit()
            raise error

    def start(self):
        print("Starting Smart Hygrometer System...")
        
        # Configure audio stream
        # Using int16 as requested, 1 channel, 48kHz
        with sd.InputStream(callback=self.audio_callback,
                            channels=conf['CHANNELS'],
                            samplerate=conf['SAMPLE_RATE'],
                            dtype='int16', 
                            blocksize=4800): # 100ms blocks
            
            last_inference_time = time.time()
            last_sensor_time = time.time()

            while self.running:
                current_time = time.time()

                # Inference every 1 second
                if current_time - last_inference_time >= conf['INFERENCE_INTERVAL']:
                    probs = self.process_audio()
                    if probs is not None:
                        self.handle_command(probs)
                    last_inference_time = current_time

                # Sensor reading every 5 seconds if enabled
                if self.data_collection_enabled:
                    if current_time - last_sensor_time >= conf['DHT_INTERVAL']:
                        self.read_and_upload_sensor()
                        last_sensor_time = current_time
                
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smart Hygrometer V2.0')
    parser.add_argument('--host', type=str, required=True, help='Redis Cloud host')
    parser.add_argument('--port', type=int, required=True, help='Redis Cloud port')
    parser.add_argument('--user', type=str, required=True, help='Redis Cloud username')
    parser.add_argument('--password', type=str, required=True, help='Redis Cloud password')
    
    args = parser.parse_args()

    app = SmartHygrometer(args)
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        # Cleanup
        if 'app' in locals():
            app.dht_device.exit()