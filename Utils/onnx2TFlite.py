import onnx
import onnx_tf as o2t
import tensorflow as tf
import tensorflow.python.keras.backend as K

"""set up in an python 3.6.smt env and use pip to install onnx-tf". Also install tensorflow"""

#path to onnx file (ends in .onnx)
onnx_path = "D:/Projects/reciept-scanner/Utils/converterFile/yolov8n.onnx"

#path to save folder
tflite_path = "D:/Projects/reciept-scanner/Utils/converterFile"

onnx_model = onnx.load(onnx_path)

#convert to and save to tflite model
tf_model = o2t.backend.prepare(onnx_model)
tf_model.export_graph(tflite_path + "/model.tf")

#tf to tflite
converter = tf.lite.TFLiteConverter.from_saved_model(tflite_path + "/model.tf")
tflite_model = converter.convert()

open(tflite_path + "/model.tflite", 'wb').write(tflite_model)

