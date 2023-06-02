import torch
import onnx
import tensorflow as tf
import onnx_tf
# Load the ONNX model
onnx_model = onnx.load('./output/model.onnx')

# Convert the ONNX model to TensorFlow format
tf_model_path = 'model.pb'
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

# Convert the TensorFlow model to TensorFlow Lite format
#converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_path)
#tflite_model = converter.convert()

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open("new_converted_model_fp16.tflite", "wb").write(tflite_model)

# Save the TensorFlow Lite model to a file
#with open('model.tflite', 'wb') as f:
#    f.write(tflite_model)