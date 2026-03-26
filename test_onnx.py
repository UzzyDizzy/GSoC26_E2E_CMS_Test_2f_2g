import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load("custom_model.onnx")
onnx.checker.check_model(onnx_model)

sess = ort.InferenceSession("custom_model.onnx")

x = np.random.randn(1, 4, 125, 125).astype(np.float32)
output = sess.run(None, {"input": x})

print(output)