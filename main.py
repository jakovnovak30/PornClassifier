from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def klasificiraj(slika):
  interpreter = tf.lite.Interpreter(model_path='./ejaj.tflite', num_threads=None)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(slika).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - 0) / 255

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels('kategorije.txt')

  output = []

  for i in top_k:
    if floating_model:
      output.append('{:08.6f} % {}'.format(float(results[i]*100), labels[i]))
    else:
      output.append('{:08.6f} % {}'.format(float(results[i] / 255.0 * 100), labels[i]))

  return output

@app.route('/')
def render():
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      return render_template('rezultati.html', rezultati=klasificiraj(f))

if __name__ == '__main__':
    app.run(debug=True)
