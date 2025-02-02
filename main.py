from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels mapping
class_labels = [
    "Mild Dementia",          # Class 0
    "Moderate Dementia",      # Class 1
    "Non Dementia",           # Class 2
    "Very Mild Dementia"      # Class 3
]

# Preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the input image for the model."""
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0                 # Normalize pixel values
    return img_array.astype(np.float32)

# Function to run inference
def predict_tflite(image):
    """Runs inference using the TFLite model."""
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route('/')
def index():
    return "Welcome to the Dementia Detection API!"

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Convert file to a format compatible with `load_img`
        file_stream = io.BytesIO(file.read())
        img = load_img(file_stream, target_size=(224, 224))

        # Preprocess the image
        processed_image = preprocess_image(img)

        # Run inference
        predictions = predict_tflite(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Prepare the result
        result = {
            "Predicted Class": class_labels[predicted_class_index],
            "Confidence": f"{confidence*100:.2f}%"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
