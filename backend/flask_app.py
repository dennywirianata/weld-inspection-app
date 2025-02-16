from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['TRAIN_DATA_FOLDER'] = 'train_data/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAIN_DATA_FOLDER'], exist_ok=True)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the best model
#model_path = os.path.join('models', 'retinanet_best_model_r0.keras')
#best_model = tf.keras.models.load_model(model_path)
class RetinaNet(nn.Module):
    def __init__(self, num_classes=2):
        super(RetinaNet, self).__init__()
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get intermediate features
        self.layer3 = nn.Sequential(*list(self.backbone.children())[:6])  # conv3_block4_out
        self.layer4 = nn.Sequential(*list(self.backbone.children())[6:7])  # conv4_block6_out
        self.layer5 = nn.Sequential(*list(self.backbone.children())[7:8])  # conv5_block3_out

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),  # ResNet50's last conv layer has 2048 channels
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, loss, and optimizer
model = RetinaNet(num_classes=2)
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid and BCELoss
optimizer = optim.Adam(model.parameters())

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = model.to(device)
model_path = os.path.join('models', '12_feb_restinanet_best_model_r0.pth')
best_model.load_state_dict(torch.load(model_path))
best_model.eval()

def predict_image(file_path):
    
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    

    # Predict using the best model
    #prediction = best_model.predict(img_batch)

   # Load and preprocess the image
    #img = Image.open(file_path).convert('RGB')
    #img_tensor = image_transform(img).unsqueeze(0).to(device)

    # Predict using the best model
    with torch.no_grad():
        # If new_image_batch is a TensorFlow tensor, convert it to a PyTorch tensor first:
        new_image_batch_torch = torch.from_numpy(img_batch.numpy())
        new_image_batch_torch = new_image_batch_torch.permute(0, 3, 1, 2)
        prediction = best_model(new_image_batch_torch.to(device).float())
        #prediction = best_model(img_tensor)
        #confidence = prediction[0][0].item()
        print(prediction)
    
    sum_value=abs(torch.sum(prediction[0]))
    # print("Sum Value:", sum_value)
    p_true=abs(prediction[0][0])
    p_false=abs(prediction[0][1])
    print("p_true:",p_true)
    print("p_false:",p_false)

    if p_true > 55:
        result = "Accepted"
        confidence = p_true
    else:
        result = "Rejected"
        confidence = p_false
    # Interpret the prediction
    #if confidence > -0.5:
     #   result = "Accepted"
    #else:
     #   result = "Rejected"
    
    # Generate a detailed analysis report (optional)
    #confidence = float(prediction[0][0])
    details = f"{confidence:.2f}"

    return {"result": result, "confidence": details}


# Endpoint to handle image upload and inspection
@app.route('/upload/image', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    # Save the file (for demonstration purposes)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Placeholder for processing logic
    # Call AI model for prediction
    result = predict_image(file_path)
    status = result.get('result')
    details = result.get('confidence')
    result = {"status": status, "details": details}
    
    return jsonify(result), 200

# Endpoint to handle video upload and inspection
@app.route('/upload/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    

    # Process video frames
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 5)
    acceptable_frames = 0
    rejectable_frames = 0
    frame_predictions = []

    frame_index = 0
    frame_count = 0
    frame_skip_interval = 5

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video writer
    output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], "Detection_" + filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if (frame_count+1) % frame_skip_interval != 0:
            continue

        # Convert frame to image for prediction
        #img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #img_tensor = image_transform(img).unsqueeze(0).to(device)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Convert to PyTorch tensor and move to device
        input_frame = torch.from_numpy(input_frame).to(device).float()

        # Permute dimensions to [batch_size, channels, height, width]
        input_frame = input_frame.permute(0, 3, 1, 2)

        # Predict using the best model
        with torch.no_grad():
            #prediction = best_model(img_tensor.float())
            prediction = model(input_frame)
            #probabilities = torch.softmax(prediction, dim=1).cpu().numpy()[0]
            #confidence = probabilities[1]  # Assuming the second element is the probability of "Acceptable"

        #sum_value=abs(torch.sum(prediction))
        # print("Sum Value:", sum_value)
        #p_true=abs(prediction[0][0])/sum_value
        #p_false=abs(prediction[0][1])/sum_value
        sum_value=torch.sum(abs(prediction[0]))
        # print("Sum Value:", sum_value)
        p_true=abs(prediction[0][0])
        p_false=abs(prediction[0][1])
        print("p_true:",p_true)
        print("p_false:",p_false)

        if p_true < 0.85:#if p_true > p_false:
            result = "Accepted"
            acceptable_frames += 1
            confidence = float(p_true)
        else:
            result = "Rejected"
            rejectable_frames += 1
            confidence = float(p_false)
        # Interpret the prediction
        #if confidence > 0.3:
        #    result = "Acceptable"
        #    acceptable_frames += 1
        #else:
         #   result = "Rejectable"
         #   rejectable_frames += 1

        frame_predictions.append({
            "index": frame_index,
            "status": result,
            "confidence": float(confidence)  # Convert confidence to float
        })
        frame_index += 1

        # Overlay results
        text_color = (0, 0, 255)  # Red color (BGR format)
        if result == "Rejected":
            text_color = (0, 0, 255) # Red color (BGR format)

        else:
            text_color = (0, 255, 0) # Green color (BGR format)

        cv2.putText(frame, f"Class: {result}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write to output video
        out.write(frame)

         # Save frame for preview
        frame_filename = f'frame_{frame_index}.jpg'
        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
        cv2.imwrite(frame_path, frame)

    cap.release()
    out.release()

    # Calculate percentages
    acceptable_percentage = (acceptable_frames / total_frames) * 100
    rejectable_percentage = (rejectable_frames / total_frames) * 100

    return jsonify({
        "total_frames": total_frames,
        "acceptable_frames": acceptable_frames,
        "rejectable_frames": rejectable_frames,
        "acceptable_percentage": acceptable_percentage,
        "rejectable_percentage": rejectable_percentage,
        "frame_predictions": frame_predictions,
        "video_url": f"/uploads/{filename}"
    }), 200


# Endpoint to handle additional training data upload
@app.route('/train', methods=['POST'])
def train_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    # Save the file (for demonstration purposes)
    file_path = os.path.join('training_data', file.filename)
    file.save(file_path)
    return jsonify({"message": "Training data uploaded successfully"}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create uploads and training_data directories if they don't exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)
    app.run(debug=True)