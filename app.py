import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms
import matplotlib.pyplot as plt
import io
import torch.nn as nn
from ultralytics import YOLO

# Initialize YOLO and CRNN models
text_det_model_path = './yolov8_best.pt'
crnn_resnet_model_path = './crnn_resnet_best.pt'

# YOLO model for text detection
yolo = YOLO(text_det_model_path)

# CRNN model for text recognition
chars = '0123456789abcdefghijklmnopqrstuvwxyz-'
vocab_size = len(chars)
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

class CRNN_Resnet(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3):
        super(CRNN_Resnet, self).__init__()
        backbone = timm.create_model('resnet101', in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.require_grad = True
        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            1024, hidden_size, n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)
        return x

hidden_size = 256
n_layers = 2
dropout_prob = 0.3
unfreeze_layers = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

crnn_resnet_model = CRNN_Resnet(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    n_layers=n_layers,
    dropout=dropout_prob,
    unfreeze_layers=unfreeze_layers
).to(device)
crnn_resnet_model.load_state_dict(torch.load(crnn_resnet_model_path, map_location=device))

# Image transformations
data_transforms = transforms.Compose([
    transforms.Resize((100, 420)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def decode(encoded_sequences, idx_to_char, blank_char='-'):
    decoded_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None
        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char and char != prev_char:
                    decoded_label.append(char)
                prev_char = char
        decoded_sequences.append(''.join(decoded_label))
    return decoded_sequences

def text_detection(img_path, text_det_model):
    text_det_results = text_det_model(img_path, verbose=False)[0]
    bboxes = text_det_results.boxes.xyxy.tolist()
    classes = text_det_results.boxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()
    return bboxes, classes, names, confs

def text_recognition(img, data_transforms, text_reg_model, idx_to_char, device):
    transformed_image = data_transforms(img).unsqueeze(0).to(device)
    text_reg_model.eval()
    with torch.no_grad():
        logits = text_reg_model(transformed_image).detach().cpu()
    text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)
    return text

def visualize_detections(img, detections):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    for bbox, detected_class, confidence, transcribed_text in detections:
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1 - 10, f"{detected_class} ({confidence:.2f}) {transcribed_text}",
                 fontsize=9, bbox=dict(facecolor='red', alpha=0.5))
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return buf

def predict(img_path, data_transforms, text_det_model, text_reg_model, idx_to_char, device):
    bboxes, classes, names, confs = text_detection(img_path, text_det_model)
    img = Image.open(img_path)
    predictions = []
    for bbox, cls, conf in zip(bboxes, classes, confs):
        x1, y1, x2, y2 = bbox
        cropped_image = img.crop((x1, y1, x2, y2))
        transcribed_text = text_recognition(cropped_image, data_transforms, text_reg_model, idx_to_char, device)
        predictions.append((bbox, names[int(cls)], conf, transcribed_text))
    return img, predictions

# Streamlit App

# Load and display the uploaded image for illustration purposes
illustration_image_path = 'assets/image1.png'  # Path to the image in the 'assets' folder
illustration_image = Image.open(illustration_image_path)

st.title("Scene Text Recognition (STR)")

# Display the illustration image
st.image(illustration_image, use_column_width=True)

st.write("Upload your image of an orange or choose one of the example images below:")

# Example images (stored in the 'assets' folder)
example_images = {
    "Example 1": "assets/example1.jpg",
    "Example 2": "assets/example2.jpg",
    "Example 3": "assets/example3.jpg",
    "Example 4": "assets/example4.jpg",
    "Example 5": "assets/example5.jpg",
    "Example 6": "assets/example6.jpg"
}

# Dropdown menu for selecting example images
example_choice = st.selectbox("Choose an example image:", ["None"] + list(example_images.keys()))

# File uploader for user's image
uploaded_file = st.file_uploader("Or upload your own image...", type=["jpg", "png", "jpeg"])

# Determine the image to use: either an uploaded file or an example
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_path = './uploaded_image.jpg'
    img.save(img_path)
elif example_choice != "None":
    img_path = example_images[example_choice]
    img = Image.open(img_path)
else:
    img = None

# Proceed with text detection and recognition if an image was selected or uploaded
if img is not None:
    st.image(img, caption="Selected Image", use_column_width=True)
    st.write("Processing...")

    img, detections = predict(img_path, data_transforms, yolo, crnn_resnet_model, idx_to_char, device)

    if detections:
        st.write("### Detected Text Regions and Recognized Text")
        for bbox, detected_class, confidence, transcribed_text in detections:
            st.write(f"Detected: {detected_class}, Confidence: {confidence:.2f}, Text: {transcribed_text}")
        
        buf = visualize_detections(img, detections)
        st.image(buf, caption='Detected Text Regions', use_column_width=True)
    else:
        st.write("No text detected.")
