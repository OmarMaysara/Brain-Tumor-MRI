import sys
import os
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont

import torch
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np  
import torchvision.transforms as transforms
from PIL import Image

# =====================================================================
# MODEL INITIALIZATION 
# =====================================================================

CLASSES = ["Glioma", "Healthy", "Meningioma", "Pituitary"] 
MODEL_PATH = "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

try:
    # Load the checkpoint file safely on the current target device
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # CASE 1: The file is a dictionary (either raw state_dict or a full checkpoint)
    if isinstance(checkpoint, dict):
        print("💡 Info: 'best_model.pth' recognized as a dictionary dictionary structure.")
        print("🔧 Instantiating standard DenseNet121 architecture to map weights...")
        
        # Reconstruct structural architecture matching your training configuration
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASSES))
        
        # Extract the nested weights if saved as a comprehensive training checkpoint
        if "model_state_dict" in checkpoint:
            print("📦 Full training checkpoint detected. Extracting 'model_state_dict'...")
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Clean potential data-parallel prefixes ('module.') if saved from multi-GPU setups
        first_key = list(state_dict.keys())[0]
        if first_key.startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        print("✅ Success: Weights successfully mapped to DenseNet121 configuration.")
        
    # CASE 2: The file is a fully serialized model structure + weights
    else:
        model = checkpoint
        print("✅ Success: Fully serialized model loaded successfully.")

    model.to(device)
    model.eval()  # Set network layers to absolute evaluation mode

except Exception as e:
    print("❌ CRITICAL ERROR LOADING MODEL:")
    traceback.print_exc()

# =====================================================================
# MEDICAL PREPROCESSING PIPELINE (Matches your Data Lake script)
# =====================================================================
def crop_brain_contour(image):
    """Finds the brain and crops out the black dead space."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]
    return image

def letterbox_resize(image, target_size=(224, 224)):
    """Resizes image by padding to maintain aspect ratio (no stretching)."""
    h, w = image.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create black canvas and center the resized image
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    offset_h = (target_size[0] - new_h) // 2
    offset_w = (target_size[1] - new_w) // 2
    canvas[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = resized
    return canvas

def apply_medical_enhancement(image):
    """Applies CLAHE in the LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)

# =====================================================================
# THE INFERENCE FUNCTION
# =====================================================================
def run_model_inference(image_path):
    global model
    if model is None:
        return "Model Init Failure", 0.0
        
    try:
        # 1. Read Image and convert BGR (OpenCV default) to RGB
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Unreadable Image", 0.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Apply your exact Data Lake preprocessing pipeline
        img = crop_brain_contour(img)
        img = letterbox_resize(img, target_size=(224, 224))
        processed_img = apply_medical_enhancement(img)
        
        # 3. Convert the NumPy array back to a PIL Image for PyTorch
        pil_img = Image.fromarray(processed_img)
        
        # 4. PyTorch Transforms (Matching your `eval_tf` from training)
        transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 5. Apply transforms and add the Batch Dimension -> (1, 3, 224, 224)
        input_tensor = transform_pipeline(pil_img).unsqueeze(0).to(device)

        # 6. Model Execution
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence_tensor, class_idx_tensor = torch.max(probabilities, dim=1)
            
            confidence = confidence_tensor.item() * 100
            class_idx = class_idx_tensor.item()
            
        detected_class = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Unknown ({class_idx})"
        return detected_class, confidence
        
    except Exception as e:
        print("\n❌ INFERENCE RUNTIME ERROR DETECTED:")
        traceback.print_exc()
        return "Processing Error", 0.0

# =====================================================================
# WINDOW 2: Results Display
# =====================================================================
class ResultWindow(QWidget):
    def __init__(self, image_path, class_name, confidence):
        super().__init__()
        self.setWindowTitle("Analysis Results")
        self.setFixedSize(450, 500)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Scan Analysis Complete")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #a6e3a1;") 
        layout.addWidget(title)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("border: 2px solid #45475a; border-radius: 10px; background-color: #11111b;")
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(300, 250, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.img_label.setPixmap(scaled_pixmap)
        layout.addWidget(self.img_label)

        result_layout = QVBoxLayout()
        class_label = QLabel(f"Detected Class: {class_name}")
        class_label.setFont(QFont("Arial", 14, QFont.Weight.Medium))
        class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        conf_label = QLabel(f"Confidence Score: {confidence:.2f}%")
        conf_label.setFont(QFont("Arial", 13))
        conf_label.setStyleSheet("color: #bac2de;")
        conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        result_layout.addWidget(class_label)
        result_layout.addWidget(conf_label)
        layout.addLayout(result_layout)

        self.close_btn = QPushButton("Scan Another Image")
        self.close_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa; color: #11111b; 
                border-radius: 8px; padding: 12px;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)
        self.setLayout(layout)

# =====================================================================
# WINDOW 1: Main Detector & Drag/Drop Target
# =====================================================================
class BrainTumorDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Detector")
        self.setFixedSize(600, 450)
        self.setAcceptDrops(True) 
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; font-family: 'Arial'; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        title_label = QLabel("Brain Tumor Detector")
        title_label.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #89b4fa;")
        main_layout.addWidget(title_label)

        self.drop_area = QLabel("Drag & Drop MRI Scan Here\n\n— or —\n\nClick to Browse Files")
        self.drop_area.setFont(QFont("Arial", 12))
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 3px dashed #45475a;
                border-radius: 15px;
                background-color: #11111b;
                color: #a6adc8;
            }
            QLabel:hover {
                border-color: #89b4fa;
                background-color: #181825;
                color: #cdd6f4;
            }
        """)
        self.drop_area.mousePressEvent = self.open_file_dialog
        main_layout.addWidget(self.drop_area)
        self.result_windows = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_area.setStyleSheet("QLabel { border: 3px dashed #a6e3a1; background-color: #181825; color: #a6e3a1; border-radius: 15px; }")

    def dragLeaveEvent(self, event):
        self.drop_area.setStyleSheet("QLabel { border: 3px dashed #45475a; background-color: #11111b; color: #a6adc8; border-radius: 15px; }")

    def dropEvent(self, event):
        self.dragLeaveEvent(event)
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.process_image(file_path)

    def open_file_dialog(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open MRI Scan", "", "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)")
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        class_name, confidence = run_model_inference(file_path)
        res_window = ResultWindow(file_path, class_name, confidence)
        res_window.show()
        self.result_windows.append(res_window)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrainTumorDetectorApp()
    window.show()
    sys.exit(app.exec())
