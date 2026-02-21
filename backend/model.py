import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class XRayModel:
    def __init__(self):
        if not os.path.exists('xray_model.pth'):
            raise FileNotFoundError("Model file 'xray_model.pth' not found. Please train the model first.")
        
        # Try loading MobileNetV2 first
        try:
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
            self.model.load_state_dict(torch.load('xray_model.pth', map_location='cpu'))
            self.model_type = 'mobilenet'
        except:
            # Fallback to ResNet18
            try:
                self.model = models.resnet18(pretrained=False)
                self.model.fc = nn.Linear(512, 2)
                self.model.load_state_dict(torch.load('xray_model.pth', map_location='cpu'))
                self.model_type = 'resnet'
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
        
        self.classes = ["Normal", "Pneumonia"]
        self.gradients = None
        self.activations = None
        
        # Register hooks for Grad-CAM
        if self.model_type == 'resnet':
            self.target_layer = self.model.layer4[-1]
        else:
            self.target_layer = self.model.features[-1]
        
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor.requires_grad_(True)

            # Forward pass
            self.model.train()
            output = self.model(image_tensor)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            # Backward pass for Grad-CAM
            self.model.zero_grad()
            output[0, predicted].backward()

            disease = self.classes[predicted.item()]
            confidence_score = float(confidence)
            
            # Generate Grad-CAM heatmap
            self.model.eval()
            heatmap_path = self._generate_gradcam(image_path)

            return disease, confidence_score, heatmap_path
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown", 0.0, self.create_simple_heatmap(image_path)
    
    def _generate_gradcam(self, image_path):
        try:
            if self.gradients is None or self.activations is None:
                print("No gradients or activations captured")
                return self.create_simple_heatmap(image_path)
            
            # Compute weights from gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze()
            
            # Apply ReLU and normalize
            cam = torch.relu(cam)
            cam = cam.detach().cpu().numpy()
            cam = cam / (np.max(cam) + 1e-8)
            cam = cv2.resize(cam, (224, 224))
            
            # Load original image
            img = cv2.imread(image_path)
            if img is None:
                return self.create_simple_heatmap(image_path)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (224, 224))
            
            # Create heatmap overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            
            # Create side-by-side visualization
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Original image
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original X-Ray', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Heatmap overlay
            axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            output_path = 'heatmap.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            import traceback
            traceback.print_exc()
            return self.create_simple_heatmap(image_path)
    
    def create_simple_heatmap(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                cv2.imwrite('heatmap.png', img)
            return 'heatmap.png'
        except:
            return None
