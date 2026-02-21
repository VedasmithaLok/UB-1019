import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook to capture gradients and activations
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, image_tensor, class_idx):
        # Forward pass
        self.model.eval()
        output = self.model(image_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        # Generate heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

def apply_heatmap_on_image(image_path, heatmap, output_path="heatmap.png"):
    # Load original image
    img = cv2.imread(image_path)
    
    # Convert grayscale to 3-channel if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img = cv2.resize(img, (224, 224))
    
    # Normalize heatmap to 0-1 range
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    
    # Lower threshold to show more regions
    threshold = 0.1
    heatmap[heatmap < threshold] = 0
    
    # Stronger contrast boost
    heatmap = np.power(heatmap, 0.4)
    
    # Resize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply JET colormap for red regions
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Keep original image brighter
    img_bright = (img * 0.6).astype(np.uint8)
    
    # Overlay with stronger heatmap
    superimposed = cv2.addWeighted(img_bright, 0.4, heatmap_colored, 0.6, 0)
    
    # Boost red channel more aggressively
    superimposed[:, :, 2] = np.clip(superimposed[:, :, 2] * 2.0, 0, 255)
    
    cv2.imwrite(output_path, superimposed)
    
    return output_path
