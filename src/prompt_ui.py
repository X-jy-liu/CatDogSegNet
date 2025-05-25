import sys
import os
import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QRadioButton, QButtonGroup,
                             QSlider, QGroupBox, QComboBox)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtCore import Qt, QPoint, QRect

from models.prompt_segmentation import PromptSegmentation
from data.preprocessing import resize_with_padding
from utils.restore_image_size import restore_original_mask

class SegmentationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Segmentation Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize state variables
        self.original_image = None
        self.display_image = None
        self.original_size = None
        self.current_mask = None
        self.combined_mask = None
        self.prompts = []  # Store points/boxes for inference
        self.temp_box = None  # For box drawing
        self.mode = "point"  # Default mode is point placement
        self.current_class = 1  # Default class is foreground (1)
        self.target_size = 512  # Model input size
        self.threshold = 0.5  # Default threshold for segmentation
        
        # Load model
        self.model = self.load_model()
        
        # Set up UI
        self.setup_ui()
    
    def load_model(self):
        """Load the prompt-based segmentation model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load model
        model = PromptSegmentation()
        checkpoint_path = r"E:\Oxford-IIIT_Pet_Dataset_Segementation_test\params\best_prompt_checkpoint_epoch_1.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        return model
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left side: image display
        self.canvas = QLabel()
        self.canvas.setMinimumSize(512, 512)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.canvas.mousePressEvent = self.canvas_click
        self.canvas.mouseMoveEvent = self.canvas_move
        self.canvas.mouseReleaseEvent = self.canvas_release
        
        # Right side: controls
        controls_layout = QVBoxLayout()
        
        # Image loading group
        image_group = QGroupBox("Image")
        image_layout = QVBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        image_layout.addWidget(self.load_image_btn)
        image_group.setLayout(image_layout)
        controls_layout.addWidget(image_group)
        
        # Prompt tools group
        prompt_group = QGroupBox("Prompt Tools")
        prompt_layout = QVBoxLayout()
        
        # Prompt type selection
        prompt_type_layout = QHBoxLayout()
        self.point_radio = QRadioButton("Point")
        self.box_radio = QRadioButton("Box")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(lambda: self.set_mode("point"))
        self.box_radio.toggled.connect(lambda: self.set_mode("box"))
        prompt_type_layout.addWidget(self.point_radio)
        prompt_type_layout.addWidget(self.box_radio)
        prompt_layout.addLayout(prompt_type_layout)
        
        # # Class selection
        # class_layout = QHBoxLayout()
        # class_layout.addWidget(QLabel("Class:"))
        # self.class_combo = QComboBox()
        # self.class_combo.addItems(["Background (0)", "Foreground (1)"])
        # self.class_combo.setCurrentIndex(1)  # Default to foreground
        # self.class_combo.currentIndexChanged.connect(self.change_class)
        # class_layout.addWidget(self.class_combo)
        # prompt_layout.addLayout(class_layout)
        
        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.valueChanged.connect(self.change_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel(f"{self.threshold:.2f}")
        threshold_layout.addWidget(self.threshold_label)
        prompt_layout.addLayout(threshold_layout)
        
        # Action buttons
        self.run_btn = QPushButton("Run Segmentation")
        self.run_btn.clicked.connect(self.run_segmentation)
        self.run_btn.setEnabled(False)
        prompt_layout.addWidget(self.run_btn)
        
        self.clear_prompts_btn = QPushButton("Clear Prompts")
        self.clear_prompts_btn.clicked.connect(self.clear_prompts)
        self.clear_prompts_btn.setEnabled(False)
        prompt_layout.addWidget(self.clear_prompts_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all)
        self.clear_all_btn.setEnabled(False)
        prompt_layout.addWidget(self.clear_all_btn)
        
        prompt_group.setLayout(prompt_layout)
        controls_layout.addWidget(prompt_group)
        
        # Result group
        result_group = QGroupBox("Results")
        result_layout = QVBoxLayout()
        
        self.save_mask_btn = QPushButton("Save Mask")
        self.save_mask_btn.clicked.connect(self.save_mask)
        self.save_mask_btn.setEnabled(False)
        result_layout.addWidget(self.save_mask_btn)
        
        self.toggle_view_btn = QPushButton("Toggle Overlay")
        self.toggle_view_btn.clicked.connect(self.toggle_overlay)
        self.toggle_view_btn.setEnabled(False)
        result_layout.addWidget(self.toggle_view_btn)
        
        result_group.setLayout(result_layout)
        controls_layout.addWidget(result_group)
        
        # Status indicator
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        
        # Add stretcher to push everything up
        controls_layout.addStretch()
        
        # Arrange layouts
        main_layout.addWidget(self.canvas, 3)
        main_layout.addLayout(controls_layout, 1)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def set_mode(self, mode):
        """Set the prompt mode (point or box)"""
        self.mode = mode
        self.temp_box = None
        self.update_canvas()
    
    def change_class(self, index):
        """Update the current class when combo box selection changes"""
        self.current_class = 0 if index == 0 else 1
    
    def change_threshold(self, value):
        """Update threshold value from slider"""
        self.threshold = value / 100.0
        self.threshold_label.setText(f"{self.threshold:.2f}")
        # Re-run segmentation if we have a current mask
        if self.current_mask is not None:
            self.run_segmentation()
    
    def load_image(self):
        """Load an image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # Load and store the original image
            self.original_image = np.array(Image.open(file_path).convert("RGB"))
            self.original_size = self.original_image.shape[:2]
            
            # Resize for display
            self.resize_and_display_image()
            
            # Enable UI elements
            self.run_btn.setEnabled(True)
            self.clear_all_btn.setEnabled(True)
            
            # Clear existing prompts and masks
            self.prompts = []
            self.current_mask = None
            self.combined_mask = None
            self.status_label.setText("Image loaded. Add prompts and run segmentation.")
    
    def resize_and_display_image(self):
        """Resize the original image and update display"""
        # Resize while maintaining aspect ratio
        img_resized = resize_with_padding(self.original_image, self.target_size, False)
        self.display_image = img_resized.copy()
        
        # Update canvas
        self.update_canvas()
    
    def update_canvas(self):
        """Update the canvas with current image and overlays"""
        if self.display_image is None:
            return
        
        # Create a copy of the display image for drawing
        canvas_img = self.display_image.copy()
        
        # Draw mask overlay if available
        if self.combined_mask is not None:
            # Create a colored overlay (semi-transparent)
            overlay = np.zeros_like(canvas_img)
            overlay[self.combined_mask > 0] = [0, 255, 0]  # Green overlay
            
            # Blend with original image
            alpha = 0.5
            canvas_img = cv2.addWeighted(overlay, alpha, canvas_img, 1-alpha, 0)
        
        # Draw prompts
        for prompt in self.prompts:
            if prompt[0] == "point":
                x, y, cls = prompt[1]
                color = (0, 0, 255) if cls == 0 else (255, 0, 0)  # Red for foreground, blue for background
                cv2.circle(canvas_img, (x, y), 5, color, -1)
            elif prompt[0] == "box":
                x1, y1, x2, y2, cls = prompt[1]
                color = (0, 0, 255) if cls == 0 else (255, 0, 0)
                cv2.rectangle(canvas_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw temporary box if in progress
        if self.temp_box is not None:
            x1, y1, x2, y2 = self.temp_box
            cv2.rectangle(canvas_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Convert to QImage and display
        h, w, c = canvas_img.shape
        q_img = QImage(canvas_img.data, w, h, w*c, QImage.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(q_img))
    
    def canvas_click(self, event):
        """Handle mouse click on canvas"""
        if self.display_image is None:
            return
        
        # Calculate image position within the canvas
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        img_height, img_width = self.display_image.shape[:2]
        
        # Calculate offsets for centered image
        x_offset = max(0, (canvas_width - img_width) // 2)
        y_offset = max(0, (canvas_height - img_height) // 2)
        
        # Adjust coordinates
        x = event.x() - x_offset
        y = event.y() - y_offset
        
        # Check if click is within image bounds
        if 0 <= x < img_width and 0 <= y < img_height:
            if self.mode == "point":
                # Add point prompt immediately
                self.prompts.append(("point", (x, y, self.current_class)))
                self.clear_prompts_btn.setEnabled(True)
                self.update_canvas()
            elif self.mode == "box":
                # Start box drawing
                self.temp_box = [x, y, x, y]
        else:
            # Click outside the image area
            self.status_label.setText("Click within the image area")

    def canvas_move(self, event):
        """Handle mouse movement on canvas (for box drawing)"""
        if self.temp_box is None:
            return
        
        # Calculate image position within the canvas
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        img_height, img_width = self.display_image.shape[:2]
        
        # Calculate offsets for centered image
        x_offset = max(0, (canvas_width - img_width) // 2)
        y_offset = max(0, (canvas_height - img_height) // 2)
        
        # Adjust coordinates
        x = event.x() - x_offset
        y = event.y() - y_offset
        
        # Constrain coordinates to image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        self.temp_box[2], self.temp_box[3] = x, y
        self.update_canvas()

    def canvas_release(self, event):
        """Handle mouse release on canvas (for box drawing)"""
        if self.temp_box is None:
            return
        
        # Calculate image position within the canvas
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        img_height, img_width = self.display_image.shape[:2]
        
        # Calculate offsets for centered image
        x_offset = max(0, (canvas_width - img_width) // 2)
        y_offset = max(0, (canvas_height - img_height) // 2)
        
        # Adjust coordinates
        x = event.x() - x_offset
        y = event.y() - y_offset
        
        # Constrain coordinates to image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        x1, y1 = self.temp_box[0], self.temp_box[1]
        x2, y2 = x, y
        
        # Ensure correct ordering (x1,y1 is top-left, x2,y2 is bottom-right)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Only add if box has some area
        if x2 > x1 and y2 > y1:
            self.prompts.append(("box", (x1, y1, x2, y2, self.current_class)))
            self.clear_prompts_btn.setEnabled(True)
        
        self.temp_box = None
        self.update_canvas()
    
    def generate_heatmap(self, prompt, img_shape):
        """Generate a heatmap for the given prompt"""
        h, w = img_shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if prompt[0] == "point":
            x, y, _ = prompt[1]
            # Create a Gaussian heatmap centered at the point
            sigma = 10.0  # Controls the spread of the Gaussian
            y_grid, x_grid = np.mgrid[0:h, 0:w]
            heatmap = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        elif prompt[0] == "box":
            x1, y1, x2, y2, _ = prompt[1]
            # Create a binary heatmap for the box region
            heatmap[y1:y2+1, x1:x2+1] = 1.0
        
        return heatmap
    
    def run_segmentation(self):
        """Run segmentation based on current prompts"""
        if not self.prompts or self.display_image is None:
            self.status_label.setText("Add at least one prompt before running segmentation")
            return
        
        self.status_label.setText("Running segmentation...")
        
        # Process each prompt
        self.combined_mask = None
        
        for prompt in self.prompts:
            # Generate heatmap for this prompt
            heatmap = self.generate_heatmap(prompt, (self.target_size, self.target_size))
            
            # Prepare inputs for the model - APPLY STANDARD TRANSFORM
            from data.preprocessing import standard_transform
            # Convert to PIL Image for transform
            pil_img = Image.fromarray(self.display_image)
            # Apply the same transform used during training/inference
            img_tensor = standard_transform(pil_img).unsqueeze(0)  # Add batch dimension
            
            heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
            
            # Move tensors to device
            device = next(self.model.parameters()).device
            img_tensor = img_tensor.to(device)
            heatmap_tensor = heatmap_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(
                    image=img_tensor,
                    prompt_heatmap=heatmap_tensor,
                    point_class= None
                )
            
            # Apply threshold to get binary mask
            pred_mask = (torch.sigmoid(output) > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
            
            # Union with previous masks
            if self.combined_mask is None:
                self.combined_mask = pred_mask
            else:
                self.combined_mask = np.logical_or(self.combined_mask, pred_mask).astype(np.uint8)
        
        # Enable save mask button
        self.save_mask_btn.setEnabled(True)
        self.toggle_view_btn.setEnabled(True)
        
        # Update canvas with mask overlay
        self.update_canvas()
        
        self.status_label.setText("Segmentation complete")
    
    def toggle_overlay(self):
        """Toggle between showing the mask overlay and original image"""
        if self.combined_mask is None:
            return
        
        # Toggle by temporarily removing and restoring the mask
        if hasattr(self, '_temp_mask'):
            self.combined_mask = self._temp_mask
            delattr(self, '_temp_mask')
        else:
            self._temp_mask = self.combined_mask
            self.combined_mask = None
        
        self.update_canvas()
    
    def save_mask(self):
        """Save the current mask to a file"""
        if self.combined_mask is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "", "PNG Files (*.png)"
        )
        
        if file_path:
            # Resize mask back to original image size
            mask_resized = cv2.resize(
                self.combined_mask * 255,  # Scale to 0-255
                (self.original_size[1], self.original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Save mask
            cv2.imwrite(file_path, mask_resized)
            self.status_label.setText(f"Mask saved to {file_path}")
    
    def clear_prompts(self):
        """Clear all prompts but keep the loaded image"""
        self.prompts = []
        self.combined_mask = None
        self.clear_prompts_btn.setEnabled(False)
        self.save_mask_btn.setEnabled(False)
        self.toggle_view_btn.setEnabled(False)
        self.update_canvas()
        self.status_label.setText("Prompts cleared")
    
    def clear_all(self):
        """Clear everything including the loaded image"""
        self.original_image = None
        self.display_image = None
        self.original_size = None
        self.prompts = []
        self.combined_mask = None
        self.temp_box = None
        
        # Reset UI
        self.canvas.clear()
        self.canvas.setPixmap(QPixmap())
        self.run_btn.setEnabled(False)
        self.clear_prompts_btn.setEnabled(False)
        self.clear_all_btn.setEnabled(False)
        self.save_mask_btn.setEnabled(False)
        self.toggle_view_btn.setEnabled(False)
        self.status_label.setText("Ready")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationUI()
    window.show()
    sys.exit(app.exec_())