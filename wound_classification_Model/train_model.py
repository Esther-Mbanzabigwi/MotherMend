import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
base_path = "wound-classification-using-images-and-locations-main/wound-classification-using-images-and-locations-main/dataset"
train_csv = os.path.join(base_path, "Train/wound_locations_Labels_AZH_Train.csv")
test_csv = os.path.join(base_path, "Test/wound_locations_Labels_AZH_Test.csv")
train_img_dir = os.path.join(base_path, "Train")
test_img_dir = os.path.join(base_path, "Test")

# Read CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Clean data
train_df["index"] = train_df["index"].str.replace(r"\\", "/", regex=True)
test_df["index"] = test_df["index"].str.replace(r"\\", "/", regex=True)

# Drop NaNs and filter labels
valid_labels = [0, 1, 2, 3, 4, 5]
train_df.dropna(subset=["Labels"], inplace=True)
test_df.dropna(subset=["Labels"], inplace=True)
train_df = train_df[train_df["Labels"].isin(valid_labels)]
test_df = test_df[test_df["Labels"].isin(valid_labels)]

# Create label mapping
label_map = {
    0: "Eschar", 1: "Granulating Tissue", 2: "Healthy Tissue",
    3: "Necrotic Tissue", 4: "Slough", 5: "Undefined"
}
train_df["label_name"] = train_df["Labels"].map(label_map)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["Labels"]),
    y=train_df["Labels"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Dataset class
class WoundDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_train=True):
        """
        Initialize the dataset.
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels
            image_dir (str): Directory containing the images
            transform: Optional transform to be applied on a sample
            is_train (bool): Whether this is training data
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
        # Validate data at initialization
        self._validate_data()
        
        # Cache image paths
        self.image_paths = []
        self.valid_indices = []
        
        for idx in range(len(self.df)):
            img_name = self.df.loc[idx, 'index']
            img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
            
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Image not found at {img_path}")
        
        print(f"Dataset initialized with {len(self.valid_indices)} valid images out of {len(self.df)} total entries")
    
    def _validate_data(self):
        """Validate the dataset at initialization."""
        # Check if directory exists
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        # Check required columns
        required_cols = ['index', 'Labels']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate labels
        valid_labels = set(range(6))  # 0 to 5
        invalid_labels = set(self.df['Labels'].unique()) - valid_labels
        if invalid_labels:
            raise ValueError(f"Invalid labels found in dataset: {invalid_labels}")
    
    def __len__(self):
        """Return the number of valid images in the dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label) where image is a transformed PIL Image and label is an integer
        """
        try:
            # Get the actual index from valid indices
            actual_idx = self.valid_indices[idx]
            
            # Get image path and label
            img_path = self.image_paths[idx]
            label = int(self.df.loc[actual_idx, 'Labels'])
            
            try:
                # Load and convert image
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
                
                # Apply transformations
                if self.transform:
                    try:
                        image = self.transform(image)
                    except Exception as e:
                        print(f"Error applying transform to image {img_path}: {str(e)}")
                        # Return a basic transformed image
                        image = self._get_default_image()
                
                return image, label
                
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                return self._get_default_image(), 5  # Return default image and Undefined class
                
        except Exception as e:
            print(f"Critical error in __getitem__ for idx {idx}: {str(e)}")
            return self._get_default_image(), 5
    
    def _get_default_image(self):
        """Return a default image tensor when loading fails."""
        if self.transform:
            try:
                # Create a black image and apply the same transforms
                default_img = Image.new('RGB', (224, 224), 'black')
                return self.transform(default_img)
            except:
                # If transform fails, return a zero tensor
                return torch.zeros((3, 224, 224))
        return torch.zeros((3, 224, 224))
    
    def get_image_path(self, idx):
        """Get the image path for a given index (useful for debugging)."""
        if 0 <= idx < len(self.valid_indices):
            return self.image_paths[idx]
        return None

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data and create dataloaders
train_df_sampled, val_df = train_test_split(
    train_df, test_size=0.15, stratify=train_df["Labels"], random_state=42
)

# Create datasets with error handling
try:
    train_dataset = WoundDataset(train_df_sampled, train_img_dir, train_transform, is_train=True)
    val_dataset = WoundDataset(val_df, train_img_dir, val_transform, is_train=False)
    test_dataset = WoundDataset(test_df, test_img_dir, val_transform, is_train=False)
except Exception as e:
    print(f"Error creating datasets: {str(e)}")
    raise

# Create dataloaders with proper worker initialization
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# Create dataloaders with minimal workers and proper initialization
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid multiprocessing issues
    pin_memory=True if torch.cuda.is_available() else False,
    worker_init_fn=worker_init_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False,
    num_workers=0,  # Set to 0 to avoid multiprocessing issues
    pin_memory=True if torch.cuda.is_available() else False,
    worker_init_fn=worker_init_fn
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False,
    num_workers=0,  # Set to 0 to avoid multiprocessing issues
    pin_memory=True if torch.cuda.is_available() else False,
    worker_init_fn=worker_init_fn
)

# Model
class ImprovedWoundClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ImprovedWoundClassifier, self).__init__()
        
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-8]:
            param.requires_grad = False
            
        # Modify final layers
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# Initialize model and training components
model = ImprovedWoundClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_val_acc = 0.0
    best_model_state = None
    patience = 7
    patience_counter = 0
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Accuracy: {val_accuracy:.2f}%\n')
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Save checkpoint at epochs 25 and 50
        if (epoch + 1) in [25, 50]:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss / len(train_loader),
                'val_accuracy': val_accuracy,
                'class_mapping': label_map,
            }
            torch.save(checkpoint, f'wound_classifier_checkpoint_epoch_{epoch+1}.pth')
            print(f'Saved checkpoint at epoch {epoch+1}')
        
        # Early stopping check and save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict()
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss / len(train_loader),
                'val_accuracy': val_accuracy,
                'class_mapping': label_map,
            }
            torch.save(checkpoint, 'wound_classifier_best.pth')
            print(f'New best model saved with validation accuracy: {val_accuracy:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    return best_val_acc

# Train the model
print("Starting model training...")
best_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
print(f"\nBest validation accuracy: {best_accuracy:.2f}%")

# Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=list(label_map.values()),
                              zero_division=0))

# Evaluate the model
print("\nEvaluating model on test set:")
evaluate_model(model, test_loader)

# Save the model
# Load best model state
model.load_state_dict(best_model_state)

# Save the best model to a consistent filename
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_mapping': label_map,
}, 'improved_wound_classifier.pth')
