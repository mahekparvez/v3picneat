from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model
print("Loading model...")
device = torch.device('cpu')

# THIS IS THE ACTUAL STRUCTURE FROM YOUR TRAINED MODEL
class FoodClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=False) # MobileNetV2 identification ability, my weights
        # Replace the backbone's classifier (we'll use our own)
        self.backbone.classifier = torch.nn.Identity()
        
        # Custom classifier head - EXACTLY as trained
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),           # classifier.0
            torch.nn.Linear(1280, 256),      # classifier.1
            torch.nn.ReLU(),                 # classifier.2
            torch.nn.Dropout(0.2),           # classifier.3
            torch.nn.Linear(256, 3)          # classifier.4
        )
        
        # Quantity estimation head
        self.quantity = torch.nn.Sequential(
            torch.nn.Linear(1280, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        features = torch.flatten(features, 1)
        food_class = self.classifier(features)
        quantity = self.quantity(features)
        return food_class, quantity

model = FoodClassifier()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("✓ Model loaded!")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ACCURATE USDA nutrition data per 100g
NUTRITION_DB = {
    'pizza': {
        'name': 'Pizza',
        'cal_per_100g': 266,
        'prot_per_100g': 11.0,
        'carb_per_100g': 33.0,
        'fat_per_100g': 10.0,
        'typical_serving_g': 150,
    },
    'pasta': {
        'name': 'Pasta',
        'cal_per_100g': 131,
        'prot_per_100g': 5.0,
        'carb_per_100g': 25.0,
        'fat_per_100g': 0.9,
        'typical_serving_g': 200,
    },
    'pancakes': {
        'name': 'Pancakes',
        'cal_per_100g': 227,
        'prot_per_100g': 6.4,
        'carb_per_100g': 28.0,
        'fat_per_100g': 9.0,
        'typical_serving_g': 100,
    }
}

CLASS_NAMES = ['pizza', 'pasta', 'pancakes']

def estimate_portion_weight(food_type, quantity_score, confidence):
    """Estimate portion weight based on model predictions"""
    typical_serving = NUTRITION_DB[food_type]['typical_serving_g']
    
    # Map quantity score to multiplier
    if quantity_score < 0.3:
        multiplier = 0.6
    elif quantity_score < 0.5:
        multiplier = 0.8
    elif quantity_score < 0.7:
        multiplier = 1.0
    elif quantity_score < 0.85:
        multiplier = 1.3
    else:
        multiplier = 1.6
    
    # Adjust for low confidence
    if confidence < 0.7:
        multiplier *= 0.85
    
    weight = typical_serving * multiplier
    return round(weight)

def calculate_nutrition(food_type, weight_grams):
    """Calculate nutrition based on weight"""
    nutrition = NUTRITION_DB[food_type]
    multiplier = weight_grams / 100.0
    
    return {
        'food': nutrition['name'],
        'weight': weight_grams,
        'cal': round(nutrition['cal_per_100g'] * multiplier, 1),
        'prot': round(nutrition['prot_per_100g'] * multiplier, 1),
        'carb': round(nutrition['carb_per_100g'] * multiplier, 1),
        'fat': round(nutrition['fat_per_100g'] * multiplier, 1),
    }

@app.route('/')
def home():
    return "Nutrition Tracker API is running! ✓"

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Transform and predict
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            class_output, quantity_output = model(img_tensor)
            probabilities = torch.softmax(class_output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            food_type = CLASS_NAMES[predicted_class.item()]
            conf_value = confidence.item()
            quantity_score = quantity_output.item()
        
        # Estimate weight
        weight = estimate_portion_weight(food_type, quantity_score, conf_value)
        
        # Calculate nutrition
        nutrition = calculate_nutrition(food_type, weight)
        nutrition['conf'] = conf_value
        
        # Calculate totals
        total = {
            'cal': nutrition['cal'],
            'prot': nutrition['prot'],
            'carb': nutrition['carb'],
            'fat': nutrition['fat'],
            'weight': weight
        }
        
        return jsonify({
            'items': [nutrition],
            'total': total
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Nutrition Tracker API Starting...")
    print("="*50)
    print("Server: https://pay-film-uncle-carl.trycloudflare.com")
    print("Test: https://pay-film-uncle-carl.trycloudflare.com")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=True)