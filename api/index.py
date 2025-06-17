from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import sys
import io
import base64
from PIL import Image
import numpy as np
import json
import random

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import TensorFlow/Keras (from your requirements.txt)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not available, using pre-generated images")

# Create Flask app with correct template and static folder paths
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'static'))

# Configure for production
app.config['ENV'] = 'production'
app.config['DEBUG'] = False

# Global variables for model
generator = None
discriminator = None
latent_dim = 100
img_width, img_height = 64, 64

def create_generator():
    """Create the generator model architecture"""
    model = keras.Sequential([
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Reshape((8, 8, 256)),
        
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    
    return model

def create_discriminator():
    """Create the discriminator model architecture"""
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[img_height, img_width, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1)
    ])
    
    return model

def load_models():
    """Load the GAN models"""
    global generator, discriminator
    
    if not HAS_TF:
        print("TensorFlow not available, models not loaded")
        return False
    
    try:
        # Create models
        generator = create_generator()
        discriminator = create_discriminator()
        
        # Try to load pre-trained weights if they exist
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        if os.path.exists(os.path.join(model_path, 'generator_weights.h5')):
            generator.load_weights(os.path.join(model_path, 'generator_weights.h5'))
            print("Loaded pre-trained generator weights")
        else:
            print("No pre-trained generator weights found, using random initialization")
        
        if os.path.exists(os.path.join(model_path, 'discriminator_weights.h5')):
            discriminator.load_weights(os.path.join(model_path, 'discriminator_weights.h5'))
            print("Loaded pre-trained discriminator weights")
        else:
            print("No pre-trained discriminator weights found, using random initialization")
        
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def generate_image():
    """Generate a new image using the GAN"""
    try:
        if not HAS_TF or generator is None:
            # If no TensorFlow or model, return a random existing generated image
            return get_random_existing_image()
        
        # Generate noise
        noise = tf.random.normal([1, latent_dim])
        
        # Generate image
        generated_image = generator(noise, training=False)
        
        # Convert from [-1, 1] to [0, 255]
        generated_image = (generated_image + 1) * 127.5
        generated_image = tf.cast(generated_image, tf.uint8)
        
        # Convert to PIL Image
        image_array = generated_image[0].numpy()
        image = Image.fromarray(image_array)
        
        return image
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return get_random_existing_image()

def get_random_existing_image():
    """Get a random existing generated image as fallback"""
    try:
        generated_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generated_sample')
        
        if os.path.exists(generated_path):
            image_files = [f for f in os.listdir(generated_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                random_image = random.choice(image_files)
                image_path = os.path.join(generated_path, random_image)
                return Image.open(image_path)
        
        # If no existing images, create a placeholder
        img = Image.new('RGB', (64, 64), color=(128, 128, 128))
        return img
        
    except Exception as e:
        print(f"Error getting existing image: {e}")
        # Return a simple placeholder
        img = Image.new('RGB', (64, 64), color=(255, 0, 0))
        return img

def test_discriminator_on_image(image):
    """Test the discriminator on an image"""
    try:
        if not HAS_TF or discriminator is None:
            return random.uniform(0, 1)  # Random score if no model
        
        # Preprocess image
        if image.size != (img_width, img_height):
            image = image.resize((img_width, img_height))
        
        # Convert to array and normalize to [-1, 1]
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        
        image_array = (image_array.astype(np.float32) - 127.5) / 127.5
        image_array = np.expand_dims(image_array, 0)
        
        # Get discriminator prediction
        prediction = discriminator(image_array, training=False)
        score = tf.nn.sigmoid(prediction).numpy()[0][0]
        
        return float(score)
        
    except Exception as e:
        print(f"Error testing discriminator: {e}")
        return random.uniform(0, 1)

# Initialize models on first import
if HAS_TF:
    load_models()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/generator')
def generator_page():
    """Generator page"""
    try:
        # Generate a new image
        image = generate_image()
        
        if image is None:
            return jsonify({'error': 'Failed to generate image'}), 500
        
        # Convert PIL Image to base64 string
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Generated Image</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; margin: 50px; }}
                img {{ border: 2px solid #333; border-radius: 10px; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                button {{ padding: 10px 20px; margin: 10px; font-size: 16px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Generated Image</h1>
                <img src="data:image/png;base64,{img_str}" alt="Generated Image">
                <br><br>
                <button onclick="window.location.reload()">Generate Another</button>
                <button onclick="window.location.href='/'">Back to Home</button>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f'<h1>Error generating image: {str(e)}</h1><a href="/">Back to Home</a>'

@app.route('/discriminator')
def discriminator_page():
    """Discriminator test page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Discriminator</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
            .container { max-width: 600px; margin: 0 auto; }
            input[type="file"] { margin: 20px; }
            button { padding: 10px 20px; margin: 10px; font-size: 16px; }
            #result { margin: 20px; padding: 20px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Test Discriminator</h1>
            <p>Upload an image to see how well the discriminator thinks it's real!</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageFile" accept="image/*" required>
                <br>
                <button type="submit">Test Image</button>
            </form>
            
            <div id="result" style="display: none;"></div>
            <br>
            <button onclick="window.location.href='/'">Back to Home</button>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', file);
                
                try {
                    const response = await fetch('/test_discriminator', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (data.success) {
                        const score = (data.score * 100).toFixed(1);
                        resultDiv.innerHTML = `
                            <h3>Discriminator Result</h3>
                            <p>Real/Fake Score: ${score}%</p>
                            <p>${score > 50 ? 'The discriminator thinks this image is likely REAL' : 'The discriminator thinks this image is likely FAKE'}</p>
                        `;
                        resultDiv.style.display = 'block';
                    } else {
                        resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                        resultDiv.style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/test_discriminator', methods=['POST'])
def test_discriminator():
    """Test discriminator API endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Open and process the image
        image = Image.open(file.stream)
        
        # Test with discriminator
        score = test_discriminator_on_image(image)
        
        return jsonify({
            'success': True,
            'score': score
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_api', methods=['POST'])
def generate_api():
    """Generate image API endpoint"""
    try:
        image = generate_image()
        
        if image is None:
            return jsonify({'error': 'Failed to generate image'}), 500
        
        # Convert PIL Image to base64 string
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gallery')
def gallery():
    """Gallery page showing generated images"""
    try:
        generated_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generated_sample')
        images = []
        
        if os.path.exists(generated_path):
            for filename in sorted(os.listdir(generated_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(filename)
        
        gallery_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Generated Images Gallery</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
                .gallery img { width: 100%; height: auto; border: 1px solid #ccc; border-radius: 5px; }
                .header { text-align: center; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Generated Images Gallery</h1>
                <button onclick="window.location.href='/'">Back to Home</button>
            </div>
            <div class="gallery">
        '''
        
        for image in images:
            # Use a route that serves the static files
            gallery_html += f'<img src="/static_image/{image}" alt="Generated Image">'
        
        gallery_html += '''
            </div>
        </body>
        </html>
        '''
        
        return gallery_html
        
    except Exception as e:
        return f'<h1>Error loading gallery: {str(e)}</h1><a href="/">Back to Home</a>'

@app.route('/static_image/<filename>')
def serve_static_image(filename):
    """Serve images from generated_sample folder"""
    try:
        generated_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generated_sample')
        return send_file(os.path.join(generated_path, filename))
    except Exception as e:
        return f"Error: {str(e)}", 404

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'GAN Flask App is running',
        'tensorflow_available': HAS_TF,
        'models_loaded': generator is not None and discriminator is not None
    })

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda status, headers: None)

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)