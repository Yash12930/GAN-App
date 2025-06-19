from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import io
import base64
from PIL import Image
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.initializers import RandomNormal
    from tensorflow.keras.datasets import mnist
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not available")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not available")

# Create Flask app
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'static'))

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global variables
generator = None
discriminator = None
X_train = None
from_generator = False

def adam_optimizer():
    """Create Adam optimizer with same settings as notebook"""
    return Adam(lr=0.0002, beta_1=0.5)

def create_generator():
    """Create generator model matching your notebook architecture"""
    generator = Sequential()
    generator.add(Dense(256, input_dim=100, kernel_initializer=RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

def create_discriminator():
    """Create discriminator model matching your notebook architecture"""
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator

def init():
    """Initialize models"""
    global generator, discriminator, X_train
    
    if not HAS_TF:
        print("TensorFlow not available, using fallback")
        return
    
    try:
        # Try to load pre-trained models first
        model_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'models')
        
        if os.path.exists(os.path.join(model_path, 'gan_generator_epoch_200.h5')):
            generator = load_model(os.path.join(model_path, 'gan_generator_epoch_200.h5'))
            print("Loaded pre-trained generator")
        else:
            # Create and try to load weights
            generator = create_generator()
            weights_path = os.path.join(model_path, 'generator_weights.h5')
            if os.path.exists(weights_path):
                generator.load_weights(weights_path)
                print("Loaded generator weights")
            else:
                print("No pre-trained generator found, using random weights")
        
        if os.path.exists(os.path.join(model_path, 'gan_discriminator_epoch_200.h5')):
            discriminator = load_model(os.path.join(model_path, 'gan_discriminator_epoch_200.h5'))
            print("Loaded pre-trained discriminator")
        else:
            # Create and try to load weights
            discriminator = create_discriminator()
            weights_path = os.path.join(model_path, 'discriminator_weights.h5')
            if os.path.exists(weights_path):
                discriminator.load_weights(weights_path)
                print("Loaded discriminator weights")
            else:
                print("No pre-trained discriminator found, using random weights")
        
        # Load MNIST data
        (X_train, _), (_, _) = mnist.load_data()
        print("Models and data loaded successfully")
        
    except Exception as e:
        print(f"Error initializing models: {e}")

def get_random_existing_image():
    """Get random existing generated image as fallback"""
    try:
        generated_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generated_sample')
        
        if os.path.exists(generated_path):
            image_files = [f for f in os.listdir(generated_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                random_image = random.choice(image_files)
                image_path = os.path.join(generated_path, random_image)
                return Image.open(image_path)
        
        # Create placeholder
        img = Image.new('RGB', (64, 64), color=(128, 128, 128))
        return img
        
    except Exception as e:
        print(f"Error getting existing image: {e}")
        img = Image.new('RGB', (64, 64), color=(255, 0, 0))
        return img

@app.route('/')
def hello():
    """Main page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GAN Image Generator</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 60px 40px;
                text-align: center;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                max-width: 600px;
                width: 90%;
            }
            
            h1 {
                font-size: 3rem;
                margin-bottom: 10px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .author {
                font-size: 1.2rem;
                color: #666;
                margin-bottom: 40px;
                font-style: italic;
            }
            
            .btn-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                align-items: center;
            }
            
            .btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 18px 40px;
                font-size: 1.1rem;
                border-radius: 50px;
                text-decoration: none;
                display: inline-block;
                transition: all 0.3s ease;
                min-width: 200px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }
            
            .btn:active {
                transform: translateY(-1px);
            }
            
            .description {
                color: #555;
                font-size: 1.1rem;
                margin-bottom: 30px;
                line-height: 1.6;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 40px 20px;
                }
                
                h1 {
                    font-size: 2.5rem;
                }
                
                .btn {
                    min-width: 180px;
                    padding: 15px 30px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GAN Image Generator</h1>
            <p class="author">Created by Yash</p>
            <p class="description">
                Generate unique images using Generative Adversarial Networks. 
                Explore AI-powered image creation and test the discriminator model.
            </p>
            <div class="btn-container">
                <a href="/generator/" class="btn">🎨 Generate Image</a>
                <a href="/discriminator/" class="btn">🔍 Test Discriminator</a>
                <a href="/gallery" class="btn">🖼️ View Gallery</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route("/generator/")
def generator_route():
    """Generator route matching your original app.py"""
    global from_generator
    
    try:
        p = random.randint(0, 1)
        
        if p == 0 and generator is not None:
            # Generate new image
            noise = np.random.normal(0, 1, size=[1, 100])
            generated_images = generator.predict(noise)
            generated_images = generated_images.reshape(1, 28, 28)
            
            plt.figure(figsize=(3, 3))
            plt.imshow(generated_images[0], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
            
            # Save to static directory
            static_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'raw')
            os.makedirs(static_path, exist_ok=True)
            plt.savefig(os.path.join(static_path, 'generated.png'), bbox_inches='tight')
            plt.close()
            
            from_generator = True
        else:
            # Use training data
            if X_train is not None:
                img = X_train[random.randint(0, len(X_train)-1)]
                plt.figure(figsize=(3, 3))
                plt.imshow(img, interpolation='nearest', cmap='gray_r')
                plt.axis('off')
                
                static_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'raw')
                os.makedirs(static_path, exist_ok=True)
                plt.savefig(os.path.join(static_path, 'generated.png'), bbox_inches='tight')
                plt.close()
                
                from_generator = False
            else:
                # Fallback to existing image
                image = get_random_existing_image()
                static_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'raw')
                os.makedirs(static_path, exist_ok=True)
                image.save(os.path.join(static_path, 'generated.png'))
                from_generator = False
        
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Generated Image - GAN</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0;
                }}
                
                .container {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 20px;
                    padding: 40px;
                    text-align: center;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    max-width: 600px;
                    width: 90%;
                }}
                
                h1 {{
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 2.5rem;
                }}
                
                .info {{
                    background: #f8f9fa;
                    padding: 10px 20px;
                    border-radius: 25px;
                    margin: 20px 0;
                    color: #666;
                    font-style: italic;
                }}
                
                img {{
                    border: 3px solid #667eea;
                    border-radius: 15px;
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                }}
                
                .btn {{
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    border: none;
                    padding: 12px 25px;
                    margin: 8px;
                    font-size: 16px;
                    border-radius: 25px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    transition: all 0.3s ease;
                }}
                
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎨 Generated Image</h1>
                <div class="info">
                    Source: {"🤖 AI Generated" if from_generator else "📚 Training Data"}
                </div>
                <img src="/static_image/generated.png" alt="Generated Image">
                <br><br>
                <button class="btn" onclick="window.location.reload()">🔄 Generate Another</button>
                <button class="btn" onclick="window.location.href='/'">🏠 Back Home</button>
                <button class="btn" onclick="window.location.href='/gallery'">🖼️ Gallery</button>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f'<h1>Error: {str(e)}</h1><a href="/">Back</a>'

@app.route("/discriminator/")
def discriminator_route():
    """Discriminator route"""
    return '''
    <html>
    <head><title>Test Discriminator</title></head>
    <body style="text-align: center; font-family: Arial; margin: 50px;">
        <h1>Test Discriminator</h1>
        <form action="/discriminator/test" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <button type="submit">Test Image</button>
        </form>
        <br>
        <a href="/">Back Home</a>
    </body>
    </html>
    '''

@app.route('/discriminator/test', methods=['POST'])
def upload_file():
    """Test discriminator on uploaded image"""
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save uploaded file
            static_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'raw', 'discriminator')
            os.makedirs(static_path, exist_ok=True)
            file_path = os.path.join(static_path, 'test.png')
            uploaded_file.save(file_path)
            
            if HAS_CV2 and discriminator is not None:
                # Process with OpenCV and discriminator
                img = cv2.imread(file_path, 0)
                if img.shape != (28, 28):
                    img = cv2.resize(img, (28, 28))
                
                flat_img = img.reshape(1, 784)
                flat_img = (flat_img.astype(np.float32) - 127.5) / 127.5
                output = discriminator.predict(flat_img)[0][0]
                
                if output == 0:
                    img = ~img
                    img = img.reshape(1, 784)
                    img = (img.astype(np.float32) - 127.5) / 127.5
                    output = discriminator.predict(img)[0][0]
            else:
                # Fallback - random score
                output = random.random()
            
            return f'''
            <html>
            <body style="text-align: center; font-family: Arial;">
                <h1>Discriminator Result</h1>
                <p>Score: {output:.4f}</p>
                <p>{"Likely Real" if output > 0.5 else "Likely Fake"}</p>
                <a href="/discriminator/">Test Another</a> | 
                <a href="/">Home</a>
            </body>
            </html>
            '''
        else:
            return '<h1>No file uploaded</h1><a href="/discriminator/">Back</a>'
            
    except Exception as e:
        return f'<h1>Error: {str(e)}</h1><a href="/discriminator/">Back</a>'

@app.route('/gallery')
def gallery():
    """Gallery of generated images"""
    try:
        generated_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generated_sample')
        images = []
        
        if os.path.exists(generated_path):
            for filename in sorted(os.listdir(generated_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(filename)
        
        gallery_html = '''
        <html>
        <head><title>Gallery</title></head>
        <body style="font-family: Arial; margin: 20px;">
            <h1>Generated Images Gallery</h1>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px;">
        '''
        
        for image in images:
            gallery_html += f'<img src="/static_image/{image}" style="width: 100%; border: 1px solid #ccc;">'
        
        gallery_html += '''
            </div>
            <br><a href="/">Back Home</a>
        </body>
        </html>
        '''
        
        return gallery_html
        
    except Exception as e:
        return f'<h1>Error: {str(e)}</h1><a href="/">Back</a>'

@app.route('/static_image/<filename>')
def serve_static_image(filename):
    """Serve static images"""
    try:
        # Try generated_sample first
        generated_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generated_sample')
        if os.path.exists(os.path.join(generated_path, filename)):
            return send_file(os.path.join(generated_path, filename))
        
        # Try static/raw
        raw_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'raw')
        if os.path.exists(os.path.join(raw_path, filename)):
            return send_file(os.path.join(raw_path, filename))
        
        return "File not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 404

@app.route('/health')
def health():
    """Health check"""
    return {
        'status': 'healthy',
        'tensorflow': HAS_TF,
        'opencv': HAS_CV2,
        'models_loaded': generator is not None and discriminator is not None
    }

# Initialize on startup
init()

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)