from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import datetime
import secrets
import os
from flask_cors import CORS
import logging 
from werkzeug.utils import secure_filename
import uuid
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
import numpy as np
from PIL import Image
import collections
import cv2

app = Flask(__name__)
CORS(app)

# Securely manage configurations
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'As+s01galaxysa')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'smart_home')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))

mysql = MySQL(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

#########################################################################################


# Smoothing window size
SMOOTHING_WINDOW = 5
smoothed_predictions = collections.deque(maxlen=SMOOTHING_WINDOW)

class_labels = ['happiness', 'neutral', 'sadness', 'angry']

# Load model from the database
def load_model_from_db():
    try:
        cursor = mysql.connection.cursor()
        # Fetch the active model from the database
        cursor.execute("SELECT file_path FROM models WHERE disabled = 0 ORDER BY created_at DESC LIMIT 1")
        model_data = cursor.fetchone()

        if model_data:
            model_path = model_data[0]
            loaded_model = tf.keras.models.load_model(model_path, compile=False)
            loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            return loaded_model
        else:
            raise Exception("No active model found in the database")
    
    except Exception as e:
        print(f"Error loading model from database: {str(e)}")
        raise
    
# Load the model using app context
with app.app_context():
    model = load_model_from_db()
    
import base64
import io

@app.route('/api/emotion-capture', methods=['POST'])
def emotion_capture():
    try:
        file = request.files['image']
        image = Image.open(file.stream)

        # Process image: Convert to grayscale and crop the face
        img_array, face_detected, processed_face = process_image(image)
        
        if not face_detected:
            return jsonify({'error': 'No face detected'}), 400

        # Predict emotion if a face is detected
        predicted_emotion = predict_emotion(img_array)

        # Encode the processed image (grayscale + cropped) as base64
        _, buffer = cv2.imencode('.jpg', processed_face)
        face_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'emotion': predicted_emotion,
            'face_image': face_image_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_image(image):
    # Convert the image to grayscale
    img = np.array(image.convert('L'))
    
    # Load OpenCV's pre-trained Haarcascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if len(faces) > 0:
        # If faces are found, crop the first face detected
        (x, y, w, h) = faces[0]
        cropped_face = img[y:y+h, x:x+w]

        # Resize the cropped face to the size expected by the model
        cropped_face_resized = cv2.resize(cropped_face, (224, 224))
        
        # Normalize and expand dimensions for model prediction
        img_array = np.expand_dims(cropped_face_resized, axis=(0, -1))  # Add batch and channel dimensions
        return img_array, True, cropped_face
    else:
        return None, False, None


def predict_emotion(img_array):
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_emotion = class_labels[np.argmax(score)]

    # Apply smoothing
    smoothed_predictions.append(predicted_emotion)
    smoothed_emotion = max(set(smoothed_predictions), key=smoothed_predictions.count)
    
    return smoothed_emotion


##############################################################################################


# Registration route
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not all(key in data for key in ('username', 'password', 'email', 'date_of_birth')):
        return jsonify({'message': 'Missing required fields'}), 400
    
    username = data['username']
    password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    email = data['email']
    userRole = 'Admin'  # Set the default role, e.g., 'Admin' or 'User'
    
    try:
        date_of_birth = datetime.strptime(data['date_of_birth'], '%Y-%m-%d')
    except ValueError:
        return jsonify({'message': 'Invalid date format'}), 400
    
    age = datetime.now().year - date_of_birth.year - ((datetime.now().month, datetime.now().day) < (date_of_birth.month, date_of_birth.day))
    
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT username, email FROM users WHERE username = %s OR email = %s', (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            return jsonify({'message': 'Username or email already taken'}), 400
        
        # Insert into users table, including userRole
        cursor.execute('INSERT INTO users(username, password, email, date_of_birth, age, userRole) VALUES(%s, %s, %s, %s, %s, %s)', 
                       (username, password, email, date_of_birth, age, userRole))
        mysql.connection.commit()
        cursor.close()
        return jsonify({'message': 'User registered successfully!'})
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500


# Login route
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not all(key in data for key in ('username', 'password')):
        return jsonify({'message': 'Missing required fields'}), 400
    
    username = data['username']
    password = data['password']
    
    try:
        cursor = mysql.connection.cursor()
        # Fetch user credentials and userRole
        cursor.execute('SELECT id, username, password, userRole FROM users WHERE username = %s', [username])
        user = cursor.fetchone()
        cursor.close()
        
        if user and bcrypt.check_password_hash(user[2], password):
            # Create access token (JWT)
            access_token = create_access_token(identity={'username': user[1], 'id': user[0]})
            
            # Generate a session token (can use UUID for uniqueness)
            session_token = str(uuid.uuid4())
            
            # Insert the session into the user_sessions table
            cursor = mysql.connection.cursor()
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, active)
                VALUES (%s, %s, %s)
            ''', (user[0], session_token, True))
            mysql.connection.commit()
            cursor.close()
            
            # Return the access token and userRole
            return jsonify({
                'access_token': access_token,
                'session_token': session_token,  # Include session token in the response
                'userRole': user[3]  # Return userRole ('Admin' or 'User')
            })
        else:
            return jsonify({'message': 'Invalid credentials!'}), 401
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500
    
# Logout route
@app.route('/api/logout', methods=['POST'])
@jwt_required()  # Require JWT authentication to log out
def logout():
    session_token = request.headers.get('Authorization').split("Bearer ")[1]  # Get the session token

    try:
        # Mark the session as inactive
        cursor = mysql.connection.cursor()
        cursor.execute('UPDATE user_sessions SET active = 0 WHERE session_token = %s', [session_token])
        mysql.connection.commit()
        cursor.close()

        return jsonify({'message': 'Logout successful'}), 200
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500


# Get user details route
@app.route('/api/user', methods=['GET'])
@jwt_required()
def get_user_details():
    current_user = get_jwt_identity()
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT username, email, date_of_birth FROM users WHERE id = %s', [current_user['id']])
    user = cursor.fetchone()
    cursor.close()
    
    if user:
        return jsonify({
            'username': user[0],
            'email': user[1],
            'date_of_birth': user[2].strftime('%Y-%m-%d')
        })
    return jsonify({'message': 'User not found'}), 404

# Update user profile route
@app.route('/api/profile/update', methods=['POST'])
@jwt_required()
def update_profile():
    current_user = get_jwt_identity()
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')

    if not all([username, email]):
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        cursor = mysql.connection.cursor()

        # Check if username or email is already taken by another user
        cursor.execute('SELECT id FROM users WHERE (username = %s OR email = %s) AND id != %s', (username, email, current_user['id']))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            return jsonify({'message': 'Username or email already taken'}), 400

        # Update username and email
        cursor.execute('UPDATE users SET username = %s, email = %s WHERE id = %s', (username, email, current_user['id']))
        mysql.connection.commit()

        cursor.close()
        return jsonify({'message': 'Profile updated successfully'})
    except Exception as e:
        # Return full error message for debugging
        return jsonify({'message': f'Failed to update details: {str(e)}'}), 500

# Change password route
@app.route('/api/change-password', methods=['POST'])
@jwt_required()
def change_password():
    data = request.get_json()
    if not all(key in data for key in ('current_password', 'new_password')):
        return jsonify({'message': 'Missing required fields'}), 400
    
    current_password = data['current_password']
    new_password = data['new_password']
    current_user = get_jwt_identity()
    user_id = current_user['id']
    
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT password FROM users WHERE id = %s', [user_id])
        user = cursor.fetchone()
        cursor.close()
        
        # Check if the current password matches
        if user and bcrypt.check_password_hash(user[0], current_password):
            # Hash and update the new password
            hashed_new_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
            cursor = mysql.connection.cursor()
            cursor.execute('UPDATE users SET password = %s WHERE id = %s', (hashed_new_password, user_id))
            mysql.connection.commit()
            cursor.close()
            return jsonify({'message': 'Password changed successfully!'})
        else:
            return jsonify({'message': 'Current password is incorrect'}), 401
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500

# Contact Us route
@app.route('/api/contact', methods=['POST'])
def contact_us():
    data = request.get_json()

    # Ensure all required fields are present
    if not all(key in data for key in ('name', 'email', 'message')):
        return jsonify({'message': 'Missing required fields'}), 400

    name = data['name']
    email = data['email']
    message = data['message']

    try:
        # Store the contact message in the database
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO contact_messages (name, email, message) VALUES (%s, %s, %s)', 
                       (name, email, message))
        mysql.connection.commit()
        cursor.close()

        return jsonify({'message': 'Your message has been received! We will get back to you soon.'}), 200
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500


# Fetch all users route (Admin only)
@app.route('/api/users', methods=['GET'])
@jwt_required()
def get_all_users():
    current_user = get_jwt_identity()
    
    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403
    
    # Fetch all users
    cursor.execute('SELECT id, username, email, userRole, created_at FROM users')
    users = cursor.fetchall()
    cursor.close()

    # Return user data as JSON
    user_list = []
    for user in users:
        user_list.append({
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'userRole': user[3],
            'created_at': user[4].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(user_list), 200

# Fetch unread messages for notifications (Admin only)
@app.route('/api/unread-messages', methods=['GET'])
@jwt_required()
def get_unread_messages():
    current_user = get_jwt_identity()

    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403

    # Fetch unread messages
    cursor.execute('SELECT id, name, email, message, created_at FROM contact_messages WHERE status = "Unread"')
    unread_messages = cursor.fetchall()
    cursor.close()

    return jsonify([
        {
            'id': msg[0],
            'name': msg[1],
            'email': msg[2],
            'message': msg[3],
            'created_at': msg[4].strftime('%Y-%m-%d %H:%M:%S')
        } for msg in unread_messages
    ]), 200


# Fetch all contact messages (Admin only)
@app.route('/api/contact-messages', methods=['GET'])
@jwt_required()
def get_all_contact_messages():
    current_user = get_jwt_identity()

    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403

    # Fetch all contact messages
    cursor.execute('SELECT id, name, email, message, status, created_at FROM contact_messages')
    messages = cursor.fetchall()
    cursor.close()

    return jsonify([
        {
            'id': msg[0],
            'name': msg[1],
            'email': msg[2],
            'message': msg[3],
            'status': msg[4],
            'created_at': msg[5].strftime('%Y-%m-%d %H:%M:%S')
        } for msg in messages
    ]), 200


# Mark a message as read (Admin only)
@app.route('/api/mark-message-read/<int:message_id>', methods=['POST'])
@jwt_required()
def mark_message_read(message_id):
    current_user = get_jwt_identity()

    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403

    # Mark the message as read
    cursor = mysql.connection.cursor()
    cursor.execute('UPDATE contact_messages SET status = "Read" WHERE id = %s', [message_id])
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Message marked as read'}), 200

# Update message status route (Admin only)
@app.route('/api/update-message-status/<int:message_id>', methods=['POST'])
@jwt_required()
def update_message_status(message_id):
    current_user = get_jwt_identity()

    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403

    data = request.get_json()
    new_status = data.get('status')

    # Update the message status in the database
    cursor = mysql.connection.cursor()
    cursor.execute('UPDATE contact_messages SET status = %s WHERE id = %s', (new_status, message_id))
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Message status updated successfully.'}), 200


# Delete message route (Admin only)
@app.route('/api/delete-message/<int:message_id>', methods=['DELETE'])
@jwt_required()
def delete_message(message_id):
    current_user = get_jwt_identity()
    
    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403
    
    # Delete the message from the database
    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM contact_messages WHERE id = %s', [message_id])
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Message deleted successfully'}), 200

# Update user route (Admin only)
@app.route('/api/profile/update/<int:user_id>', methods=['POST'])
@jwt_required()
def update_user(user_id):
    current_user = get_jwt_identity()

    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403

    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    role = data.get('role')

    # Update the user in the database
    cursor = mysql.connection.cursor()
    cursor.execute('UPDATE users SET username = %s, email = %s, userRole = %s WHERE id = %s',
                   (username, email, role, user_id))
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'User updated successfully.'}), 200

# Delete user route (Admin only)
@app.route('/api/delete-user/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    current_user = get_jwt_identity()

    # Ensure the current user has admin rights
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT userRole FROM users WHERE id = %s', [current_user['id']])
    userRole = cursor.fetchone()[0]
    if userRole != 'Admin':
        return jsonify({'message': 'Access denied. Admins only.'}), 403

    # Delete the user from the database
    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM users WHERE id = %s', [user_id])
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'User deleted successfully.'}), 200


# Upload directory for saving files (relative path)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'Uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# API to upload a new resource (POST)
@app.route('/api/upload-resource', methods=['POST'])  # Correct route to upload resource
def upload_new_resource():
    name = request.form['name']
    resource_type = request.form['type']
    category = request.form['category']
    file = request.files['file']

    if file:
        # Sanitize the file name before saving it
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        cursor = mysql.connection.cursor()
        cursor.execute(
            'INSERT INTO resources (name, type, category, file_path) VALUES (%s, %s, %s, %s)',
            (name, resource_type, category, file_path)
        )
        mysql.connection.commit()
        cursor.close()

        return jsonify({'message': 'Resource uploaded successfully'}), 201

# API to get all resources (GET)
@app.route('/api/resources', methods=['GET'])
def get_resources():
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT id, name, type, category, file_path, created_at, disabled FROM resources')
    resources = cursor.fetchall()
    cursor.close()

    return jsonify([
        {
            'id': resource[0],
            'name': resource[1],
            'type': resource[2],
            'category': resource[3],
            'file_path': resource[4],
            'created_at': resource[5].strftime('%Y-%m-%d %H:%M:%S'),
            'disabled': resource[6],
        }
        for resource in resources
    ])

# API to update a resource
@app.route('/api/resources/<int:id>', methods=['PUT'])
def update_resource(id):
    data = request.get_json()
    cursor = mysql.connection.cursor()
    cursor.execute(
        'UPDATE resources SET name = %s, category = %s, disabled = %s WHERE id = %s',
        (data['name'], data['category'], data['disabled'], id)
    )
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Resource updated successfully'})

# API to delete a resource
@app.route('/api/resources/<int:id>', methods=['DELETE'])
def delete_resource(id):
    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM resources WHERE id = %s', [id])
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Resource deleted successfully'}), 200

# API to enable/disable a resource
@app.route('/api/resources/<int:id>', methods=['PUT'])
def toggle_resource(id):
    data = request.get_json()
    cursor = mysql.connection.cursor()
    cursor.execute('UPDATE resources SET disabled = %s WHERE id = %s', (data['disabled'], id))
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Resource status updated successfully'}), 200

# Upload directory for saving model files (relative path)
MODEL_UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'ModelUploads')
app.config['MODEL_UPLOAD_FOLDER'] = MODEL_UPLOAD_FOLDER

# API to upload a new model
@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    name = request.form['name']
    comments = request.form['comments']
    file = request.files['file']

    if file and file.filename.endswith('.h5'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['MODEL_UPLOAD_FOLDER'], filename)
        file.save(file_path)

        cursor = mysql.connection.cursor()
        cursor.execute(
            'INSERT INTO models (name, file_path, comments) VALUES (%s, %s, %s)',
            (name, file_path, comments)
        )
        mysql.connection.commit()
        cursor.close()

        return jsonify({'message': 'Model uploaded successfully'}), 201
    else:
        return jsonify({'error': 'Invalid file format, only .h5 files are allowed.'}), 400

# API to get all models
@app.route('/api/models', methods=['GET'])
def get_models():
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT id, name, file_path, comments, created_at, disabled FROM models')
    models = cursor.fetchall()
    cursor.close()

    return jsonify([
        {
            'id': model[0],
            'name': model[1],
            'file_path': model[2],
            'comments': model[3],
            'created_at': model[4].strftime('%Y-%m-%d %H:%M:%S'),
            'disabled': model[5]
        }
        for model in models
    ])

# API to disable/enable a model
@app.route('/api/models/<int:id>', methods=['PUT'])
def toggle_model(id):
    data = request.get_json()
    cursor = mysql.connection.cursor()
    cursor.execute(
        'UPDATE models SET disabled = %s WHERE id = %s',
        (data['disabled'], id)
    )
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Model status updated successfully'}), 200

# POST API to create or update an emotion setting
@app.route('/api/emotion-settings', methods=['POST'])
@jwt_required()
def add_or_update_emotion_setting():
    data = request.get_json()
    cursor = mysql.connection.cursor()

    # Create or update the emotion setting
    if 'id' in data:
        # Update existing emotion setting
        cursor.execute('''
            UPDATE emotion_settings 
            SET emotion = %s, wallpaper_command = %s, music_command = %s, lighting_command = %s, comment = %s
            WHERE id = %s
        ''', (data['emotion'], data['wallpaper_command'], data['music_command'], data['lighting_command'], data['comment'], data['id']))
        emotion_setting_id = data['id']
    else:
        # Insert new emotion setting
        cursor.execute('''
            INSERT INTO emotion_settings (emotion, wallpaper_command, music_command, lighting_command, comment)
            VALUES (%s, %s, %s, %s, %s)
        ''', (data['emotion'], data['wallpaper_command'], data['music_command'], data['lighting_command'], data['comment']))
        emotion_setting_id = cursor.lastrowid

    # Clear existing resources for this emotion setting
    cursor.execute('DELETE FROM emotion_resources WHERE emotion_setting_id = %s', [emotion_setting_id])

    # Insert selected resources (music, images, videos)
    for resource_id in data['resource_ids']:
        cursor.execute('INSERT INTO emotion_resources (emotion_setting_id, resource_id) VALUES (%s, %s)', (emotion_setting_id, resource_id))

    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Emotion setting saved successfully'}), 201

# GET API to fetch all emotion settings with linked resources
@app.route('/api/emotion-settings', methods=['GET'])
@jwt_required()
def get_emotion_settings():
    cursor = mysql.connection.cursor()

    # Fetch emotion settings
    cursor.execute('SELECT * FROM emotion_settings')
    emotion_settings = cursor.fetchall()

    # Fetch associated resources for each emotion setting
    result = []
    for setting in emotion_settings:
        cursor.execute('''
            SELECT r.id, r.name, r.type, r.category, r.file_path
            FROM resources r 
            JOIN emotion_resources er ON er.resource_id = r.id
            WHERE er.emotion_setting_id = %s
        ''', [setting[0]])  # setting[0] is the id of the emotion setting
        resources = cursor.fetchall()
        result.append({
            'id': setting[0],
            'emotion': setting[1],
            'wallpaper_command': setting[2],
            'music_command': setting[3],
            'lighting_command': setting[4],
            'comment': setting[5],
            'resources': [{'id': res[0], 'name': res[1], 'type': res[2], 'category': res[3], 'file_path': res[4]} for res in resources]
        })

    cursor.close()
    return jsonify(result)

# DELETE API to delete an emotion setting
@app.route('/api/emotion-settings/<int:id>', methods=['DELETE'])
@jwt_required()
def delete_emotion_setting(id):
    cursor = mysql.connection.cursor()

    # Delete emotion setting and associated resources
    cursor.execute('DELETE FROM emotion_settings WHERE id = %s', [id])
    cursor.execute('DELETE FROM emotion_resources WHERE emotion_setting_id = %s', [id])

    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Emotion setting deleted successfully'}), 200

# Store the time when the app starts
start_time = time.time()

# Setup logging
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'system.log'),
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Get system uptime
def get_system_uptime():
    uptime_seconds = time.time() - start_time
    uptime_string = str(datetime.fromtimestamp(uptime_seconds) - datetime(1970, 1, 1))
    return uptime_string

# API to fetch system logs
@app.route('/api/logs', methods=['GET'])
@jwt_required()
def get_logs():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'system.log'), 'r') as log_file:
            logs = log_file.readlines()
        return jsonify(logs), 200
    except Exception as e:
        logging.error(f"Error fetching logs: {str(e)}")
        return jsonify({'message': 'Error fetching logs'}), 500

# API to clear system logs
@app.route('/api/clear-logs', methods=['POST'])
@jwt_required()
def clear_logs():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'system.log'), 'w') as log_file:
            log_file.truncate(0)  # Clear the log file
        logging.info("System logs cleared by admin.")
        return jsonify({'message': 'Logs cleared successfully'}), 200
    except Exception as e:
        logging.error(f"Error clearing logs: {str(e)}")
        return jsonify({'message': 'Error clearing logs'}), 500

# Example API that logs an action
@app.route('/api/some-action', methods=['POST'])
@jwt_required()
def some_action():
    try:
        # Simulate some action
        logging.info(f"Admin triggered some action at {datetime.now()}")
        return jsonify({'message': 'Action completed successfully'}), 200
    except Exception as e:
        logging.error(f"Error in some action: {str(e)}")
        return jsonify({'message': 'Error performing action'}), 500

# API to fetch analytics data
@app.route('/api/admin/analytics', methods=['GET'])
@jwt_required()
def get_analytics():
    cursor = mysql.connection.cursor()

    # Get total users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Get active sessions
    cursor.execute("SELECT COUNT(*) FROM user_sessions WHERE active = 1")
    active_sessions = cursor.fetchone()[0]

    # Get system uptime in a readable format
    system_uptime = get_system_uptime()

    # Count the number of errors in the log file
    log_file_path = os.path.join(os.path.dirname(__file__), 'system.log')
    error_count = 0
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                if 'ERROR' in line:
                    error_count += 1

    cursor.close()

    return jsonify({
        'total_users': total_users,
        'active_sessions': active_sessions,
        'system_uptime': system_uptime,
        'errors_logged': error_count
    }), 200





if __name__ == '__main__':
     # Create the Uploads directory if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(MODEL_UPLOAD_FOLDER):
        os.makedirs(MODEL_UPLOAD_FOLDER)
    app.run(debug=True)
