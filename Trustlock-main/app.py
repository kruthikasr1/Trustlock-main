import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import train_faces as knn_train
import cv2
import face_recognition
import numpy as np
import pandas as pd
import pickle
import joblib
import re
import sys
import test as spam_detect
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Get the base directory
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
# Use absolute path for database
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "database.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)  # Added username for face recognition
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    emails_sent = db.relationship('Email', foreign_keys='Email.sender_id', backref='sender_user', lazy=True)
    emails_received = db.relationship('Email', foreign_keys='Email.user_id', backref='recipient_user', lazy=True)
    # New field to track face verification status
    face_verified = db.Column(db.Boolean, default=False)

class Email(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(100), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Link to sender
    recipient = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    is_spam = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Link to receiver
    face_verified = db.Column(db.Boolean, default=False)  # Whether sender was face verified
    # New field to track if spam check was performed
    spam_checked = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc)
            for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def identify_faces(video_capture):
    buf_length = 10
    buf = [[]] * buf_length
    i = 0
    process_this_frame = True
    last_face_names = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="./models/trained_model.clf")
        process_this_frame = not process_this_frame

        face_names = []
        for name, (top, right, bottom, left) in predictions:
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            face_names.append(name)

        last_face_names = face_names or last_face_names
        buf[i] = face_names
        i = (i + 1) % buf_length

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # If any unknown detected, abort early
        if any(n == "unknown" for n in face_names):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return last_face_names

def verify_face_for_username(expected_username):
    """Captures from default cam and confirms if the expected user is detected."""
    cam = cv2.VideoCapture(0)
    try:
        names = identify_faces(cam)
        if not names:
            return False
        detected = names[0].replace(']', '').replace("'", "")
        return detected == expected_username
    finally:
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

# Import spam detection function from test.py
    
    # Fallback spam detection function
def detect_spam(email_text):
    res=spam_detect.detect_spamemail(email_text)
    return res

# Context processor to make variables available to all templates
@app.context_processor
def inject_user_data():
    if current_user.is_authenticated:
        try:
            # Get user's face verification status from session or database
            face_verified = session.get('face_verified', False)
            if not face_verified:
                # Check database if not in session
                user = User.query.get(current_user.id)
                face_verified = user.face_verified if user else False
                session['face_verified'] = face_verified
            
            unread_emails = Email.query.filter_by(
                user_id=current_user.id, 
                is_read=False, 
                is_spam=False
            ).count()
            spam_emails = Email.query.filter_by(
                user_id=current_user.id, 
                is_spam=True
            ).count()
        except:
            unread_emails = 0
            spam_emails = 0
            face_verified = False
    else:
        unread_emails = 0
        spam_emails = 0
        face_verified = False
    
    return dict(
        unread_emails=unread_emails,
        spam_emails=spam_emails,
        face_verified=face_verified
    )



# Original spam detection function (kept as backup)


# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

def TakeImages(name):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        raise Exception("Camera not accessible")

    DIR = f"./Dataset/{name}"
    os.makedirs(DIR, exist_ok=True)

    img_counter = len(os.listdir(DIR))

    while True:
        ret, frame = cam.read()

        # ✅ ALWAYS check ret first
        if not ret or frame is None:
            print("Failed to grab frame from camera")
            break

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            img_name = f"{DIR}/opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

       
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')  # Get username for face recognition
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken!', 'danger')
            return redirect(url_for('register'))
        
        # Fixed: Use correct method for password hashing
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, username=username, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            TakeImages(username)

            knn_train.trainer()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=email).first()
        
        if user and check_password_hash(user.password, password):
            # Face verification step
            cam = cv2.VideoCapture(0)
            esname=identify_faces(cam)
            esname=esname[0]
            
            print("esname",esname)
            print("enter user name==",email)
            if esname==email:
                session['face_verified'] = True
                user.face_verified = email
                login_user(user)
                flash('Face verification successful! Login completed.', 'success')
                return redirect(url_for('dashboard'))
            else:
                login_user(user)
                flash('Face verification failed! You have limited access.', 'warning')
                return redirect(url_for('dashboard'))

        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    # Clear face verification status from session
    session.pop('face_verified', None)
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get face verification status
        face_verified = session.get('face_verified', False)
        
        # Get email statistics
        total_emails = Email.query.filter_by(user_id=current_user.id).count()
        
        # Get recent emails (only show normal emails if face verified)
        if face_verified:
            recent_emails = Email.query.filter_by(
                user_id=current_user.id, 
                is_spam=False
            ).order_by(Email.timestamp.desc()).limit(5).all()
        else:
            # Show empty recent emails for non-verified users
            recent_emails = []
        
        return render_template('dashboard.html', 
                             total_emails=total_emails,
                             recent_emails=recent_emails,
                             face_verified=face_verified)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return render_template('dashboard.html', 
                             total_emails=0,
                             recent_emails=[],
                             face_verified=False)

@app.route('/compose', methods=['GET', 'POST'])
@login_required
def compose():
    if request.method == 'POST':
        # Check if face is already verified from session
        face_verified=session.get('face_verified')
        

        recipient = request.form.get('recipient')
        subject = request.form.get('subject')
        body = request.form.get('body')
        is_spam_res=detect_spam(body)
        is_spam_flag=False
        if is_spam_res=="This is a Normal Email!":
            is_spam_flag=True
        else:
            is_spam_flag=False

        
        # Find recipient user
        recipient_user = User.query.filter_by(email=recipient).first()
        
        if recipient_user:
            if face_verified:
                if is_spam_flag:
                    # Face verified, send directly
                    is_spam = 0 
                    # Create email for recipient
                    new_email = Email( sender=current_user.email,sender_id=current_user.id,recipient=recipient,subject=subject,body=body,is_spam=is_spam,face_verified=True,spam_checked=True,user_id=recipient_user.id )
                    db.session.add(new_email)
                    db.session.commit()
                    flash('Email send successfully', 'success')
                else:
                    is_spam = 1 
                    # Create email for recipient
                    new_email = Email( sender=current_user.email,sender_id=current_user.id,recipient=recipient,subject=subject,body=body,is_spam=is_spam,face_verified=True,spam_checked=True,user_id=recipient_user.id )
                    db.session.add(new_email)
                    db.session.commit()
                    flash('Email send successfully', 'success')

                                
            else:
                is_spam = 1 
                # Create email for recipient
                new_email = Email( sender=current_user.email,sender_id=current_user.id,recipient=recipient,subject=subject,body=body,is_spam=is_spam,face_verified=True,spam_checked=False,user_id=recipient_user.id )
                db.session.add(new_email)
                db.session.commit()
                flash('Email send successfully', 'success')
        
        return redirect(url_for('compose'))
    
    return render_template('compose.html')


@app.route('/face_verify_send', methods=['GET', 'POST'])
@login_required
def face_verify_send():
    if request.method == 'POST':
        recipient = request.form.get('recipient')
        subject = request.form.get('subject')
        body = request.form.get('body')
        
        # Verify face
        face_matched = verify_face_for_username(current_user.username)
        
        if face_matched:
            # Update face verification status
            session['face_verified'] = True
            current_user.face_verified = True
            db.session.commit()
            
            recipient_user = User.query.filter_by(email=recipient).first()
            
            if recipient_user:
                # Check if email is spam using the model
                email_text = subject + ' ' + body
                is_spam = "Spam Email"
                
                # Create email for recipient
                new_email = Email(
                    sender=current_user.email,
                    sender_id=current_user.id,
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    is_spam=is_spam,
                    face_verified=True,
                    spam_checked=True,
                    user_id=recipient_user.id
                )
                
                try:
                    db.session.add(new_email)
                    db.session.commit()
                    flash('Email sent successfully with face verification!', 'success')
                except Exception as e:
                    db.session.rollback()
                    flash(f'Failed to send email: {str(e)}', 'danger')
            else:
                flash('Recipient not found!', 'danger')
            
            return redirect(url_for('compose'))
        else:
            # Face verification failed - send to spam
            recipient_user = User.query.filter_by(email=recipient).first()
            
            if recipient_user:
                new_email = Email(
                    sender=current_user.email,
                    sender_id=current_user.id,
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    is_spam=True,  # Automatically spam if face not verified
                    face_verified=False,
                    spam_checked=True,
                    user_id=recipient_user.id
                )
                
                try:
                    db.session.add(new_email)
                    db.session.commit()
                    flash('Face verification failed! Email sent but marked as SPAM.', 'warning')
                except Exception as e:
                    db.session.rollback()
                    flash(f'Failed to send email: {str(e)}', 'danger')
            
            return redirect(url_for('compose'))
    
    # GET request - show verification page
    recipient = request.args.get('recipient')
    subject = request.args.get('subject')
    body = request.args.get('body')
    
    return render_template('face_verify_send.html', 
                         recipient=recipient,
                         subject=subject,
                         body=body)

@app.route('/inbox')
@login_required
def inbox():
    face_verified = session.get('face_verified', False)
    
    if not face_verified:
        flash('Face verification required to access inbox!', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        page = request.args.get('page', 1, type=int)
        # Only show emails that are not spam AND face verified
        emails = Email.query.filter_by(
            user_id=current_user.id, 
            is_spam=False
        ).filter(Email.face_verified == True).order_by(Email.timestamp.desc()).paginate(page=page, per_page=10)
        
        return render_template('inbox.html', emails=emails)
    except Exception as e:
        flash(f'Error loading inbox: {str(e)}', 'danger')
        return render_template('inbox.html', emails=None)

@app.route('/spam')
@login_required
def spam():
    try:
        page = request.args.get('page', 1, type=int)
        # Show emails that are marked as spam OR not face verified
        emails = Email.query.filter_by(
            user_id=current_user.id
        ).filter(
            (Email.is_spam == True) | (Email.face_verified == False)
        ).order_by(Email.timestamp.desc()).paginate(page=page, per_page=10)
        
        return render_template('spam.html', emails=emails)
    except Exception as e:
        flash(f'Error loading spam folder: {str(e)}', 'danger')
        return render_template('spam.html', emails=None)

@app.route('/verify_spam')
@login_required
def verify_spam():
    """Verify all spam emails using the trained model"""
    try:
        # Get all spam emails for current user
        spam_emails = Email.query.filter_by(
            user_id=current_user.id,
            is_spam=True
        ).all()
        
        moved_to_inbox = 0
        kept_in_spam = 0
        
        for email in spam_emails:
            # Check if spam check was already performed
            if not email.spam_checked:
                # Use the model to check if it's actually spam
                email_text = email.subject + ' ' + email.body
                result = detect_spam(email_text)
                
                if "Normal Email" in result:
                    # Move to inbox
                    email.is_spam = False
                    email.face_verified = True  # Mark as verified since model says it's normal
                    email.spam_checked = True
                    moved_to_inbox += 1
                else:
                    # Keep in spam
                    email.spam_checked = True
                    kept_in_spam += 1
        
        db.session.commit()
        
        if moved_to_inbox > 0 or kept_in_spam > 0:
            flash(f'Spam verification completed! Moved {moved_to_inbox} emails to inbox, kept {kept_in_spam} in spam.', 'success')
        else:
            flash('No unverified spam emails found.', 'info')
            
    except Exception as e:
        flash(f'Error verifying spam: {str(e)}', 'danger')
    
    return redirect(url_for('spam'))

@app.route('/email/<int:email_id>')
@login_required
def view_email(email_id):
    try:
        email = Email.query.get_or_404(email_id)
        
        # Check if email belongs to current user
        if email.user_id != current_user.id:
            flash('Unauthorized access!', 'danger')
            return redirect(url_for('dashboard'))
        
        # Mark as read
        if not email.is_read:
            email.is_read = True
            db.session.commit()
        
        return render_template('view_email.html', email=email)
    except Exception as e:
        flash(f'Error viewing email: {str(e)}', 'danger')
        return redirect(url_for('inbox'))

@app.route('/mark_spam/<int:email_id>')
@login_required
def mark_spam(email_id):
    try:
        email = Email.query.get_or_404(email_id)
        
        if email.user_id == current_user.id:
            email.is_spam = True
            email.spam_checked = True
            db.session.commit()
            flash('Email marked as spam!', 'success')
    
        return redirect(request.referrer or url_for('inbox'))
    except Exception as e:
        flash(f'Error marking as spam: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('inbox'))

@app.route('/mark_not_spam/<int:email_id>')
@login_required
def mark_not_spam(email_id):
    try:
        email = Email.query.get_or_404(email_id)
        
        if email.user_id == current_user.id:
            email.is_spam = False
            email.face_verified = True
            email.spam_checked = True
            db.session.commit()
            flash('Email marked as not spam!', 'success')
    
        return redirect(request.referrer or url_for('spam'))
    except Exception as e:
        flash(f'Error marking as not spam: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('spam'))

@app.route('/delete_email/<int:email_id>')
@login_required
def delete_email(email_id):
    try:
        email = Email.query.get_or_404(email_id)
        
        if email.user_id == current_user.id:
            db.session.delete(email)
            db.session.commit()
            flash('Email deleted!', 'success')
    
        return redirect(request.referrer or url_for('inbox'))
    except Exception as e:
        flash(f'Error deleting email: {str(e)}', 'danger')
        return redirect(request.referrer or url_for('inbox'))

# Create some test users and emails for demo
def create_test_data():
    with app.app_context():
        # Check if users exist
        if User.query.count() == 0:
            print("Creating test users...")
            
            # Create test users
            users_data = [
                {'email': 'user1@example.com', 'username': 'user1', 'password': 'password123'},
                {'email': 'user2@example.com', 'username': 'user2', 'password': 'password123'},
                {'email': 'user3@example.com', 'username': 'user3', 'password': 'password123'},
                {'email': 'admin@example.com', 'username': 'admin', 'password': 'admin123'},
            ]
            
            for user_data in users_data:
                hashed_password = generate_password_hash(user_data['password'])
                user = User(email=user_data['email'], username=user_data['username'], password=hashed_password)
                db.session.add(user)
            
            db.session.commit()
            print("Test users created!")
            
            # Create some test emails
            print("Creating test emails...")
            users = User.query.all()
            
            sample_emails = [
                {
                    'sender': 'user2@example.com',
                    'sender_username': 'user2',
                    'subject': 'Meeting Tomorrow',
                    'body': 'Hi, just reminding you about our meeting tomorrow at 10 AM.',
                    'is_spam': False,
                    'face_verified': True
                },
                {
                    'sender': 'user3@example.com',
                    'sender_username': 'user3',
                    'subject': 'WIN A FREE IPHONE!!!',
                    'body': 'Congratulations! You have won a FREE iPhone! Click here to claim: http://free-iphone-scam.com',
                    'is_spam': True,
                    'face_verified': True
                },
                {
                    'sender': 'admin@example.com',
                    'sender_username': 'admin',
                    'subject': 'Important System Update',
                    'body': 'Dear user, please update your system for security patches.',
                    'is_spam': False,
                    'face_verified': True
                },
                {
                    'sender': 'user1@example.com',
                    'sender_username': 'user1',
                    'subject': 'Project Proposal',
                    'body': 'Here is the project proposal we discussed last week.',
                    'is_spam': False,
                    'face_verified': False  # Not face verified - should go to spam
                }
            ]
            
            for email_data in sample_emails:
                # Find sender user
                sender_user = User.query.filter_by(email=email_data['sender']).first()
                if sender_user:
                    # Send to all other users
                    for recipient_user in users:
                        if recipient_user.id != sender_user.id:
                            email = Email(
                                sender=email_data['sender'],
                                sender_id=sender_user.id,
                                recipient=recipient_user.email,
                                subject=email_data['subject'],
                                body=email_data['body'],
                                is_spam=email_data['is_spam'] or not email_data['face_verified'],
                                face_verified=email_data['face_verified'],
                                user_id=recipient_user.id
                            )
                            db.session.add(email)
            
            db.session.commit()
            print("Test emails created!")

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    try:
        with app.app_context():
            if not os.path.exists(os.path.join(basedir, "database.db")):
                print("📦 Creating database for the first time...")
                db.create_all()
                create_test_data()
            else:
                print("✅ Using existing database")
            print("=" * 50)
            print("Email Dashboard Application with Face Verification")
            print("=" * 50)
            print(f"Database: {os.path.join(basedir, 'database.db')}")
            print("Test users created:")
            print("1. user1@example.com / password123 (username: user1)")
            print("2. user2@example.com / password123 (username: user2)")
            print("3. user3@example.com / password123 (username: user3)")
            print("4. admin@example.com / admin123 (username: admin)")
            print("\nImportant Features:")
            print("- Face verification required for full access")
            print("- Face verification required for sending emails")
            print("- Emails from unverified senders automatically go to SPAM")
            print("- Trained spam detection model integrated")
            print("- Spam verification button in spam folder")
            print("\nAccess the application at: http://localhost:5000")
            print("=" * 50)
    except Exception as e:
        print(f"❌ Fatal startup error: {e}")
        raise e
    
    app.run(debug=True, port=5000)
