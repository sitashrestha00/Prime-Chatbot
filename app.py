import re
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import json
import random
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user, login_required
from flask_bcrypt import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length
from flask_bcrypt import Bcrypt
from login import LoginForm, SignupForm

# Initialize the Flask app
app = Flask(__name__)

# Configure app
app.config['SECRET_KEY'] = "de9e5b220476ba0aba47040eb9b2fea9"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'

# Initialize extensions
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = "info"
db = SQLAlchemy(app)

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(20), nullable=False)
    lastname = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(128), nullable=False)

    def __repr__(self):
        return f"User({self.id}, {self.firstname}, {self.email})"

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=True)
            return redirect(url_for('home'))

        flash('Login unsuccessful. Please check email and password.', 'danger')

    return render_template('login.html', form=form)

# Signup route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('signup.html', form=form)

        hashed_password = generate_password_hash(form.password.data)
        new_user = User(
            firstname=form.firstname.data,
            lastname=form.lastname.data,
            email=form.email.data,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html', form=form)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load pre-trained chatbot model
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
bot_name = "Prime_Bot"

lemmatizer = WordNetLemmatizer()

def cleaning(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[-()\"#/@;:<>=|.?,]", "", sentence)
    sentence_words = nltk.word_tokenize(sentence)
    return sentence_words

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I don't understand."
@app.route('/')
@app.route('/home')
def home():
    return render_template('landing_page.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    message = request.json['message']
    ints = predict_class(message)
    response = get_response(ints, intents)
    return jsonify({'response': response})

@app.route("/user_logout", methods=["POST"])
@login_required
def user_logout():
    logout_user()
    return jsonify(success=True)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
