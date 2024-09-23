# app/__init__.py

from flask import Flask, session
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os

load_dotenv()

socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    socketio.init_app(app)
    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    return app