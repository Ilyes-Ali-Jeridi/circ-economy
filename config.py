import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345'
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///waste_management.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False