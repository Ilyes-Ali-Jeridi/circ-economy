from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    materials = db.relationship('MaterialNeed', backref='company', lazy=True)
    wastes = db.relationship('WasteProduction', backref='company', lazy=True)

class MaterialNeed(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    material_type = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Float, nullable=False)  # tons/month
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class WasteProduction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    waste_type = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Float, nullable=False)  # tons/month
    created_at = db.Column(db.DateTime, default=datetime.utcnow)