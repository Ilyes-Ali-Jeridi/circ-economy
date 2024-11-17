from flask import Blueprint, render_template, request, jsonify
from models.database import db, Company, MaterialNeed, WasteProduction

dashboard = Blueprint('dashboard', __name__)

@dashboard.route('/dashboard')
def show_dashboard():
    companies = Company.query.all()
    return render_template('dashboard.html', companies=companies)

@dashboard.route('/api/companies', methods=['POST'])
def add_company():
    data = request.json
    company = Company(
        name=data['name'],
        latitude=data['latitude'],
        longitude=data['longitude']
    )
    db.session.add(company)
    db.session.commit()
    return jsonify({'id': company.id}), 201

@dashboard.route('/api/companies/<int:company_id>/materials', methods=['POST'])
def add_material_need(company_id):
    data = request.json
    material = MaterialNeed(
        company_id=company_id,
        material_type=data['material_type'],
        quantity=data['quantity']
    )
    db.session.add(material)
    db.session.commit()
    return jsonify({'id': material.id}), 201

@dashboard.route('/api/companies/<int:company_id>/wastes', methods=['POST'])
def add_waste(company_id):
    data = request.json
    waste = WasteProduction(
        company_id=company_id,
        waste_type=data['waste_type'],
        quantity=data['quantity']
    )
    db.session.add(waste)
    db.session.commit()
    return jsonify({'id': waste.id}), 201

@dashboard.route('/api/companies/<int:company_id>', methods=['GET'])
def get_company_details(company_id):
    company = Company.query.get_or_404(company_id)
    return jsonify({
        'id': company.id,
        'name': company.name,
        'latitude': company.latitude,
        'longitude': company.longitude,
        'materials': [{
            'id': m.id,
            'type': m.material_type,
            'quantity': m.quantity
        } for m in company.materials],
        'wastes': [{
            'id': w.id,
            'type': w.waste_type,
            'quantity': w.quantity
        } for w in company.wastes]
    })