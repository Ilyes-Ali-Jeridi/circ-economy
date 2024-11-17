from flask import Blueprint, request, jsonify
from services.matcher import WasteMatcher

api = Blueprint('api', __name__)

def init_api(waste_data):
    matcher = WasteMatcher(waste_data)
    
    @api.route('/find-matches', methods=['POST'])
    def find_matches():
        data = request.json
        matches = matcher.find_matches(data)
        return jsonify(matches)
    
    return api