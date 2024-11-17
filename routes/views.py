from flask import Blueprint, render_template

views = Blueprint('views', __name__)

def init_views(waste_data):
    @views.route('/')
    def index():
        materials = waste_data.get_all_materials()
        return render_template('index.html', materials=materials)
    
    return views