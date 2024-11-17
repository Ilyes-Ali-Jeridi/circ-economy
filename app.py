from flask import Flask
from config import Config
from models.waste_data import WasteData
from models.database import db
from routes.api import init_api
from routes.views import init_views
from routes.dashboard import dashboard

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    db.init_app(app)
    with app.app_context():
        db.create_all()
    
    # Initialize data
    waste_data = WasteData()
    
    # Register blueprints
    app.register_blueprint(init_views(waste_data))
    app.register_blueprint(init_api(waste_data), url_prefix='/api')
    app.register_blueprint(dashboard)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)