from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

db = SQLAlchemy()

class Component(db.Model):
    __tablename__ = 'components'
    
    id = db.Column(db.Integer, primary_key=True)
    component_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    shape = db.Column(db.String(100))
    node_id = db.Column(db.String(50))
    quantity = db.Column(db.Integer, default=1)
    estimated_cost = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'component_id': self.component_id,
            'name': self.name,
            'shape': self.shape,
            'node_id': self.node_id,
            'quantity': self.quantity,
            'estimated_cost': self.estimated_cost,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

def init_db(app):
    # Get database URL from environment variable or use SQLite for local development
    database_url = os.getenv('DATABASE_URL', 'sqlite:///local.db')
    
    # Configure SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize the database
    db.init_app(app)
    
    # Create tables
    with app.app_context():
        db.create_all()

def push_to_db(data):
    """Push data to PostgreSQL database"""
    try:
        for item in data:
            component = Component(
                component_id=item.get('component_id'),
                name=item.get('name'),
                shape=item.get('shape'),
                node_id=item.get('node_id'),
                quantity=item.get('quantity', 1),
                estimated_cost=item.get('estimated_cost', 0.0)
            )
            db.session.merge(component)  # Use merge instead of add to handle updates
        
        db.session.commit()
        return True, f"Successfully pushed {len(data)} components to database"
    except Exception as e:
        db.session.rollback()
        return False, str(e)

def get_all_components():
    """Get all components from database"""
    try:
        components = Component.query.all()
        return [component.to_dict() for component in components]
    except Exception as e:
        return [] 