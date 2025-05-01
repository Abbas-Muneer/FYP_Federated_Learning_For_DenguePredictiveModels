from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String, nullable=False)
    dataset_name = db.Column(db.String, nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)
    contribution_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=db.func.now())

class FLConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String, nullable=False, unique=True)
    value = db.Column(db.String, nullable=False)

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)
    password = db.Column(db.String, nullable=False) 
    is_admin = db.Column(db.Boolean, default=False)
    joined_date = db.Column(db.DateTime, default=db.func.now())
