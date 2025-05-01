from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_backend.models import db
from flask_cors import CORS


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key' 

db.init_app(app)  # Ensure this is only called once
migrate = Migrate(app, db)

# Blueprint imports here
from flask_backend.routes.client_routes import client_blueprint
from flask_backend.routes.admin_routes import admin_blueprint

app.register_blueprint(client_blueprint)
app.register_blueprint(admin_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
