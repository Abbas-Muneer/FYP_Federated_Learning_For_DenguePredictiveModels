from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_backend.models import db
from flask_cors import CORS


app = Flask(__name__)
app.config['SECRET_KEY'] = "1104"

CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db.init_app(app)  
migrate = Migrate(app, db)


from flask_backend.routes.client_routes import client_blueprint
from flask_backend.routes.admin_routes import admin_blueprint

app.register_blueprint(client_blueprint)
app.register_blueprint(admin_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
