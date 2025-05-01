from flask_backend.app import app, db

with app.app_context():
    db.create_all()
    print("New database created successfully!")