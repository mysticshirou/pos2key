from flask import Flask, send_from_directory
from pos2key.subway_surfers_interface import SubwaySurfer

app = Flask(__name__, static_folder="vue_dist", static_url_path="")
subway_surfer = SubwaySurfer(flask_app=app)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# Example API route
@app.route("/api/hello")
def hello():
    return {"message": "Hello from Flask!"}

if __name__ == "__main__":
    subway_surfer.run(app, host='0.0.0.0', port=5000)