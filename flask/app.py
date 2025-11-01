from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="vue_dist", static_url_path="")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# Example API route
@app.route("/api/hello")
def hello():
    return {"message": "Hello from Flask!"}

if __name__ == "__main__":
    app.run(debug=True)
