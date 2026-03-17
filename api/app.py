# api/app.py

from flask import Flask, request, jsonify
from core.search import search

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search_api():
    file = request.files["image"]
    path = "temp.jpg"
    file.save(path)

    results = search(path)

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)