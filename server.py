import os
from flask import Flask, request, jsonify
from flask_cors import CORS

import service

app = Flask(__name__)
CORS(app)

@app.route('/search', methods=['GET'])
def search():
  query = request.args.get('query')
  result = service.search(query)
  return jsonify(result)

@app.route('/doc/<id>', methods=['GET'])
def get_by_id(id):
  result = service.get_by_id(int(id))
  return jsonify(result)

@app.route('/', methods=['GET'])
def home():
  return "<h1>Search Engine</h1>"

if __name__ == '__main__':
  host = os.environ.get('HOST', '0.0.0.0')
  port = os.environ.get('PORT', 5000)
  debug = eval(os.environ.get('DEBUG', "True"))
  app.run(host=host, port=port, debug=debug)

