from flask import Flask, request
from flask_restful import Resource, Api

from sentence_compression import load_model, predict

import os

app = Flask(__name__)
api = Api(app)

word_dict, reversed_dict, model, sess = load_model("model.ckpt-78125")

class About(Resource):
	"""About Us"""
	def get(self):
		return {'about': 'Welcome to Mukhtasar-The only solution to abstractive multi document summarisation'}


class SentenceCompression(Resource):
	"""Sentence Compression"""
	def post(self):
		request_data = request.json

		model_name =  request_data['model_name']
		sentences =  request_data['sentences']
		return {'compressed_sentences': predict(sentences, word_dict, reversed_dict, model, sess)}


api.add_resource(About, '/about')
api.add_resource(SentenceCompression, '/compress-sentences')


if __name__ == "__main__":
	app.run()