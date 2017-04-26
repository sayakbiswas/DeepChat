#!/usr/bin/python3

import numpy as np
import nltk
import os
import ast
import pickle
import tensorflow as tf
import random
import configparser
from tqdm import tqdm
import string
import argparse
import math
import sys

def getWordID(word, shouldAddToDict=True):
	word = word.lower()
	wordID = wordIDMap.get(word, -1)
	if wordID == -1:
		if shouldAddToDict:
			wordID = len(wordIDMap)
			wordIDMap[word] = wordID
			IDWordMap[wordID] = word
		else:
			wordID = unknownToken
	return wordID

def getWordsFromLine(line, isReply=False):
	'''Returns the word IDs from the vovabulary'''
	words = []
	sentences = nltk.sent_tokenize(line)
	#print(sentences)
	# Since we are limited by a maxmimum length of sentences, we keep the last lines if the statement is a question/input
	# and the first few lines if the statement is an answer/reply
	for i in range(len(sentences)):
		if not isReply:
			i = len(sentences) - 1 - i
		tokensFromCurrSent = nltk.word_tokenize(sentences[i])
		#print(tokensFromCurrSent)
		if len(words) + len(tokensFromCurrSent) > sentMaxLength:
			break
		else:
			temp = []
			for token in tokensFromCurrSent:
				temp.append(getWordID(token))
			if isReply:
				words = words + temp
			else:
				words = temp + words # Append in the reverse order because we're considering the last few lines
	return words

def make_lstm_cell():
	encoderDecoderCell = tf.contrib.rnn.BasicLSTMCell(cellUnitCount)
	encoderDecoderCell = tf.contrib.rnn.DropoutWrapper(encoderDecoderCell, input_keep_prob=1.0, output_keep_prob=dropout)
	return encoderDecoderCell

def generateNextSample():
	for i in range(0, len(trainingSamples), batchSize):
		yield trainingSamples[i:min(i + batchSize, len(trainingSamples))]

class Batch:
	def __init__(self):
		self.encoderSeqs = []
		self.decoderSeqs = []
		self.targetSeqs = []
		self.weights = []

def saveModel(saver, sess, isDone=False):
	if globalStep - 10 > 0 and not isDone:
		os.remove(os.path.join(cwd, 'saved_model/model_' + str(globalStep - 10) + '.ckpt'))
		os.remove(os.path.join(cwd, 'saved_model/model_' + str(globalStep - 10) + '.ckpt.meta'))

	model_name = 'model_' + str(globalStep) + '.ckpt'

	if isDone == True:
		model_name = 'model.ckpt'
	print('Saving model checkpoint...{}'.format(model_name))
	saver.save(sess, os.path.join(cwd, 'saved_model', model_name))
	print('Done')

def makeBatch(samples):
	batch = Batch()
	batchSize = len(samples)
	for i in range(batchSize):
		sample = samples[i]
		#print(sample)
		batch.encoderSeqs.append(list(reversed(sample[0])))
		batch.decoderSeqs.append([goToken] + sample[1] + [eosToken])
		batch.targetSeqs.append(batch.decoderSeqs[-1][1:])

		batch.encoderSeqs[i] = [padToken] * (encoderMaxLength - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]
		batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (decoderMaxLength - len(batch.targetSeqs[i])))
		batch.decoderSeqs[i] = batch.decoderSeqs[i] + [padToken] * (decoderMaxLength - len(batch.decoderSeqs[i]))
		batch.targetSeqs[i] = batch.targetSeqs[i] + [padToken] * (decoderMaxLength - len(batch.targetSeqs[i]))

	encoderSeqListT = []
	for i in range(encoderMaxLength):
		encoderSeqT = []
		for j in range(batchSize):
			encoderSeqT.append(batch.encoderSeqs[j][i])
		encoderSeqListT.append(encoderSeqT)
	batch.encoderSeqs = encoderSeqListT

	decoderSeqListT = []
	targetSeqListT = []
	weightListT = []
	for i in range(decoderMaxLength):
		decoderSeqT = []
		targetSeqT = []
		weightT = []
		for j in range(batchSize):
			#print('j: {}, i:{}'.format(j,i))
			decoderSeqT.append(batch.decoderSeqs[j][i])
			targetSeqT.append(batch.targetSeqs[j][i])
			weightT.append(batch.weights[j][i])
		decoderSeqListT.append(decoderSeqT)
		targetSeqListT.append(targetSeqT)
		weightListT.append(weightT)
	batch.decoderSeqs = decoderSeqListT
	batch.targetSeqs = targetSeqListT
	batch.weights = weightListT
	return batch

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--predict', default=None, help='start in predict mode')
	parser.add_argument('--log', nargs='?', default='chatbot.log', help='Log all stdouts to a file')
	args = parser.parse_args()

	global globalStep
	global goToken
	global eosToken
	global padToken
	global unknownToken
	global trainingSamples
	global wordIDMap
	global IDWordMap

	if args.log and not args.predict:
		print('Logging all output to {}'.format(args.log))
		old_stdout = sys.stdout
		log_file = open(os.path.join(cwd, 'logs', args.log), 'w')
		sys.stdout = log_file

	if not args.predict:
		with open(os.path.join(corpusDir, 'movie_lines.txt'), 'r', encoding='iso-8859-1') as f:
			for line in f:
				#print(line)
				fields = line.split(' +++$+++ ')
				#print(fields)
				obj = {}
				obj['lineID'] = fields[0]
				obj['characterID'] = fields[1]
				obj['movieID'] = fields[2]
				obj['characterName'] = fields[3]
				obj['text'] = fields[4]
				lines[fields[0]] = obj
		#print(lines)

		with open(os.path.join(corpusDir, 'movie_conversations.txt'), 'r', encoding='iso-8859-1') as f:
			for line in f:
				#print(line)
				fields = line.split(' +++$+++ ')
				#print(fields)
				obj = {}
				obj['character1ID'] = fields[0]
				obj['character2ID'] = fields[1]
				obj['movieID'] = fields[2]
				#obj['lineIDs'] = fields[3]
				#print(obj)
				lineIDs = ast.literal_eval(fields[3])
				#print(lineIDs)
				obj['lineIDs'] = lineIDs
				#print(obj)
				obj['lines'] = []
				for lineID in lineIDs:
					#print(lineID, "--", lines[lineID])
					obj['lines'].append(lines[lineID])
				conversations.append(obj)
		#print(conversations)

		padToken = getWordID('<pad>')
		unknownToken = getWordID('<unknown>')
		eosToken = getWordID('<eos>')
		goToken = getWordID('<go>')
		for conversation in conversations:
			#print(conversation)
			for i in range(len(conversation['lines']) - 1):
				#print(conversation['lines'][i])
				inputStatement = conversation['lines'][i]
				#print(inputStatement)
				replyStatement = conversation['lines'][i + 1]
				inputWords = getWordsFromLine(inputStatement['text'])
				replyWords = getWordsFromLine(replyStatement['text'], True)
				#print(inputWords)
				#print(replyWords)

				if inputWords and replyWords:
					trainingSamples.append([inputWords, replyWords])
		#print(trainingSamples)

		print("Saving dataset samples ...")
		with open(os.path.join(cwd, 'data/samples', 'sampleData.pkl'), 'wb') as f:
			data = {
				'wordIDMap': wordIDMap,
				'IDWordMap': IDWordMap,
				'trainingSamples': trainingSamples
			}
			pickle.dump(data, f, -1)
		print('Done')
	else:
		padToken = getWordID('<pad>')
		unknownToken = getWordID('<unknown>')
		eosToken = getWordID('<eos>')
		goToken = getWordID('<go>')
		print("Loading dataset samples ...")
		with open(os.path.join(cwd, 'data/samples', 'sampleData.pkl'), 'rb') as f:
			data = pickle.load(f)
			wordIDMap = data['wordIDMap']
			IDWordMap = data['IDWordMap']
			trainingSamples = data['trainingSamples']
		print('Done')

	with tf.device('/gpu:0'):
		#Expand the list comprehension below
		encoderDecoderCell = tf.contrib.rnn.MultiRNNCell(
			[make_lstm_cell() for _ in range(numOfLayers)],
		)

		with tf.name_scope('encoder'):
			encoderInputs = [tf.placeholder(tf.int32, [None, ]) for _ in range(sentMaxLength)]
		with tf.name_scope('decoder'):
			decoderInputs = [tf.placeholder(tf.int32, [None, ], name="inputs") for _ in range(sentMaxLength + 2)]
			decoderTargets = [tf.placeholder(tf.int32, [None, ], name="targets") for _ in range(sentMaxLength + 2)]
			decoderWeights = [tf.placeholder(tf.float32, [None, ], name="weights") for _ in range(sentMaxLength + 2)]

		#Verify this - is different from the existing
		decoderOutput, state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
			encoderInputs,
			decoderInputs,
			encoderDecoderCell,
			len(wordIDMap),
			len(wordIDMap),
			embeddingSize,
			output_projection=None,
			feed_previous=bool(args.predict)
		)

		lossFunc = tf.contrib.legacy_seq2seq.sequence_loss(
			decoderOutput,
			decoderTargets,
			decoderWeights,
			len(wordIDMap),
			softmax_loss_function=None
		)
		tf.summary.scalar('loss', lossFunc)

		optimizer = tf.train.AdamOptimizer(
			learning_rate=learningRate,
			beta1=0.9,
			beta2=0.999,
			epsilon=1e-08
		)
		optimizationOperation = optimizer.minimize(lossFunc)

	writer = tf.summary.FileWriter('seq2seq')
	saver = tf.train.Saver(max_to_keep=200, write_version=tf.train.SaverDef.V1)
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	#config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())

	#Change variable scope name
	with tf.variable_scope("embedding_rnn_seq2seq/rnn/embedding_wrapper", reuse=True):
		in_embedding = tf.get_variable("embedding")
	with tf.variable_scope("embedding_rnn_seq2seq/embedding_rnn_decoder", reuse=True):
		out_embedding = tf.get_variable("embedding")

	embedding_vars = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
	embedding_vars.remove(in_embedding)
	embedding_vars.remove(out_embedding)

	with open(os.path.join(cwd, 'data/word2vec/GoogleNews-vectors-negative300.bin'), "rb", 0) as f:
		header = f.readline().split()
		#print(header)
		vocabulary_size = int(header[0])
		word_vector_size = int(header[1])
		#print('{}, {}'.format(vocabulary_size, word_vector_size))
		binary_length = np.dtype('float32').itemsize * word_vector_size
		#print(binary_length)
		initial_weights = np.random.uniform(-0.25, 0.25, (len(wordIDMap), word_vector_size))
		#print(initial_weights)
		for line in range(word_vector_size):
			word = []
			while True:
				ch = f.read(1)
				if ch == b' ':
					word = b''.join(word).decode('utf-8')
					break
				if ch != b'\n':
					word.append(ch)
			if word in wordIDMap:
				initial_weights[wordIDMap[word]] = np.fromstring(f.read(binary_length), dtype='float32')
			else:
				f.read(binary_length)

	if embeddingSize < word_vector_size:
		u, s, vt = np.linalg.svd(initial_weights, full_matrices=False)
		S = np.zeros((word_vector_size, word_vector_size), dtype=complex)
		S[:word_vector_size, :word_vector_size] = np.diag(s)
		initial_weights = np.dot(u[:, :embeddingSize], S[:embeddingSize, :embeddingSize])

	sess.run(in_embedding.assign(initial_weights))
	sess.run(out_embedding.assign(initial_weights))

	if args.predict:
		saved_model_dir = 'saved_model'
		model_name = 'model.ckpt'
		if os.path.exists(os.path.join(cwd, saved_model_dir, model_name)):
			print('Restoring model {}'.format(model_name))
			saver.restore(sess, os.path.join(cwd, saved_model_dir, model_name))
			print('Welcome to DeepChat! I am Alex. You can ask me questions and ponder on the answers I provide.')
			print('Type \'exit\' to end the chat.')
			while True:
				user_input = input('You: ')
				if user_input == '':
					print('Alex: Please say something! I don\'t like silence!')
				if user_input == 'exit':
					break
				inputSequence = []
				tokens = nltk.word_tokenize(user_input)
				if len(tokens) > sentMaxLength:
					print('I didn\'t understand! Please try a smaller sentence')
					continue
				wordIDs = []
				for token in tokens:
					wordIDs.append(getWordID(token, shouldAddToDict=False))
				batch = makeBatch([[wordIDs,[]]])
				inputSequence.extend(batch.encoderSeqs)
				feedDict = {}
				ops = None
				for i in range(encoderMaxLength):
					feedDict[encoderInputs[i]] = batch.encoderSeqs[i]
				feedDict[decoderInputs[0]] = [goToken]
				#print('decoderOutput {}'.format(decoderOutput))
				ops = (decoderOutput,)
				outputs = sess.run(ops[0], feedDict)
				outputSequence = []
				for output in outputs:
					outputSequence.append(np.argmax(output))
				#print('outputSequence {}'.format(outputSequence))
				responseTokens = []
				for wordID in outputSequence:
					if wordID == eosToken:
						break
					elif wordID != padToken and wordID != goToken:
						responseTokens.append(IDWordMap[wordID])
				#print('responseTokens {}'.format(responseTokens))
				response = ""
				responseTokens = [t.replace(t, ' ' + t) if not t.startswith('\'') and t not in string.punctuation else t for t in responseTokens]
				#print('responseTokens after replace {}'.format(responseTokens))
				response = ''.join(responseTokens).strip().capitalize()
				print('Alex: ' + response)
				print()
		else:
			print('Error: Model not found! Check the saved_model folder.')
	else:
		# Training Loop
		completeSummary = tf.summary.merge_all()
		if globalStep == 0:
			writer.add_graph(sess.graph)
		try:
			for epoch in range(numOfEpochs):
				print("\nEpoch {}".format(epoch+1))
				random.shuffle(trainingSamples)

				batches = []
				for samples in generateNextSample():
					batch = makeBatch(samples)
					batches.append(batch)

				for batch in tqdm(batches, desc="Training"):
					feedDict = {}
					ops = None
					for i in range(encoderMaxLength):
						feedDict[encoderInputs[i]] = batch.encoderSeqs[i]
					for i in range(decoderMaxLength):
						feedDict[decoderInputs[i]] = batch.decoderSeqs[i]
						feedDict[decoderTargets[i]] = batch.targetSeqs[i]
						feedDict[decoderWeights[i]] = batch.weights[i]
					ops = (optimizationOperation, lossFunc)
					assert len(ops) == 2
					#print(feedDict)
					_, loss, summary = sess.run(ops + (completeSummary,), feedDict)
					writer.add_summary(summary, globalStep)
					globalStep += 1
					if globalStep % 100 == 0:
						perplexity = math.exp(float(loss))
						print("Step %d " % (globalStep))
						print("Loss %.2f" % (loss))
						print("Perplexity %.2f" % (perplexity))
					if globalStep % 10 == 0:
						saveModel(saver, sess)
		except (KeyboardInterrupt, SystemExit):
			print('Saving and Exiting...')
		saveModel(saver, sess, isDone=True)
		sess.close()
		if args.log and not args.predict:
			sys.stdout = old_stdout
			log_file.close()

if __name__ == "__main__":
	cwd = os.getcwd()
	#print(cwd)
	corpusDir = os.path.join(cwd, 'data/cornell')
	#print(corpusDir)

	lines = {}
	conversations = []

	wordIDMap = {}
	IDWordMap = {}
	unknownToken = -1
	trainingSamples = []
	goToken = -1
	eosToken = -1
	padToken = -1
	sentMaxLength = 10 #maximum length of an input or output sentence
	encoderMaxLength = sentMaxLength
	decoderMaxLength = sentMaxLength + 2

	#Parameters
	globalStep = 0
	cellUnitCount = 512
	numOfLayers = 2
	embeddingSize = 64
	learningRate = 0.02
	batchSize = 256
	dropout = 0.9
	softmaxSamples = 0
	numOfEpochs = 30

	main()
