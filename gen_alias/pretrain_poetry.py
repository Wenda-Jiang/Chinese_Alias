# coding=utf-8
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train():
  f = open('data/shici/all_shici').readlines()

  lines = [filter(lambda a: len(a) > 0, list(line.strip().decode('utf-8'))) for line in f]
  from collections import defaultdict
  vocab = defaultdict(int)
  for line in lines:
    for word in line:
      vocab[word] += 1

  for term, count in vocab.items():
    print term, count

  model = Word2Vec(lines, size=100, window=2, min_count=3, workers=5, iter=4)
  model.save_word2vec_format('model/word_vec', binary=False)

def gen_word_id():
  f = open('model/word_vec').readlines()
  words = [line.strip().decode('utf-8').split()[0] for line in f]
  f = open('data/shici/name_zi.txt').readlines()
  for line in f:
    words.extend(list(line.strip().decode('utf-8')))

  words = set(words)
  words.remove(' ')

  f = open('model/word_id', 'w')
  for word, index in zip(list(words), range(len(words))):
    f.write('%s %d\n' % (word.encode('utf-8'), index + 2))


def test():
  word2vec = gensim.models.Word2Vec.load_word2vec_format('model/word_vec', binary=False)
  word2vec.init_sims(replace=True)
  for term, weight in word2vec.most_similar('君'.decode('utf-8')):
    print term, weight

gen_word_id()