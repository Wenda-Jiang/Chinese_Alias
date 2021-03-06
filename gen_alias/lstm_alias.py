# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
from collections import defaultdict
from tensorflow.python.framework import graph_io
from tensorflow.python.training import saver as saver_mod

term_size = 5000
beam_size = 10
Go_Id = 0
End_Id = 1
UNK_Id = 2


def create_restore_fn(checkpoint_path, saver, sess):
  if tf.gfile.IsDirectory(checkpoint_path):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if not latest_checkpoint:
      return

  tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)

  ckpt = saver_mod.get_checkpoint_state(checkpoint_path)
  while not ckpt or not ckpt.model_checkpoint_path:
    return
  print(ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)
  saver.recover_last_checkpoints([checkpoint_path])
  tf.logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path))


def _bahdanau_score(processed_query, keys, normalize):
  '''
  process attention score
  :param processed_query: singe step hidden state, 2-D tensor [batch_size, num_units]
  :param keys: encoder ouput, 3-D tensor [batch_size, max_times, num_units]
  :param normalize: 
  :return: softmax score, 3-D tensor [batch_size, max_times, num_units]
  '''

  dtype = processed_query.dtype
  num_units = keys.shape[2].value or tf.shape(keys)[2]

  processed_query = tf.expand_dims(processed_query, 1)
  v = tf.get_variable("attention_v", [num_units], dtype=dtype)
  if normalize:
    # Scalar used in weight normalization
    g = tf.get_variable("attention_g", dtype=dtype,
        initializer=tf.math.sqrt((1. / num_units)))
    # Bias added prior to the nonlinearity
    b = tf.get_variable("attention_b", [num_units], dtype=dtype, initializer=tf.zeros_initializer())
    # normed_v = g * v / ||v||
    normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))
    return tf.reduce_sum(normed_v * tf.tanh(keys + processed_query + b), [2])
  else:
    return tf.reduce_sum(v * tf.tanh(keys + processed_query), [2])

def attention(processed_query, keys, normalize):

  score = _bahdanau_score(processed_query, keys, normalize)

  att = tf.reduce_sum(score * keys, axis=1)

  return att


def _extract_argmax(embedding,
                    beam_size,
                    output_projection=True,
                    update_embedding=False):
  def loop_function(prev, _):
    if output_projection is not None:
      prev = tf.layers.dense(prev, term_size)
    values_first, ids_prev = tf.nn.top_k(prev, k=beam_size)
    emb_prev = tf.nn.embedding_lookup(embedding, ids_prev)

    emb_prev = tf.unstack(emb_prev, axis=1, num=beam_size)
    value_prev = tf.unstack(values_first, axis=1, num=beam_size)

    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev, value_prev, ids_prev

  return loop_function


class Alias(object):
  def __init__(self, init_embedding=None, mode='train'):
    print(np.shape(init_embedding))
    self.cell = tf.nn.rnn_cell.LSTMCell(64)
    self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, input_keep_prob=0.8 if mode == 'train' else 1.0)
    self.Go_id = 0
    self.mode = mode
    self.state = None
    self.word_emb = tf.get_variable(name="term_embedding", shape=[term_size, 100], dtype=tf.float32,
                                    initializer=tf.constant_initializer(
                                      init_embedding) if init_embedding is not None else None)

  def build_input(self):

    self.name = tf.placeholder(tf.int32, shape=[None, None], name="name")
    self.name_num = tf.placeholder(tf.int32, shape=[None], name="name_num")
    self.name_emb = tf.nn.embedding_lookup(self.word_emb, self.name)

    if self.mode == 'train':
      self.zi = tf.placeholder(tf.int32, shape=[None, 3], name="zi")
      self.target = tf.placeholder(tf.int32, shape=[None, 3], name="target")
    else:
      self.zi = tf.placeholder(tf.int32, shape=[None], name="zi")
    self.zi_emb = tf.nn.embedding_lookup(self.word_emb, self.zi)

  def encoding_name(self):
    """
    :param name_seq: 3-D tensor [batch_size, seq_len, embedding_dim]
    :return: 
    """
    with tf.variable_scope("encoder"):
      outputs, state = tf.nn.dynamic_rnn(self.cell, self.name_emb, sequence_length=self.name_num, dtype=tf.float32)
      self.state = state
      self.encoder_output = outputs
    return outputs, state

  def decode_step(self, input, pre_state, reuse=True):
    if reuse:
      tf.get_variable_scope().reuse_variables()

    if self.encoder_output is not None:
      att = attention(pre_state, self.encoder_output, False)
      input = tf.layers.dense(tf.concat([input, att], axis=1), 64)
    output, state = self.cell(input, pre_state)

    return output, state


  def decode_zi(self):
    state = self.state

    with tf.variable_scope("decoder"):
      if self.mode == 'train':
        outputs = []
        decode_input = tf.unstack(self.zi_emb, axis=1)
        with tf.variable_scope("step_iter"):

          for inp in decode_input:
            output, state = self.decode_step(inp, state, reuse=False if len(outputs) == 0 else True)
            outputs.append(output)

        logits = tf.layers.dense(tf.transpose(outputs, [1, 0, 2]), term_size, name="logits")
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=logits)

        self.batch_loss = tf.reduce_mean(losses)
        tf.losses.add_loss(self.batch_loss)

      else:
        beam_size = 4
        output, state = self.decode_step(self.zi_emb, self.state, False)
        logit = tf.nn.softmax(tf.layers.dense(output, term_size, name="logits"))
        values_first, indices_first = tf.nn.top_k(logit, k=beam_size)
        seconds_i = tf.unstack(indices_first, axis=1, num=beam_size)
        seconds_v = tf.unstack(values_first, axis=1, num=beam_size)

        self.results_i = []
        self.results_v = []

        for inp, inp_value in zip(seconds_i, seconds_v):
          values_emb = tf.nn.embedding_lookup(self.word_emb, inp)
          o2, s2 = self.decode_step(values_emb, state, reuse=True)
          l2 = tf.nn.softmax(tf.layers.dense(o2, term_size, name="logits", reuse=True))
          v2, i2 = tf.nn.top_k(l2, k=beam_size)
          for i, v in zip(tf.unstack(i2, axis=1), tf.unstack(v2, axis=1)):
            self.results_i.append(tf.squeeze(tf.stack([inp, i])))
            self.results_v.append(tf.squeeze(tf.stack([inp_value, v])))

  def build_loss(self):

    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar("losses/total", total_loss)
    self.total_loss = total_loss
    self.global_step = tf.train.create_global_step()


def test():
  word_id = {}
  id_word = {}
  word_id_f = open('model/word_id', 'r').readlines()
  for line in word_id_f:
    feilds = line.strip().split()
    id_word[int(feilds[1])] = feilds[0]
    word_id[feilds[0]] = int(feilds[1])

  id_word[0] = 'Go'
  id_word[1] = 'End'
  id_word[2] = 'Unk'

  word_id['Go'] = 0
  word_id['End'] = 1
  word_id['Unk'] = 2

  g = tf.Graph()
  with g.as_default():
    model = Alias(None, 'test')
    model.build_input()
    model.encoding_name()
    model.decode_zi()

    checkpoint_dir = 'model/alias/'
    saver = tf.train.Saver()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(graph=g, config=gpu_config) as sess:
      create_restore_fn(checkpoint_dir, saver, sess)

      names_batch = [[word_id['白'], word_id['居'], word_id['易']]]
      names_num_batch = [2]
      zis_batch = [0]

      feed_dict = {"name:0": names_batch, "name_num:0": names_num_batch, "zi:0": zis_batch}

      index, value = sess.run([model.results_i, model.results_v], feed_dict=feed_dict)
      a = []
      for i, v in zip(index, value):
        a.append((''.join([id_word[term] for term in i]), v[0] * v[1]))

      b = sorted(a, key=lambda x: x[1], reverse=True)
      for term, value in b:
        print(term, value)


def train():
  word_id = {}
  word_id_f = open('model/word_id', 'r').readlines()
  for line in word_id_f:
    feilds = line.strip().split()
    word_id[feilds[0].decode('utf-8')] = int(feilds[1])

  word_vec = np.random.random((term_size, 100))
  word_vec_f = open('model/word_vec', 'r').readlines()
  for line in word_vec_f:
    feilds = line.strip().decode('utf-8').split()
    if len(feilds) != 101:
      continue
    word_vec[word_id[feilds[0]]] = [float(i) for i in feilds[1:]]

  f = open('data/name_alias/name_zi.txt')
  inputs = defaultdict(list)
  for line in f.readlines():
    feilds = line.strip().decode('utf-8').split()
    if len(feilds) != 2 or len(feilds[1]) < 2:
      print(line)
      continue
    names = [[word_id[term] for term in list(name)] for name in feilds]
    inputs[len(names[0])].append((names[0], names[1][:2]))

  g = tf.Graph()
  with g.as_default():
    model = Alias(np.array(word_vec), 'train')
    model.build_input()
    model.encoding_name()
    model.decode_zi()
    model.build_loss()

    learning_rate = tf.train.exponential_decay(
      learning_rate=0.01,
      global_step=model.global_step,
      decay_steps=3000,
      decay_rate=0.3,
      staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_tensor = tf.contrib.slim.learning.create_train_op(
      total_loss=model.batch_loss,
      optimizer=optimizer,
      global_step=model.global_step,
      clip_gradient_norm=5.0)

    saver = tf.train.Saver()

    checkpoint_dir = 'model/alias/'

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(graph=g, config=gpu_config) as sess:

      merged_summary_op = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(checkpoint_dir, g)

      sess.run(tf.global_variables_initializer())

      # save graph def
      graph_io.write_graph(g.as_graph_def(add_shapes=True), checkpoint_dir, "graph.pbtxt", as_text=False)

      create_restore_fn(checkpoint_dir, saver, sess)

      epoch = 1000
      batch_size = 256
      total_loss = []
      for e in range(epoch):
        print(e)
        for bucket, values in inputs.items():
          if len(values) < 100:
            continue
          names = [value[0] for value in values]
          zis = [value[1] for value in values]
          i = 0
          while i < len(names):
            names_batch = names[i: i + batch_size]
            zis_batch = zis[i: i + batch_size]
            target_batch = [zi + [End_Id] for zi in zis_batch]
            zis_batch = [[Go_Id] + zi for zi in zis_batch]
            names_num_batch = [bucket] * len(names_batch)
            i += batch_size

            feed_dict = {"name:0": names_batch, "zi:0": zis_batch, "target:0": target_batch,
                         "name_num:0": names_num_batch}

            summary, batch_loss, np_global_step = \
              sess.run([merged_summary_op, train_tensor, model.global_step], feed_dict=feed_dict)
            total_loss.append(batch_loss)

            summary_writer.add_summary(summary, i)

        print(np.mean(total_loss), np_global_step)

      save_path = os.path.join(checkpoint_dir, "model.ckpt")
      saver.save(sess, save_path, global_step=model.global_step)


train()
