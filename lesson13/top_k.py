import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 返回预则结果的准确率

def accuracy(output, target, topk=(1,)):

    """
    :param output: [10,6]
    :param target: [10]
    :param topk: top_k acc
    :return:
    """

    maxk = max(topk)
    batch_size = target.shape[0]

    idx = tf.math.top_k(output, maxk).indices
    idx = tf.transpose(idx, [1, 0])
    target = tf.broadcast_to(target, idx.shape)
    correct = tf.equal(idx, target)

    result = []
    for k in topk:
        val_cor = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        res = tf.reduce_sum(val_cor)
        acc = float(res * (100.0 / batch_size))
        result.append(acc)

    return result

output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('prob:', output.numpy())
preb = tf.argmax(output, axis=1)
print('preb:', preb.numpy())
print('label:', target.numpy())

acc = accuracy(output, target, topk=(1, 2, 3, 4, 5, 6))

print('top_1-6_acc:', acc)