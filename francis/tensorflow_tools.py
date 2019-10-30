import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework import nest


def loop_tf(loop_fn, inputs, persistent_initializer, transient_initializer, n=None, time_major=False):
    def create_tensor_array(initial_tensor: tf.Tensor):
        return tf.TensorArray(initial_tensor.dtype, size=n, element_shape=initial_tensor.get_shape())

    tensor_arrays = nest.map_structure(create_tensor_array, persistent_initializer)

    def while_fn(*args):
        current_iteration = args[0]
        persistent_values = args[1]
        transient_values = args[2]
        current_tensor_arrays = args[3]
        if time_major:
            input_values = inputs[current_iteration]
        else:
            input_values = inputs[:, current_iteration]

        new_persistent, new_transient = loop_fn(input_values, persistent_values, transient_values)
        flat_new_persistent = nest.flatten(new_persistent)
        flat_tensor_arrays = nest.flatten(current_tensor_arrays)
        flat_written_tensor_arrays = [
            ta.write(current_iteration, a) for ta, a in zip(flat_tensor_arrays, flat_new_persistent)
        ]
        new_tensor_arrays = nest.pack_sequence_as(current_tensor_arrays, flat_written_tensor_arrays)
        return current_iteration + 1, new_persistent, new_transient, new_tensor_arrays

    def while_cond(*args):
        seq_len = tf.shape(inputs)[0] if time_major else tf.shape(inputs)[1]
        return tf.less(args[0], seq_len)

    _, final_persistent, final_transient, final_tensor_arrays = \
        tf.while_loop(while_cond, while_fn, (0, persistent_initializer, transient_initializer, tensor_arrays))

    final_sequence_tensors = nest.map_structure(lambda x: x.stack(), final_tensor_arrays)

    def make_batch_major(tensor):
        permutation = np.arange(len(tensor.get_shape()))
        permutation[:2] = permutation[:2][::-1]
        return tf.transpose(tensor, permutation)

    if not time_major:
        final_sequence_tensors = nest.map_structure(make_batch_major, final_sequence_tensors)

    return final_sequence_tensors


def combine_flat_list(_structure, _flat_list, axis=1):
    _combined = []
    for i in range(len(_flat_list[0])):
        t = []
        for v in _flat_list:
            t.append(v[i])
        cc = tf.concat(t, axis)
        _combined.append(cc)
    return nest.pack_sequence_as(_structure, _combined)


def tf_structure_equal(_s1, _s2):
    return tf.reduce_all([tf.reduce_all(tf.equal(a, b)) for a, b in zip(_s1, _s2)])


def read_summary(path):
    events = [os.path.join(path, a) for a in os.listdir(path) if a.count('events') > 0]

    d = dict()
    for e_path in events:
        try:
            for e in tf.train.summary_iterator(e_path):
                for v in e.summary.value:
                    if v.tag not in d.keys():
                        d[v.tag] = []
                    d[v.tag].append((e.step, v.simple_value))
                    if v.tag == 'loss':
                        print(v.simple_value)
        except:
            pass

    for k, v in d.items():
        a = np.array(d[k])
        sorted_inds = np.argsort(a[:, 0])
        d[k] = a[sorted_inds]

    return d


if __name__ == '__main__':
    def loop_fn(inputs, persistent, transient):
        new_transient = persistent**2
        result = tf.where(tf.greater(new_transient, 25.), inputs, inputs + persistent)
        return result, new_transient

    a = tf.ones((1, 10))
    b = loop_tf(loop_fn, a, tf.zeros((1,)), tf.ones((1,)), 10)
    with tf.Session() as session:
        print(session.run(b))

