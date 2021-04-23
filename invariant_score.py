import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

def compute_pairwise_similairity(x, y):
    x_norms = tf.linalg.norm(x, axis=-1)
    y_norms = tf.linalg.norm(y, axis=-1)
    x_norms = tf.expand_dims(x_norms, axis=1)
    y_norms = tf.expand_dims(y_norms, axis=0)
    norms = tf.math.maximum(x_norms * y_norms, 1e-8)
    x  = tf.expand_dims(x, axis=1)
    y  = tf.expand_dims(y, axis=0)
    cossim = tf.reduce_sum(x*y, axis=-1) / norms
    return cossim

if __name__ == '__main__':
    (_,y_train),(_,y_test) = tf.keras.datasets.mnist.load_data()
    data = np.load('all_MNIST_test_imgs_12samples_30deg_relu_activations.npz')

    shape = data['arr_0'].shape
    X = data['arr_0']

    # please modify this with
    # the not rotated features
    # of the training (64) before
    # the relu and comment out
    # y_train
    X_train = X.reshape(-1, 64)
    y_train = np.repeat(y_test[None], 12, axis=0).ravel()

    batch_size = 256
    n_batches  = int(np.ceil(shape[1] / batch_size))
    acc = []
    pbar = tqdm(range(shape[0]))
    for rot in pbar:
        for b in range(n_batches):
            li = b*batch_size
            ri = min((b+1)*batch_size, shape[1])
            dst = compute_pairwise_similairity(X[rot][li:ri], X_train)
            dst = dst.numpy() - 10.0*np.eye(*dst.shape)
            current_id = np.argmax(dst, axis=1)
            acc.append(y_train[current_id] == y_test[li:ri])
            M = np.concatenate(acc).mean()
            pbar.set_description(f'{M*100:.2f}')

    M = np.concatenate(acc).mean()
    print(M*100)
