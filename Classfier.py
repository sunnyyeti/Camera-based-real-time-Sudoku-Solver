import tensorflow as tf
import os
from gray_centre_sample import *
def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)


def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def train_gray_cntre_model():
    from sklearn.model_selection import train_test_split
    images,labels = get_gray_centre_data()
    Xtr,Xte, ytr,yte = train_test_split(images,labels,test_size=0.1)
    x = tf.placeholder(tf.float32, [None, 784], name='input_image')
    y_ = tf.placeholder(tf.float32, [None, 9], name='input_label')  # watch
    W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    b_conv1 = bias_variable([32], name='b_conv1')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    b_conv2 = bias_variable([64], name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
    b_fc1 = bias_variable([1024], name='b_fc1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 9], name='W_fc2')  # watch
    b_fc2 = bias_variable([9], name='b_fc2')  # watch

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    pred_label = tf.argmax(y_conv, 1, name="pred_label")
    pred_prob = tf.nn.softmax(y_conv, name='pred_prob')
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tr_num = Xtr.shape[0]
    print "training number is ", tr_num
    indices = np.arange(tr_num)
    np.random.shuffle(indices)
    j = 0
    global_steps = 20000
    for i in range(global_steps):
        if j+100>tr_num:
            np.random.shuffle(indices)
            j = 0
        feed_xtr = Xtr[indices[j:j + 100]]
        #print feed_xtr.shape
        feed_ytr = ytr[indices[j:j + 100]]
        #print feed_ytr.shape
        j += 100
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={
                x: feed_xtr, y_: feed_ytr, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            if i % 500 == 0:
                vald_accuracy = accuracy.eval(session=sess, feed_dict={x: Xte[:200], y_: yte[:200], keep_prob: 1.0})
                print("validation_accuracy %g" % (vald_accuracy))
        train_step.run(session=sess, feed_dict={x: feed_xtr, y_: feed_ytr, keep_prob: 0.5})

    saver = tf.train.Saver()
    folder = './model/gray_model_baseline/'
    try:
        os.makedirs(folder)
    except:
        pass
    saver.save(sess, folder+'/my_model', global_step=global_steps)  # create checkpoint
    # saver.export_meta_graph(filename= 'my_model_graph.meta')  # create meta graphs


def get_trained_model(subfolder, global_step):
    model_path = './model/%s/my_model-%s.meta'%(subfolder,global_step)
    if not os.path.exists(model_path):
        print "No trained model! Please train firstly."
        return
    sess = tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint('./model/%s/'%subfolder))
    graph = tf.get_default_graph()
    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("accuracy:0")
    x = graph.get_tensor_by_name("input_image:0")
    y_ = graph.get_tensor_by_name("input_label:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    pred_label = graph.get_tensor_by_name("pred_label:0")
    def predict(Input):
        feed = {x:Input, y_:np.zeros((81,9)), keep_prob:1.0} #watch
        prediction = pred_label.eval(session=sess, feed_dict=feed)
        return prediction
    return predict

if __name__ ==  "__main__":
    train_gray_cntre_model()
