import tensorflow as tf
import numpy as np

# This is just a practice test

# Split dataset into training (66%) and test (33%) set
filename = 'training_set.csv'
raw_data = open(filename, 'rt')
training_set = np.loadtxt(raw_data, delimiter=",")

filename = 'training_labels.csv'
raw_data = open(filename, 'rt')
training_labels = np.loadtxt(raw_data, delimiter=",")

filename = 'test_set.csv'
raw_data = open(filename, 'rt')
test_set = np.loadtxt(raw_data, delimiter=",")

filename = 'test_labels.csv'
raw_data = open(filename, 'rt')
test_labels = np.loadtxt(raw_data, delimiter=",")

print("Dataset ready.") 

# Parameters
learning_rate   = 0.01 
mini_batch_size = 10
training_epochs = 5000
display_step    = 1000

# Network Parameters
n_hidden_1  = 64    # 1st hidden layer of neurons
n_hidden_2  = 32    # 2nd hidden layer of neurons
n_hidden_3  = 16    # 3rd hidden layer of neurons
n_input     = 50   # number of features after LSA

# Tensorflow Graph input
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float64, shape=[None, n_input], name="x-data")
    y = tf.placeholder(tf.float64, shape=[None, 1], name="y-labels")

print("Creating model.")

# Create model
def multilayer_perceptron(x, weights):
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            weights['h1'] = tf.Variable(tf.random_normal([n_input, n_hidden_1], name = "W1", dtype=np.float64))
            tf.summary.histogram("layer_1_weights", weights['h1'])
            weights['h2'] = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], name = "W2", dtype=np.float64))
            tf.summary.histogram("layer_2_weights", weights['h2'])
            weights['h3'] = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], name = "W3", dtype=np.float64))
            tf.summary.histogram("layer_3_weights", weights['h3'])           
            weights['out'] = tf.Variable(tf.random_normal([n_hidden_3, 1], name = "W4", dtype=np.float64))
            tf.summary.histogram("output_layer_weights", weights['out'])            
        with tf.name_scope("inputs"):
            # First hidden layer with SIGMOID activation
            layer_1 = tf.matmul(x, weights['h1'])
            # Second hidden layer with SIGMOID activation
            layer_2 = tf.matmul(layer_1, weights['h2'])
            # Third hidden layer with SIGMOID activation
            layer_3 = tf.matmul(layer_2, weights['h3'])
            # Output layer with SIGMOID activation
            out_layer = tf.matmul(layer_3, weights['out'])
        
        layer_1 = tf.nn.sigmoid(layer_1) 
        tf.summary.histogram("layer_1_outputs", layer_1)
        layer_2 = tf.nn.sigmoid(layer_2)
        tf.summary.histogram("layer_2_outputs", layer_2)
        layer_3 = tf.nn.sigmoid(layer_3)
        tf.summary.histogram("layer_3_outputs", layer_3)
        out_layer = tf.nn.sigmoid(out_layer)
        tf.summary.histogram("output_layer_outputs", out_layer)
    return out_layer

# Layer weights, should change them to see results
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], dtype=np.float64)),       
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], dtype=np.float64)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],dtype=np.float64)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, 1], dtype=np.float64))
}

# Construct model
pred = multilayer_perceptron(x, weights)

# Define loss and optimizer
with tf.name_scope("loss"):
    cost = tf.nn.l2_loss(pred-y,name="squared_error_cost")
    tf.summary.scalar("loss", cost)
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

print("Model ready.")

# Launch the graph
with tf.Session() as sess:
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    sess.run(init)

    print("Starting Training.")

    # Training cycle
    for epoch in range(training_epochs):
        #avg_cost = 0.
        # minibatch loading
        minibatch_x = training_set[mini_batch_size*epoch:mini_batch_size*(epoch+1)]
        minibatch_y = training_labels[mini_batch_size*epoch:mini_batch_size*(epoch+1)]
        # Run optimization op (backprop) and cost op
        minibatch_Y = np.array(minibatch_y).reshape(len(minibatch_y),1)
        _, c = sess.run([optimizer, cost], feed_dict={x: minibatch_x, y: minibatch_Y})

        # Compute average loss
        avg_cost = c / (minibatch_x.shape[0]+0.01)

        # Display logs per epoch
        if (epoch) % display_step == 0:
            print("Epoch:", '%05d' % (epoch), "Training accuracy=", "{:.9f}".format((1-avg_cost)*100))
        if (epoch) % 50 == 0:    
            result = sess.run(merged,feed_dict={x: minibatch_x, y: minibatch_Y})
            writer.add_summary(result,epoch)

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    test_error = tf.nn.l2_loss(pred-y,name="squared_error_test_cost")/test_set.shape[0]
    test_labels_y = np.array(test_labels).reshape(len(test_labels),1)
    print("Test Accuracy: ", 100-test_error.eval({x: test_set, y: test_labels_y})*100, "%")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    