import tensorflow as tf

class DNN:
    def __init__(self, config):
        """
        Config parameters:
        - input_shape: int, shape of the input data
        - num_dense_layers: int, number of dense layers in the network
        - dense_nodes: list of ints, number of nodes for each dense layer
        - activation_functions: list of str, activation functions for input and dense layers
        - output_activation: str, activation function for the output layer
        - batch_normalization: bool, whether to include batch normalization after certain layers
        - dropout_rate: float, dropout rate for regularization
        - l2_regularization: float, L2 regularization parameter
        """
        self.input_shape = config['input_shape']
        self.num_dense_layers = config['num_dense_layers']
        self.dense_nodes = config['dense_nodes']
        self.activation_functions = config['activation_functions']
        self.output_activation = config['output_activation']
        self.batch_normalization = config['batch_normalization']
        self.dropout_rate = config['dropout_rate']
        self.l2_regularization = config['l2_regularization']

        # create a list of nodes according to the number of dense layers
        half_layers = self.num_dense_layers // 2
        self.nodes = [int(self.dense_nodes * (1 - abs(i - half_layers) / half_layers)) for i in range(self.num_dense_layers)]

        self.model = self.build_model()

    def build_model(self):
        input_layer = tf.keras.Input(shape=(self.input_shape, 1))

        x = tf.keras.layers.Flatten()(input_layer)
        x = tf.keras.layers.Dense(self.nodes[0], activation=self.activation_functions[0])(x)

        # add dense layers based on num_dense_layers
        for i in range(1, self.num_dense_layers):
            x = tf.keras.layers.Dense(self.nodes[i], activation=self.activation_functions[i],
                                      kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_rate > 0.0:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # output dense layer with 8 nodes (one for each 4 vector component)
        output_layer = tf.keras.layers.Dense(8, activation=self.output_activation)(x)

        # build and return the model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        return model