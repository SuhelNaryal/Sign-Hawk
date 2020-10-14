import keras
import tensorflow as tf

def HandKeyPointsLoss(y_true, y_pred):
    y_pred = keras.backend.cast(y_pred, dtype=tf.float32)
    left_hand_true = y_true[:, :, 0]
    right_hand_true = y_true[:, :, 1]

    left_keypoints_true = y_true[:, :, 2:65]
    right_keypoints_true = y_true[:, :, 65:]

    left_hand_loss = tf.reduce_sum(keras.losses.binary_crossentropy(left_hand_true, y_pred[:, :, 0]), axis=-1) / left_hand_true.shape[0]
    right_hand_loss = tf.reduce_sum(keras.losses.binary_crossentropy(right_hand_true, y_pred[:, :, 1]), axis=-1) / right_hand_true.shape[0]

    left_keypoints_loss = tf.reduce_sum(left_hand_true * keras.losses.mean_squared_error(left_keypoints_true, y_pred[:, :, 2:65]), axis=-1) / tf.reduce_sum(left_hand_true, axis=-1)
    right_keypoints_loss = tf.reduce_sum(right_hand_true * keras.losses.mean_squared_error(right_keypoints_true, y_pred[:, :, 65:]), axis=-1) / tf.reduce_sum(right_hand_true, axis=-1)

    return left_hand_loss + right_hand_loss + left_keypoints_loss, right_keypoints_loss


class HandKeyPoints():

    def __init__(self, learning_rate=0.001):
        super(HandKeyPoints, self).__init__()

        input_layer = keras.Input(shape=(256, 256, 3))

        resnetbackbone = keras.applications.ResNet50V2(input_shape=(256, 256, 3), include_top=False)

        resnetbackbone_out = resnetbackbone(input_layer)
        global_avg_pool = keras.layers.GlobalAveragePooling2D()(resnetbackbone_out)
        dense1 = keras.layers.Dense(units=2048, activation='relu')(global_avg_pool)
        dense2 = keras.layers.Dense(units=2048, activation='relu')(dense1)

        left_hand = keras.layers.Dense(units=1, activation='sigmoid', name='left_hand')(dense2)
        right_hand = keras.layers.Dense(units=1, activation='sigmoid', name='right_hand')(dense2)

        left_keypoints = keras.layers.Dense(units=63, name='left_keypoints')(dense2)
        right_keypoints = keras.layers.Dense(units=63, name='right_keypoints')(dense2)

        output = keras.backend.concatenate([left_hand, right_hand, left_keypoints, right_keypoints], axis=1)

        self.model = keras.Model(inputs=input_layer, outputs=output)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = HandKeyPointsLoss

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        print(self.model.summary())

HandKeyPoints()

