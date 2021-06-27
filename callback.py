import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(X,Y),(x,y) = mnist.load_data()
training_image = x/255.0
model=tf.keras.models.Sequential([
tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,tf.nn.relu),


])
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss')<0.4:
            print("loss")
            self.model.stop_training=True

callback = mycallback()
model.compile(optimizer="Adam",loss="sparse_categorical_crossentropy")
model.fit(X,Y,epochs=10,callbacks=[callback])
