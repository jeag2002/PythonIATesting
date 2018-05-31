from mxnet import nd
from mxnet.gluon import nn

layer = nn.Dense(2)
layer.initialize()

x = nd.random.uniform(-1,1,(3,4))
layer(x)

layer.weight.data()

net = nn.Sequential()
# Creating layers in a name scope to assign each layer a unique
# name so we can load/save their parameters later.
with net.name_scope():
    # Add a sequence of layers.
    net.add(
        # Similar to Dense, it is not necessary to specify the
        # input channels by the argument `in_channels`, which will be
        # automatically inferred in the first forward pass. Also,
        # we apply a relu activation on the output.
        #
        # In addition, we can use a tuple to specify a
        # non-square kernel size, such as `kernel_size=(2,4)`
        nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        # One can also use a tuple to specify non-symmetric
        # pool and stride sizes
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # flatten the 4-D input into 2-D with shape
        # `(x.shape[0], x.size/x.shape[0])` so that it can be used
        # by the following dense layers
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10)
    )

net.initialize()
# Input shape is (batch_size, color_channels, height, width)
x = nd.random.uniform(shape=(4,1,28,28))
y = net(x)
y.shape

print (net[0].weight.data().shape, net[5].bias.data().shape)






