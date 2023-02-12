import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, f1_score

class Model():
    def forward(self, x):
        pass
    def backward(self, x, y):
        pass

class ConvolutionLayer(Model):
    def __init__(self, num_filters, filter_size, stride, padding):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        self.input = None
        self.cache = None

    def __str__(self):
        return f'Convolution Layer: {self.num_filters} filters of size {self.filter_size}x{self.filter_size} with stride {self.stride} and padding {self.padding}'
    
    def init_weights(self, num_channels):
        self.weights = np.random.randn(self.num_filters, num_channels, self.filter_size, self.filter_size) * \
                        np.sqrt(2 / (self.filter_size * self.filter_size * num_channels))
        
        self.biases = np.zeros(self.num_filters)

    def forward(self, input):
        # input has shape (batch_size, num_channels, input_height, input_width)
        n, c, h, w = input.shape
        output_height = (h + 2 * self.padding - self.filter_size ) // self.stride + 1
        output_width = (w + 2 * self.padding - self.filter_size ) // self.stride + 1
        output = np.zeros((n, self.num_filters, output_height, output_width))
        #init weights with xavier initialization
        if self.weights is None:
            self.init_weights(c)

        # pad the input
        input_padded = np.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
                       
        batch_stride, channel_stride, height_stride, width_stride = input_padded.strides
        # print(f'batch_stride: {batch_stride}, channel_stride: {channel_stride}, height_stride: {height_stride}, width_stride: {width_stride}')
        strided_windows = np.lib.stride_tricks.as_strided(
            input_padded,
            shape = (n, c, output_height, output_width, self.filter_size, self.filter_size),
            strides = (batch_stride, channel_stride, height_stride * self.stride, width_stride * self.stride, height_stride, width_stride)
        )

        output = np.einsum('bihwkl,oikl->bohw', strided_windows, self.weights) + self.biases[None, :, None, None]

        # print(f'output.shape: {output.shape}')
        self.cache = input, strided_windows
        
        return output

    def backward(self, dout, learning_rate):
    
        input, strided_windows = self.cache

        padding = self.filter_size - 1 if self.padding == 0 else self.filter_size - 1 - self.padding
        dilate = self.stride - 1
        dout_dilated_padded = dout.copy()
        # print(f'dilate: {dilate}')
        # print(f'dout_dilated_padded:\n {dout_dilated_padded}')
        if dilate > 0:
            dout_dilated_padded = np.insert(dout_dilated_padded, range(1, dout.shape[2]), 0, axis=2)
            dout_dilated_padded = np.insert(dout_dilated_padded, range(1, dout.shape[3]), 0, axis=3)
             
        # print(f'dout_dilated_padded:\n {dout_dilated_padded}')
        if padding > 0:
            dout_dilated_padded = np.pad(dout_dilated_padded, ((0,0), (0,0), (padding, padding), (padding, padding)), 'constant')

        out_h, out_w = input.shape[2:]
        out_b, out_c= dout.shape[:2]
        batch_stride, channel_stride, height_stride, width_stride = dout_dilated_padded.strides

        dout_windows = np.lib.stride_tricks.as_strided(dout_dilated_padded, 
            shape = (out_b, out_c, out_h, out_w, self.filter_size, self.filter_size),
            strides = (batch_stride, channel_stride, height_stride * 1, width_stride * 1, height_stride, width_stride)
        )
        # print(f'dout_windows.shape: {dout_windows.shape}')
        # print(f'dout_windows: {dout_windows}')
        
        rotate_weights = np.rot90(self.weights, 2, (2, 3))

        dx = np.einsum('bohwkl, oikl -> bihw', dout_windows, rotate_weights)
        dw = np.einsum('bihwkl, bohw -> oikl', strided_windows, dout)
        db = np.sum(dout, axis=(0, 2, 3))

        # print(dout.shape)
        

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        

        return dx

class ActivationLayer(Model):
    def __init__(self):
        self.input = None

    def __str__(self):
        return 'ReLU Activation Layer'

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, dL_dout, learning_rate):
        return dL_dout * (self.input > 0)

class PoolingLayer(Model):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None

    def __str__(self):
        return f'Pooling Layer (pool_size={self.pool_size}, stride={self.stride})'


    def forward(self, input):
        self.input = input
        n, c, h, w = input.shape
        output_height = (h - self.pool_size) // self.stride + 1
        output_width = (w - self.pool_size) // self.stride + 1
       
        batch_stride, channel_stride, height_stride, width_stride = input.strides
        input_windows = np.lib.stride_tricks.as_strided(input,
            shape = (n, c, output_height, output_width, self.pool_size, self.pool_size),
            strides = (batch_stride, channel_stride, height_stride * self.stride, width_stride * self.stride, height_stride, width_stride)
        )
        output = np.max(input_windows, axis=(4, 5))
        
        if self.stride == self.pool_size:
            mask = output.repeat(self.stride, axis=-2).repeat(self.stride, axis=-1)
            # print(f'shape of mask after repeat: {mask.shape}')
            # print(f'mask: {mask}')
            # pad for odd shape
            h_pad = h - mask.shape[-2]
            w_pad = w - mask.shape[-1]
            mask = np.pad(mask, ((0,0), (0,0), (0, h_pad), (0, w_pad)), 'constant')
            # print(f'shape of mask after pad: {mask.shape}')
            # print(f'mask: {mask}')
            mask = np.equal(input, mask)
            # print(f'shape of mask after equal: {mask.shape}')
            # print(f'mask: {mask}')

            self.cache = mask
        return output

    def backward(self, dL_dout, learning_rate):
        n, c, h, w = self.input.shape
        h_out, w_out = dL_dout.shape[-2:]
        stride = self.stride
        if stride == self.pool_size:
            dL_dout = dL_dout.repeat(stride, axis=-2).repeat(stride, axis=-1)
            mask = self.cache
            # pad for odd shape
            h_pad = h - dL_dout.shape[-2]
            w_pad = w - dL_dout.shape[-1]
            dL_dout = np.pad(dL_dout, ((0,0), (0,0), (0, h_pad), (0, w_pad)), 'constant')

            return dL_dout * mask
            
        else:
            dx = np.zeros(self.input.shape)

            for i in range(n):
                for j in range(c):
                    for k in range(h_out):
                        for l in range(w_out):
                            i_t, j_t = np.where(np.max(self.input[i, j, k * self.stride : k * self.stride + self.pool_size, l * self.stride : l * self.stride + self.pool_size]) == self.input[i, j, k * self.stride : k * self.stride + self.pool_size, l * self.stride : l * self.stride + self.pool_size])
                            i_t, j_t = i_t[0], j_t[0]
                            dx[i, j, k * self.stride : k * self.stride + self.pool_size, l * self.stride : l * self.stride + self.pool_size][i_t, j_t] = dL_dout[i, j, k, l]
        
        return dx

class FlatteningLayer(Model):
    def __init__(self):
        self.input = None

    def __str__(self):
        return "Flattening Layer"

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        return np.reshape(input, (batch_size, -1))

    def backward(self, dL_dout, learning_rate):
        return np.reshape(dL_dout, self.input.shape)

class FullyConnectedLayer(Model):
    def __init__(self, output_dim):
        self.weights = None
        self.biases = None
        self.input = None
        self.output_dim = int(output_dim)
        self.cache = None
    
    def __str__(self):
        return f'Fully Connected Layer (output_dim={self.output_dim})'

    def forward(self, input):
        self.input = input
        
        if self.weights is None:
            self.weights = np.random.randn(input.shape[1], self.output_dim) * np.sqrt(2 / input.shape[1])
            self.biases = np.zeros(self.output_dim)
        return np.dot(input, self.weights) + self.biases

    def backward(self, dL_dout, learning_rate):
        dL_dW = np.dot(self.input.T, dL_dout)
        dL_db = np.sum(dL_dout, axis=0)
        dL_din = np.dot(dL_dout, self.weights.T)

        self.weights -= learning_rate * dL_dW
        self.biases -= learning_rate * dL_db

        return dL_din
    
class SoftMaxLayer(Model):
    def __init__(self):
        self.input = None

    def __str__(self):
        return 'Softmax Layer'

    def forward(self, input):
        self.input = input
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, dL_dout):
        return dL_dout

class ModelBuilder:
    def __init__(self):
        self.model = []

    def __str__(self):
        # call __str__ of each layer in model
        info = "Model: \n"
        for layer in self.model:
            info += layer.str() + "\n"
        return info


    def build(self):
        model = self.model

        model.append(ConvolutionLayer(6, 5, 1, 0))
        model.append(ActivationLayer())
        model.append(PoolingLayer(2, 2))
        model.append(ConvolutionLayer(16, 5, 1, 0))
        model.append(ActivationLayer())
        model.append(PoolingLayer(2, 2))
        model.append(FlatteningLayer())
        model.append(FullyConnectedLayer(120))
        model.append(ActivationLayer())
        model.append(FullyConnectedLayer(84))
        model.append(ActivationLayer())
        model.append(FullyConnectedLayer(10))
        model.append(SoftMaxLayer())


        return self.model
    

# train model having X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size, y is one hot encoded
def train(model, X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size):
# initialize loss and accuracy lists
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_f1 = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        # shuffle training data
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        # get number of batches
        num_batches = len(X_train) // batch_size
        loss = 0
        acc = 0
        for i in tqdm(range(num_batches)):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            y_batch = y_train[i * batch_size:(i + 1) * batch_size]
            # print(X_batch.shape, y_batch.shape)
            
            # forward pass
            out = X_batch
            for layer in model:
                out = layer.forward(out)
            # print(f'out.shape: {out.shape}')
            # print(f'out: {out}')
            
            loss += log_loss(y_batch, out)

            # calculate accuracy
            acc += accuracy_score(np.argmax(y_batch, axis=1), np.argmax(out, axis=1))

            # backward pass
            dL_dout = np.copy(out)
            dL_dout -= y_batch
            dL_dout /= batch_size
            for layer in reversed(model):
                dL_dout = layer.backward(dL_dout, learning_rate)
            
        train_loss.append(loss/num_batches)
        train_acc.append(acc/num_batches)
        # validation
        
        val_out = X_val
        for layer in model:
            val_out = layer.forward(val_out)
        val_loss.append(log_loss(y_val, val_out))
        val_acc.append(accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_out, axis=1)))
        curr_f1 = f1_score(np.argmax(y_val, axis=1), np.argmax(val_out, axis=1), average='macro')
        val_f1.append(curr_f1)
        print(f'learning_rate: {learning_rate}')

        print(f'Train Loss: {train_loss[-1]:.4f} | Train Acc: {train_acc[-1]:.4f}')
        print(f'Val Loss: {val_loss[-1]:.4f} | Val Acc: {val_acc[-1]:.4f}')
        print(f'Val F1: {val_f1[-1]:.4f}')

        

    return train_loss, train_acc, val_loss, val_acc

