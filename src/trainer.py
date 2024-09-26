import numpy as np

class MLPTrainer():
    def __init__(self, model, learning_rate=0.000001, batch_size=1, lossfn = 'sq-error'):
        self.model = model
        self.learning_rate = learning_rate
        self.lossfn = lossfn
        self.batch_size = batch_size

    def data_loader(self, data):
        num_batches = (data.shape[1] + self.batch_size - 1) // self.batch_size
        batches = []
        for i in range(num_batches):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, data.shape[1])
            batch = data[:,start_index:end_index]
            batches.append(batch)
        return batches

    def loss(self, y_pred, y_true):
        if self.lossfn == 'sq-error':
            loss = np.sum((y_pred - y_true) ** 2, axis=0, keepdims = True)
            loss = np.mean(loss)
            return loss
        elif self.lossfn == 'cross-entropy':
            epsilon = 1e-15
            loss = -np.sum(y_true * np.log(y_pred + epsilon), axis = 0, keepdims = True)
            loss = np.mean(loss)
            return loss

    def loss_grad(self, y_pred, y_true):
        if self.lossfn == 'sq-error':
            gradients = 2 * (y_pred - y_true)
            return np.mean(gradients, axis=1, keepdims=True)
        elif self.lossfn == 'cross-entropy':
            epsilon = 1e-15
            derivative = -y_true / (y_pred + epsilon)
            return derivative

    def softmax_derivative(self, x):
        y = self.model.softmax(x)
        n = len(y)
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian[i, j] = y[i][0] * (1 - y[j][0])
                else:
                    jacobian[i, j] = -y[i][0] * y[j][0]
        return jacobian

    def backward_pass(self, x, y_true):
        y_pred = self.model.forward_pass(x)
        loss_grad = self.loss_grad(y_pred, y_true)
        if self.model.is_softmax:
            self.model.output_layer.delta = self.softmax_derivative(self.model.output_layer.z)@loss_grad
        else:
            self.model.output_layer.delta = loss_grad
        next_weight = self.model.output_layer.weight
        next_delta = self.model.output_layer.delta
        self.model.output_layer.update_layer_parameters(self.learning_rate)
        for layer in reversed(self.model.layer_list):
            layer.update_delta(next_delta, next_weight)
            next_delta = layer.delta
            next_weight = layer.weight
            layer.update_layer_parameters(self.learning_rate)

    def train(self, x_train, y_train, x_val, y_val, epochs=150, acc=False):
        train_loss = [] ; val_loss = []
        if acc: train_acc = [] ; val_acc = []
        batched_x_train = self.data_loader(x_train)
        batched_y_train = self.data_loader(y_train)
        for epoch in range(epochs):
            total_loss = 0
            self.model.eval = True
            for i in range(0, len(batched_x_train)):
                x_batch = batched_x_train[i]
                y_batch = batched_y_train[i]
                y_pred = self.model.forward_pass(x_batch)
                batch_loss = self.loss(y_pred, y_batch).item()
                total_loss += batch_loss
                self.backward_pass(x_batch, y_batch)
            avg_loss = total_loss/len(batched_x_train)
            train_loss.append(avg_loss)
            v_loss = self.validate(x_val, y_val)
            val_loss.append(v_loss)
            if acc:
                train_accuracy = self.accuracy(x_train, y_train)
                val_accuracy = self.accuracy(x_val, y_val)
                train_acc.append(train_accuracy)
                val_acc.append(val_accuracy)
            if not acc:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {v_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {v_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        print('Training Finished')
        if not acc:
            return train_loss, val_loss
        else:
            return train_loss, val_loss, train_acc, val_acc

    def validate(self, x_val, y_val):
        self.model.eval = False
        y_pred = self.model.forward_pass(x_val)
        loss = self.loss(y_pred, y_val)
        return loss

    def accuracy(self, x_val, y_val):
        self.model.eval = False
        y_true = self.one_hot_to_labels(y_val)
        y_pred = self.model.forward_pass(x_val)
        y_pred = self.one_hot_to_labels(y_pred)
        num_samples = y_true.shape[1]
        num_correct = np.sum(y_true == y_pred)
        acc = num_correct / num_samples
        return acc

    def one_hot_to_labels(self,one_hot_matrix):
        num_samples = one_hot_matrix.shape[1]
        labels = []
        for i in range(num_samples):
            index = np.argmax(one_hot_matrix[:, i])
            labels.append(index)
        return np.array(labels).reshape(1,-1)