import numpy as np
from sklearn.metrics import f1_score

class SparseCategoricalCrossentropy:
    def __call__(self, y_true, y_pred):
        # Convert to one-hot
        y_true = y_true.reshape(-1)
        n_samples = y_pred.shape[0]
        
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Calculate cross entropy
        loss = -np.sum(np.log(y_pred[np.arange(n_samples), y_true])) / n_samples
        return loss

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep

    def update(self, params, grads):
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def train(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    loss_fn = SparseCategoricalCrossentropy()
    optimizer = Adam()
    
    train_losses = []
    val_losses = []
    
    n_samples = len(x_train)
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        # Training
        epoch_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # Forward pass
            output = model.forward(batch_x)
            loss = loss_fn(batch_y, output)
            epoch_loss += loss
            
            # Backward pass (to be implemented)
            # grads = model.backward(batch_y)
            # optimizer.update(model.params, grads)
        
        train_losses.append(epoch_loss / n_batches)
        
        # Validation
        val_output = model.forward(x_val)
        val_loss = loss_fn(y_val, val_output)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_losses[-1]:.4f}")
        print(f"Validation Loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses

def evaluate(model, x_test, y_test):
    predictions = model.forward(x_test)
    pred_classes = np.argmax(predictions, axis=1)
    return f1_score(y_test.reshape(-1), pred_classes, average='macro') 