import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = 0.1
        
        # Initialize weights and biases
        self.W_xh = np.random.randn(hidden_dim, input_dim)
        self.W_hh = np.random.randn(hidden_dim, hidden_dim)
        self.W_hy = np.random.randn(output_dim, hidden_dim)
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))
        
        # Internal state
        self.h = np.zeros((hidden_dim, 1))
        
        # Store intermediate values for backpropagation
        self.xs = []
        self.hs = []
        self.ys = []

    def forward(self, x):
        # Update the hidden state
        self.h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, self.h) + self.b_h)
        
        # Compute the output
        y = np.dot(self.W_hy, self.h) + self.b_y
        
        # Store intermediate values
        self.xs.append(x)
        self.hs.append(self.h)
        self.ys.append(y)
        
        return y

    def backward(self, d_y):
        # Initialize gradients
        d_W_xh = np.zeros_like(self.W_xh)
        d_W_hh = np.zeros_like(self.W_hh)
        d_W_hy = np.zeros_like(self.W_hy)
        d_b_h = np.zeros_like(self.b_h)
        d_b_y = np.zeros_like(self.b_y)
        
        # Backpropagate through time
        d_h_next = np.zeros_like(self.h)
        for t in reversed(range(len(self.xs))):
            dy = d_y  # Gradient of cross-entropy loss
            d_W_hy += np.dot(dy, self.hs[t].T)
            d_b_y += dy
            dh = np.dot(self.W_hy.T, dy) + d_h_next
            dh_raw = (1 - self.hs[t] ** 2) * dh  # Gradient through tanh
            d_b_h += dh_raw
            d_W_xh += np.dot(dh_raw, self.xs[t].T)
            d_W_hh += np.dot(dh_raw, self.hs[t - 1].T) if t > 0 else np.zeros_like(self.W_hh)
            d_h_next = np.dot(self.W_hh.T, dh_raw)
        
        # Clip gradients to prevent exploding gradients (optional)
        clip_value = 10.0  # You can adjust this value as needed
        for dparam in [d_W_xh, d_W_hh, d_W_hy, d_b_h, d_b_y]:
            np.clip(dparam, -clip_value, clip_value, out=dparam)
        
        # Update weights and biases
        self.W_xh -= learning_rate * d_W_xh
        self.W_hh -= learning_rate * d_W_hh
        self.W_hy -= learning_rate * d_W_hy
        self.b_h -= learning_rate * d_b_h
        self.b_y -= learning_rate * d_b_y

        # Reset stored values for the next iteration
        self.xs = []
        self.hs = []
        self.ys = []

if __name__ == "__main__":
    input_dim = 5
    hidden_dim = 4
    output_dim = 6
    
    rnn = RNN(input_dim, hidden_dim, output_dim)
    
    # Define input sequence
    sequence = [
        np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]),
        np.array([[0.2], [0.3], [0.4], [0.5], [0.6]]),
        np.array([[0.3], [0.4], [0.5], [0.6], [0.7]])
    ]
    
    # Target output for each time step
    targets = [
        np.array([[0.5], [0.4], [0.3], [0.2], [0.1], [0.0]]),
        np.array([[0.2], [0.2], [0.2], [0.2], [0.2], [0.2]]),
        np.array([[0.7], [0.6], [0.5], [0.4], [0.3], [0.2]])
    ]
    
    # Training loop
    learning_rate = 0.01
    for epoch in range(1000):
        total_loss = 0
        for t in range(len(sequence)):
            x = sequence[t]
            target = targets[t]
            
            y = rnn.forward(x)
            temp_y = np.exp(y - np.max(y))
            softmax_y = np.exp(temp_y) / np.sum(np.exp(temp_y))  # Apply softmax
            loss = -np.sum(target * np.log(softmax_y))
            total_loss += loss
            
            d_y = softmax_y - target
            rnn.backward(d_y)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(sequence)}")
