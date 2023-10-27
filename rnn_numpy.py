import numpy as np

def softmax(x):
    #Prevent exp from exploding
    return np.exp(x - np.max(x))/np.sum(np.exp(x - np.max(x)), axis=0)

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, time_step):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_step = time_step
        self.learning_rate = 0.01
        
        # Initialize weights and biases
        self.W_x = np.random.randn(hidden_dim, input_dim)
        self.W_h = np.random.randn(hidden_dim, hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim)
        self.b_h = np.zeros((hidden_dim))
        self.b_y = np.zeros((output_dim))
        
        # Internal state
        self.h = np.zeros((time_step, hidden_dim)) 

        # Store intermediate values for backpropagation
        self.xs = []
        self.hs = []

    # Input : Each sequence
    # Output : Predicted value for each time step
    def forward(self, x):
        y = np.zeros((time_step, output_dim))
        for t in range(time_step):
            # Update the hidden state
            if t > 0:
                self.h[t] = np.tanh(np.dot(self.W_x, x[t]) + np.dot(self.W_h, self.h[t-1]) + self.b_h)
            else:
                self.h[t] = np.tanh(np.dot(self.W_x, x[t]) + self.b_h)

            # Compute the output
            y[t] = np.dot(self.W_y, self.h[t]) + self.b_y
            y[t] = softmax(y[t])

            # Store intermediate values
            self.xs.append(x[t].reshape(-1,1))
            self.hs.append(self.h[t].reshape(-1,1))

        return y

    def backward(self, d_y):
        # Initialize gradients
        d_W_y = np.zeros_like(self.W_y)
        d_W_x = np.zeros_like(self.W_x)
        d_W_h = np.zeros_like(self.W_h)
        d_b_y = np.zeros_like(self.b_y).reshape(-1,1)
        d_b_h = np.zeros_like(self.b_h).reshape(-1,1)
        d_h_next = np.zeros((hidden_dim)).reshape(-1,1)

        # Backpropagate through time
        for t in reversed(range(time_step)):
            dy = d_y[t]
            dy = dy.reshape(-1,1)
            d_W_y += np.dot(dy, self.hs[t].T)
            d_b_y += dy
            dh = np.dot(self.W_y.T, dy) + d_h_next
            dh_tanh = (1 - self.hs[t] ** 2) * dh  # Gradient through tanh
            d_b_h += dh_tanh
            d_W_x += np.dot(dh_tanh, self.xs[t].T)
            d_W_h += np.dot(dh_tanh, self.hs[t - 1].T) if t > 0 else np.zeros_like(self.W_h)
            d_h_next = np.dot(self.W_h.T, dh_tanh)

        d_b_y = d_b_y.reshape(-1)
        d_b_h = d_b_h.reshape(-1)

         # Clip gradients to prevent exploding gradients (optional)
        clip_value = 5.0  # You can adjust this value as needed
        for dparam in [d_W_x, d_W_h, d_W_y, d_b_h, d_b_y]:
            np.clip(dparam, -clip_value, clip_value, out=dparam)

        # Update weights and biases
        self.W_x -= self.learning_rate * d_W_x
        self.W_h -= self.learning_rate * d_W_h
        self.W_y -= self.learning_rate * d_W_y
        self.b_h -= self.learning_rate * d_b_h
        self.b_y -= self.learning_rate * d_b_y

        # Reset stored values for the next iteration
        self.xs = []
        self.hs = []


if __name__ == "__main__":
    batch_size = 5
    time_step = 4
    input_dim = 5
    hidden_dim = 4
    output_dim = 3

    rnn = RNN(input_dim, hidden_dim, output_dim, time_step)

    sequence = np.random.randn(batch_size, time_step, input_dim)
    total_loss = np.zeros((batch_size, time_step))

    # Generate random one-hot vectors for output
    targets = np.zeros((batch_size, time_step, output_dim))

    for i in range(batch_size):
        for j in range(time_step):
            # Generate a random index for the one-hot vector
            index = np.random.randint(output_dim)
            # Create the one-hot vector
            targets[i, j, index] = 1


    for epoch in range(1000):
        for n in range(batch_size):
            x = sequence[n]         #one sequence
            target = targets[n]

            y = rnn.forward(x)      #Predicted value for each time step of the sequence
            
            # One-hot vector
            loss = 0
            for t in range(time_step):
                index = np.where(target[t] == 1)[0]
                loss += -np.log(y[t][index])

            d_y = y - target    
            rnn.backward(d_y)   # d_y for each timestep

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss / time_step}")
        




