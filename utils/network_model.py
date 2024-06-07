import torch
import numpy as np

class BaseNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, dtype=torch.double)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.double)
        self.fc3 = torch.nn.Linear(hidden_dim, latent_dim, dtype=torch.double)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MixtureNetwork(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, num_components):
        super().__init__()
        self.num_components = num_components
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(latent_dim, num_components, dtype=torch.double)
        self.fc2 = torch.nn.Linear(latent_dim, num_components * output_dim, dtype=torch.double)
        self.fc3 = torch.nn.Linear(latent_dim, num_components * output_dim, dtype=torch.double)
        self.softmax = torch.nn.Softmax(dim=1)
        self.exp = torch.exp

    def forward(self, x):
        pi = self.softmax(self.fc1(x)) # mixture weights
        mu = self.fc2(x).view(-1, self.num_components, self.output_dim) # means
        sigma = self.exp(self.fc3(x)).view(-1, self.num_components, self.output_dim) # standard deviations
        return pi, mu, sigma

class MDN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, num_components):
        super().__init__()
        self.base_network = BaseNetwork(input_dim, hidden_dim, latent_dim)
        self.mixture_network = MixtureNetwork(latent_dim, output_dim, num_components)
        self.input_scale = torch.from_numpy(np.array([1280, 720]).astype(float))
        self.output_scale = torch.from_numpy(np.array([40, 20, 150]).astype(float))

    def forward(self, x):
        x /= self.input_scale
        z = self.base_network(x) # latent features
        pi, mu, sigma = self.mixture_network(z) # mixture parameters
        mu = self.output_scale * mu
        sigma = self.output_scale * sigma
        return pi, mu, sigma

    def sample(self, x):
        pi, mu, sigma = self.forward(x) # get the mixture parameters
        categorical = torch.distributions.Categorical(pi) # create a categorical distribution
        normal = torch.distributions.Normal(mu, sigma) # create a normal distribution
        k = categorical.sample() # sample a component index
        y = normal.sample()[torch.arange(x.size(0)), k] # sample a value from the selected component
        return y

    def mdn_loss(self, pi, mu, sigma, y):
        normal = torch.distributions.Normal(mu, sigma) # create a normal distribution
        log_prob = normal.log_prob(y.unsqueeze(1).expand_as(mu)) # compute the log-probability of each component
        log_prob = log_prob.sum(2) # sum over the output dimension
        log_prob = torch.logsumexp(log_prob + torch.log(pi), dim=1) # combine with the mixture weights
        loss = -log_prob.mean() # compute the negative log-likelihood
        return loss