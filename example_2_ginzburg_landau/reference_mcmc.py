import torch
import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ginzburg Landau model and its parameters
dim = 5
p = 3
d = dim ** p
tau=2.0
lamb=0.5
alpha=0.1

def potential(X:torch.Tensor):
    X = torch.reshape(X, (dim,dim,dim))
    temp = torch.norm(torch.roll(X, -1, dims=0) - X)**2 \
            + torch.norm(torch.roll(X, -1, dims=1) - X)**2 \
            + torch.norm(torch.roll(X, -1, dims=2) - X)**2
    output =  0.5*(1-tau)*torch.norm(X)**2 + 0.5*tau*alpha*temp \
            + (1./4)*tau*lamb*torch.sum(X**4)
    return -output

def log_prob_func_norm(params):
    mean = torch.tensor([0.])
    stddev = torch.tensor([1.])
    return torch.distributions.Normal(mean, stddev).log_prob(params).sum()

def log_prob_func(x):
    return -torch.norm(x)**4/4

num_samples = 10**6
burn=10000
step_size = 0.001
num_steps_per_sample = 200
threshold = 1e-3
softabs_const=10**6

hamiltorch.set_random_seed(111)
params_init = torch.zeros(d)
params_hmc = hamiltorch.sample(
    log_prob_func=potential, 
    params_init=params_init,  
    num_samples=num_samples, 
    burn=burn,
    step_size=step_size, 
    num_steps_per_sample=num_steps_per_sample, 
    fixed_point_threshold=threshold,
    jitter=0.01,
    softabs_const=softabs_const,
    sampler=hamiltorch.Sampler.HMC,
    integrator=hamiltorch.Integrator.IMPLICIT
    )

samples_hamiltonian = np.array(params_hmc[burn:])

samples_hamiltonian_norm  = []

for i in tqdm(range(0, len(params_hmc))):
    samples_hamiltonian_norm.append(np.linalg.norm(params_hmc[i].numpy()))

samples_hamiltonian_norm = np.array(samples_hamiltonian_norm).astype(float)

print(np.mean(samples_hamiltonian_norm**2))
print(np.mean(samples_hamiltonian_norm**4))
print(np.mean(samples_hamiltonian_norm**6))

coord_draw = 20

plt.figure()
plt.hist(samples_hamiltonian[:,coord_draw], bins=100)

# plot trajectory of some coordinate 
sel1 = 0
sel2 = num_samples-burn

plt.figure()
plt.plot(np.arange(sel1, sel2), samples_hamiltonian[sel1:sel2, coord_draw])
plt.title("Metropolis")



