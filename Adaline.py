"""Adaptive linear neuron (Adaline) model for supervised machine learning.

Notes
-----
    This script is version v0. It provides the base for all subsequent
    iterations of the project.

Requirements
------------
    See "requirements.txt"
"""

#%% import libraries and modules
import numpy as np  
import random
import matplotlib.pyplot as plt
import os

#%% figure parameters
plt.rcParams['figure.figsize'] = (7,5)
plt.rcParams['font.size']= 20
plt.rcParams['lines.linewidth'] = 5

#%%
class Adaline:
    """Adaline class."""
    
    def __init__(self,
                 num_prototypes=3, input_size=9, activation_threshold=0.5,
                 learning_rate=0.01, min_training_error=0.001,
                 num_repetitions=1000,
                 min_noise_level=0, max_noise_level=5, noise_level_increment=0.1):
        self.num_prototypes = num_prototypes
        self.input_size = input_size
        self.activation_threshold = activation_threshold
        self.learning_rate = learning_rate
        self.min_training_error = min_training_error
        self.num_repetitions = num_repetitions
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.noise_level_increment = noise_level_increment
        
    def make_inputs(self):
        """Create input patterns."""
        input_patterns = np.zeros([self.num_prototypes, self.input_size])
        for prototype_index in range(self.num_prototypes):
            input_patterns[prototype_index,::prototype_index+self.num_prototypes-1] = 1
        
        return input_patterns
    
    def make_targets(self):
        """Create target patterns."""
        target_patterns = np.eye(self.num_prototypes)
        
        return target_patterns
    
    def step_function(self, activation):
        """Apply step function as activation function."""
        activation[activation >= self.activation_threshold] = 1 # if activation >= threshold, give 1
        activation[activation < self.activation_threshold] = 0 # if activation < threshold, give 0
        
        return activation
    
    def initialize_connection_weights(self):
        """Initialize connection weights."""
        weights = np.zeros([self.num_prototypes, self.input_size])
        
        return weights
    
    def train_model(self, input_patterns, target_patterns):
        """Train the model to learn associations between input patterns
           and corresponding target patterns."""
        # create empty list for storing mean squared errors
        mean_squared_error_list = []                                                
        # set place holder array for squared error computation
        squared_error_list = np.ones(self.num_prototypes)                                
        # initialize connection weights
        weights = Adaline(self.num_prototypes, self.input_size).initialize_connection_weights()
        # initialize iteration index
        iteration_index = 0
        
        while np.mean(squared_error_list) > self.min_training_error:
            # generation list of random samples                                        
            random_sample = random.sample(range(self.num_prototypes), self.num_prototypes)      
            # initialize sample index
            sample_index = 0                                                        
            
            while sample_index < len(random_sample):      
                # random selection of input pattern                                         
                input_vector = input_patterns[random_sample[sample_index]]
                # random selection of corresponding target pattern
                actual_target = target_patterns[random_sample[sample_index]]
                # compute activation
                obtained_target = np.dot(weights, input_vector)
                # compute error
                error = actual_target - obtained_target
                # compute squared error
                squared_error_list[sample_index] = np.dot(error, error)
                # update connection weights
                weights += self.learning_rate * np.transpose([error]) * input_vector
                # increment sample index
                sample_index += 1
            
            # compute mean squared error
            mean_squared_error_list.append(np.mean(squared_error_list))
            # increment iteration index
            iteration_index +=1
        
        return weights, mean_squared_error_list

    def test_model(self, input_patterns, target_patterns, weights):
        """Test model performance over input patterns with added noise."""
        # specify noise levels
        noise_levels = np.arange(self.min_noise_level, self.max_noise_level, self.noise_level_increment)                     
        # preallocate squared error performance
        squared_error_performance = np.zeros(self.num_prototypes)
        # create empty list for storing recall performance
        recall_performance = []
        
        for noise_index in range(len(noise_levels)):
            # create empty list for storing mean squared error
            mean_squared_error_performance = []
             
            for repetition_count in range(self.num_repetitions):
                # specify uniform noise level
                noise_vector = np.random.uniform(0, noise_levels[noise_index], self.input_size)
                # generate list of random samples
                random_sample = random.sample(range(self.num_prototypes), self.num_prototypes)
                # initialize sample index
                sample_index = 0
                
                while sample_index < len(random_sample): 
                    # random selection of noisy input pattern                                    
                    input_vector = input_patterns[random_sample[sample_index]] + noise_vector
                    # rescale noisy input between 0 and 1
                    input_vector = input_vector/max(input_vector)
                    # compute activation
                    activation = np.dot(weights, input_vector)
                    # apply step function
                    obtained_target = Adaline(self.activation_threshold).step_function(activation)
                    # random selection of corresponding target pattern
                    actual_target = target_patterns[random_sample[sample_index]]
                    # compute error
                    error = actual_target - obtained_target
                    # compute squared error 
                    squared_error_performance[sample_index] = np.dot(error, error)
                    # increment sample index
                    sample_index += 1
                
                # specify squared errors less than 1
                squared_error_performance = np.double(squared_error_performance < 1)
                # compute mean squared error 
                mean_squared_error_performance.append(np.mean(squared_error_performance))
            
            # compute recall performance in percentage
            recall_performance.append(np.mean(mean_squared_error_performance) * 100)
        
        return noise_levels, recall_performance

#%% instantiate Adaline class

model = Adaline()

#%% create input and target patterns

input_patterns = model.make_inputs()
target_patterns = model.make_targets()

#%% train and test Adaline model

weights, mseList = model.train_model(input_patterns, target_patterns)
noise_levels, recall_performance = model.test_model(input_patterns, target_patterns, weights)

#%% plot figures

cwd = os.getcwd()                                                               # get current working directory
fileName = 'images'                                                             # specify filename

# filepath and directory specifications
if os.path.exists(os.path.join(cwd, fileName)) == False:                        # if path does not exist
    os.makedirs(fileName)                                                       # create directory with specified filename
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory
else:
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory

# figure 1 - input and target patterns
fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0,0].imshow(input_patterns[0].reshape(int(np.sqrt(model.input_size)), int(np.sqrt(model.input_size))))
ax[0,0].axis('off')

ax[0,1].imshow(input_patterns[1].reshape(int(np.sqrt(model.input_size)), int(np.sqrt(model.input_size))))
ax[0,1].axis('off')
ax[0,1].set_title('Input patterns')

ax[0,2].imshow(input_patterns[2].reshape(int(np.sqrt(model.input_size)), int(np.sqrt(model.input_size))))
ax[0,2].axis('off')

ax[1,0].imshow([target_patterns[0]])
ax[1,0].axis('off')

ax[1,1].imshow([target_patterns[1]])
ax[1,1].axis('off')
ax[1,1].set_title('Target patterns')

ax[1,2].imshow([target_patterns[2]])
ax[1,2].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_1'))
    
# figure 2 - training phase
fig, ax = plt.subplots()
plt.plot(mseList, color='k')    
plt.xlabel('Epoch')
plt.ylabel('Mean squared error')
plt.title('Training')
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_2'))

# figure 3 - recall performance
fig, ax = plt.subplots()
plt.plot(noise_levels, recall_performance, color='k')
plt.xlabel('Noise level')
plt.ylabel('Recall performance (%)')    
plt.title('Recall')
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_3'))
