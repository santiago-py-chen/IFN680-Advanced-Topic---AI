'''

Implementation: Differential Evolution


'''

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn import model_selection

import pandas as pd

# ----------------------------------------------------------------------------

def differential_evolution(fobj, 
                           bounds, 
                           mut=2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = True):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    n_dimensions = len(bounds) # dimension of the input space of 'fobj'
    #    This generates our initial population of 10 random vectors. 
    #    Each component x[i] is normalized between [0, 1]. 
    #    We will use the bounds to denormalize each component only for 
    #    evaluating them with fobj.
    

    # Generate initial population of 10 random vectors
    # Create normalized (between [0,1]) population
    pop = np.random.rand(popsize, n_dimensions)
    
    # Denormalize the population
    min_bound, max_bound = np.asarray(bounds).T
    delta = np.fabs(max_bound - min_bound)
    pop_denorm = min_bound + pop*delta
    
    # Calculate the cost of the population
    cost = np.asarray([fobj(w) for w in pop_denorm])
    
    # Convert the cost to inf if the cost ends up with nan which may mess up with 
    # retrieving the smallest cost in the next step
    for i, c in enumerate(cost):  
        if np.isnan(c):
            cost[i] = np.inf     
    
    # Store the best cost and underlying index
    best_idx = np.argmin(cost)
    best = pop_denorm[best_idx]
    
    if verbose:
        print(
        '** Lowest cost in initial population = {} '
        .format(cost[best_idx]))        
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i))        
            
        for j in range(popsize):
            
            # Select three disctinct points a, b, c != pop[j]
            index_range = [idx for idx in range(popsize) if idx != j]
            index = np.random.choice(index_range, 3, replace  = False)
            a = pop[index[0]]
            b = pop[index[1]]
            c = pop[index[2]]
            
            
            # Generate mutation and clip the mutation with (0,1) 
            mutant = a + mut*(b - c)
            mutant = np.clip(mutant.copy(), 0 ,1)
                        
            # Create a list collecting the element of the trial vector 
            # Assigning the value of each entry according to given probability crossp/1-crossp
            trial = []
            for k in range(n_dimensions):
                entry = np.random.choice(a=[mutant[k], pop[j][k]], size = 1, p = [crossp,1-crossp])
                trial.append(entry)
            
            # Converting the trial vector from list to 1-D numpy array
            trial = np.asarray(trial)
            trial = trial.reshape(n_dimensions,)
            
            # Denormalize the trial vector for cost calculation
            trial_denorm = min_bound + trial*delta
            trial_denorm = np.asarray(trial_denorm)
            
            # Calculate the cost   
            cost_of_trial = fobj(trial_denorm)
            
            # Convert the cost of trial to inf 
            if np.isnan(cost_of_trial):
                cost_of_trial = np.inf
            
            # Update the cost and the underlying candidate in the population if
            # the cost of the trial vector is better
            if cost_of_trial < cost[j]:
                cost[j] = cost_of_trial.copy()
                pop[j] = trial.copy()
                print("cost_updated")
                
                # Update the best cost and underlying index if the cost of trial
                # vector is better than the current best record
                if cost_of_trial < cost[best_idx]:
                    best_idx = j
                    best = trial_denorm.copy()
                    print('best_w_updated')
            
            # yield the best trial vector and best cost 
        yield best, cost[best_idx]

# ----------------------------------------------------------------------------

def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate. 
    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def fmodel(x, w):
        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).
        '''
        
        # instantiate y as ND-zero-array according the dimension of x
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            assert type(x) is np.ndarray
            y = np.zeros_like(x)
            
        'INSERT MISSING CODE HERE'
        # construct y as the form of w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        for i in reversed(range(0, len(w))):
            y = w[i] + y*x
    
        return y
  
    
    

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        Y_pred = fmodel(X, w)
        # Return rmse
        return np.sqrt(sum((Y - Y_pred)**2)/len(Y)) 
        'INSERT MISSING CODE HERE'


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5

    # Loop over the DE generator
    for i , p in enumerate(de_gen):
        
        # w : best solution so far
        # c_w : cost of w
        w, c_w = p
        
        # Stop when solution cost is less than the target cost
        if c_w < target_cost :

            break
        
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
    #    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000))    
    #    w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()    
    

# ----------------------------------------------------------------------------

def task_2():
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        # Representing nh1, nh2, alpha, learning rate with the candidate w in population
        # Convert nh1, nh2 to int >= 1           
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1 number of neurons in the first hidden layer
                int(1+w[1]), # nh2 number of neurons in the second hidden layer
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )

        # Instantiate MLP classifier
        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        # Train/fit the MLP classifier with training data
        clf.fit(X_train_transformed, y_train)
               

        # Compute the accurary on the test set
        mean_accuracy = clf.score(X_test_transformed, y_test)
        
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    
    # Split the dataset into train/test with the fraction of 0.6/0.4
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'  
    scaler = preprocessing.StandardScaler().fit(X_train)
    
    # Transform the training set with the scaler initiated above
    X_train_transformed = scaler.transform(X_train)
    

    # Transform the training set with the scaler initiated above
    X_test_transformed =  scaler.transform(X_test)
    
    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    # Define DE generator
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=10, 
            maxiter=20,
            verbose=True)
    
    # Loop over the DE generator, print number of generation and the best cost, 
    # Break the loop till reach maxiter or when accuracy reaches 0.9
    for i, p in enumerate(de_gen):
        
        # w : best solution so far
        # c_w : cost of w
        w, c_w = p
        'INSERT MISSING CODE HERE'
        
        # Print the number of generation and the best cost
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
        if abs(c_w) > 0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    
# ----------------------------------------------------------------------------

def experiment(threshold):
    
    '''
    Goal : set up experiments to test the performance of DE using different set
           of (pop_size, max_iteration) to tune the hyper-parameter of the 
           MLPClassifier
    
    Record the number of iteration and cost of each set of 
    (pop_size, max_iteration) in csv files for further statistical test as 
    described in the report. 
    
    @param:
            threshold: a threshold accuracy to test the performance of DE using
            different set of parameters
    
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        # Representing nh1, nh2, alpha, learning rate with the candidate w in population
        # Convert nh1, nh2 to int >= 1           
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1 number of neurons in the first layer
                int(1+w[1]), # nh2 number of neurons in the second layer
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )

        # Instantiate MLP classifier
        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=50, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        # Train/fit the MLP classifier with training data
        clf.fit(X_train_transformed, y_train)
               

        # Compute the accurary on the test set
        mean_accuracy = clf.score(X_test_transformed, y_test)
        
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    
    # Split the dataset into train/test with the fraction of 0.6/0.4
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'  
    scaler = preprocessing.StandardScaler().fit(X_train)
    
    # Transform the training set with the scaler initiated above
    X_train_transformed = scaler.transform(X_train)
    
    # Transform the training set with the scaler initiated above
    X_test_transformed =  scaler.transform(X_test)
    
    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    
    def grid_search(parameter):
        '''
        Grid search the given set of parameter and identify the optimized one, 
        store the result (cost) of each given candidate then return a list of 
        result:  (number_of_iteration, 
                  cost, 
                  NH1, 
                  NH2, 
                  alpha, 
                  learning_rate, 
                  pop_size, 
                  max_iteration)
        
        @param:
            parameter: A list of (pop_size, max_iteration) candidate
        
        '''
        
        # Unpack the pop_size and max_iteration
        pop_size, max_iteration = parameter
        
        # Define DE generator
        de_gen = differential_evolution(
                eval_hyper, 
                bounds, 
                mut = 1,
                popsize=pop_size, 
                maxiter=max_iteration,
                verbose=True)
        
        # Loop over the DE generator, print no. generation and the best cost, 
        # Break the loop till reach maxiter or when accuracy reach 0.9
        for i, p in enumerate(de_gen):
            # w : best solution so far
            # c_w : cost of w
            w, c_w = p
            
            print('Generation {},  best cost {}'.format(i,abs(c_w)))
            # Stop if the accuracy is above 90%
            if abs(c_w) > threshold:
                break
            
     
        # Print the search result
        print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
        print('Hyperparameters found:')
        print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
        print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
        
        # Store the result (cost) of each (pop_size, max_iteration) candidate
        # in a list
        candidate_result = i+1, c_w, int(1+w[0]), int(1+w[1]), 10**w[2], 10**w[3], pop_size, max_iteration
        result.append(candidate_result)
            
    
    # (Pop_size, max_iteration) candidates
    parameters = [(5,40), (10,20),(20,10),(40,5)]
    
    # Initialize an empty list to store the result
    result = []
    
    # Grid search the best candidate and store the result in a list
    for parameter in parameters:
        grid_search(parameter)
    
    # Print the result
    for cost in result:
        print("--------------------------------------------------------------")
        print("Pop_size: ", cost[6])
        print("Max_iteration: ", cost[7])
        print("Iterations: ", cost[0])
        print("Cost: ", cost[1])
        print("NH1: ", cost[2])
        print("NH2: ", cost[3])
        print("Alpha: ", cost[4])
        print("Learning rate: ", cost[5])
        print("--------------------------------------------------------------")        
    return result
    
def task_3():
    """
    Run experiments, record the result for further analysis and visualization    
    
    """
    threshold = [0.86, 0.87, 0.88, 0.89]
    for t in threshold: 
        # Create a list to store the number of iteration that DE converge 
        # @ given threshold for p1(5, 40), p2(10, 20), p3(20, 10), p4(40, 5)    
        iter_p1 = []
        iter_p2 = [] 
        iter_p3 = [] 
        iter_p4 = []
        
        # Create a list to store the cost at the end of the DE  
        # p1(5, 40), p2(10, 20), p3(20, 10), p4(40, 5)
        cost_p1 = []
        cost_p2 = []
        cost_p3 = []
        cost_p4 = []      
    
        # Run the experiment and record the result for the given threshold
        experiment(t)
        
        # Loop over experiment and record the number of iteration of each set of param
        # for 30 times
        for i in range(30):
            record = experiment(t) # replace the argument with the testing threshold
            iter_p1.append(record[0][0])
            iter_p2.append(record[1][0])
            iter_p3.append(record[2][0])
            iter_p4.append(record[3][0])
            
            cost_p1.append(record[0][1])
            cost_p2.append(record[1][1])
            cost_p3.append(record[2][1])
            cost_p4.append(record[3][1])
        
        # Convert the result into a dictionary then transform it to a pandas DataFrame
        iteration_dict = {"iteration(5,40)":iter_p1, "iteration(10,20)":iter_p2, 
                       "iteration(20,10)": iter_p3, "iteration(40,5)": iter_p4}
        cost_dict = {"cost(5,40)": cost_p1, "cost(10,20)": cost_p2, 
                       "cost(20,10)": cost_p3, "cost(40,5)":cost_p4}
        df_iteration = pd.DataFrame.from_dict(iteration_dict)    
        df_iteration.to_csv("iteration_" + str(t) + ".csv")
        
        df_cost = pd.DataFrame.from_dict(cost_dict)
        df_cost.to_csv("cost_" + str(t) + ".csv")
    




# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
    task_1()    
    task_2()
    # task_3() is gonna take a long while due to this experiment run 30 execution
    # of DE using the given 4 sets of pop_size and max_iter, pls be noted before
    # execution
    task_3() 

