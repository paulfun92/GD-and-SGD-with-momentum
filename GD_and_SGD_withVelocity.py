__author__ = 'paul'

'''
This program performs optimization through Gradient Descent (GD), Stochastic Gradient Descent (SGD), or both.
Besides that, it can perform the normal version, or the momentum-based version. For normal, enter momentum = 0
By default momentum is 0.3. Further, the user can specify the step-size (alpha), the maximum number of iterations,
Epsilon (stopping criterion), and the method used for optimization: 'GD', 'SGD', or 'both'.
Sample command: python GD_and_SGD_withVelocity.py --Method='GD' --Momentum=0.25 --Stepsize=0.05 --Epsilon=1e-6 --Maximum_Iterations=200
'''

import argparse
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import random
#from scan_perform.scan_perform import *

from nose.tools import (assert_equal,
                        assert_greater,
                        assert_greater_equal)

def parse_args():
    '''
    Parses command-line args and returns them as a namespace.
    '''

    parser = argparse.ArgumentParser(
        description=("This program optimizes a function with gradient descent"))

    def positive_int(arg):
        '''Arg checker for positive ints.'''
        arg = int(arg)
        assert_greater(arg, 0)

        return arg

    def positive_float(arg):
        '''Arg checker for positive floats.'''
        arg = float(arg)
        assert_greater(arg, 0)

        return arg

    # This checks if the method is entered correctly
    def correct_method(arg):
        ''' Arg checker for input of method choice'''
        arg = str(arg)
        if arg != 'GD' and arg != 'SGD' and arg != 'both':
            raise ValueError("Please enter a valid choice for optimization method: 'GD', 'SGD', or 'both'"
                             % arg)

        return arg

    def correct_momentum(arg):
        ''' Arg checker for input of method choice'''
        arg = float(arg)
        if arg < 0 or arg >= 1:
            raise ValueError("Please specify a momentum value in the range [0,1). 0 gives normal SG/SGD"
                             % arg)

        return arg


    parser.add_argument("--Stepsize",
                        type=positive_float,
                        default=0.06,
                        help=("The step size, alpha, of the Gradient Descent"))

    parser.add_argument("--Maximum_Iterations",
                        type=positive_int,
                        default=300,
                        help=("After this number of iterations, it stops"))

    parser.add_argument("--Epsilon",
                        type=positive_float,
                        default=1e-6,
                        help="When the norm of the gradient is smaller than epsilon, we consider it small enough and stop")

    parser.add_argument("--Method",
                        type=correct_method,
                        default='both',
                        help="Type here the method you want to use to optimize the function: 'GD', 'SGD', or 'both'")

    parser.add_argument("--Momentum",
                        type=correct_momentum,
                        default=0.2,
                        help="Type here the momentum in the range [0,1). 0 means normal GD/SGD")

    result = parser.parse_args()
    return result

def GD():      # THIS IS THE NEW FUNCTION THAT PERFORMS GRADIENT DESCENT

    print 'Gradient Descent: '

    args = parse_args()

    # Set initial parameters
    epsilon = args.Epsilon
    alpha = args.Stepsize
    max_iter = args.Maximum_Iterations
    momentum = args.Momentum

    # Set iteration to 0
    iter = 0

    # Starting point as as shared variable:
    xpoint = theano.shared(np.array([1.0,1.0]))

    # Define function f
    x = T.dvector('x')
    z = pow(x[0],4) + 2*pow(x[0],3) + 2*pow(x[0],2) + pow(x[1],2) - 2*x[0]*x[1]

    f = theano.function([x], z)

    #Set up the symbolic function to calculate the gradient (as a vector):
    gradf = T.grad(z,x)
    gradientf = theano.function([x],gradf) # Returns the gradient vector at any point x

    # Define the symbolic input for the step function:
    velocity = T.dvector('velocity')    # For momentum based GD

    # Create initial log to save variables x, objective value f, the gradient, the norm of the gradient, and the velocity after each iteration for printing and/or inspection later on:
    #Calculate the first log entries before starting the loop:
    x_log = np.array([xpoint.get_value()])
    f_log = np.array([f(x_log[iter])])
    gradient_log = np.array([gradientf(x_log[iter])])
    normGradient_log = np.array([np.linalg.norm(gradient_log[iter])])
    velocity_log = np.empty(shape=[0, xpoint.get_value().shape[0]])  # For velocity, we start with an empty array, as it lags 1 iteration behind the rest.The velocity for iteration 0 will automatically be added in the loop.

    # Define the function to take a step in theano, using shared variable, it updates x as a function of the velocity:
    # Here, velocity = momentum *velocity(iter-1) + alpha*gradient(iter)
    takeStep = theano.function([velocity], xpoint, updates=[(xpoint,xpoint-velocity)]) # If momentum=0, we have normal GD.

    # Now start the algorithm main loop for optimization:
    while iter<max_iter and normGradient_log[iter]>epsilon:

        # Calculate the velocity:
        if iter == 0: # for the first iterate, velocity is just alpha*gradient
            currentVelocity = alpha*gradient_log[iter]
        else:
            currentVelocity = momentum*velocity_log[iter-1] + alpha*gradient_log[iter]  # Definition of velocity (see earlier)

        takeStep(currentVelocity)   # Now take a step

        # Update log of the velocity:
        velocity_log = np.append(velocity_log, [currentVelocity], axis=0)

        # Increase the iteration counter and calculate the new gradient, objective f, and norm gradient for the next iteration, update logs:
        iter += 1
        x_log = np.append(x_log,[xpoint.get_value()],axis=0)    # Update new x-point to the log
        f_log = np.append(f_log,[f(x_log[iter])],axis=0)        # Update new objective value to the log
        gradient_log = np.append(gradient_log,[gradientf(x_log[iter])],axis=0)  # Calculate next gradient
        normGradient_log = np.append(normGradient_log, [np.linalg.norm(gradient_log[iter])], axis=0)  # Calculate next norm gradient

        # Print information about the steps to the screen:
        print 'The new point is (',x_log[iter,0],',',x_log[iter,1] ,')'
        print 'The new f is ',f_log[iter]
        print 'The new gradient is (',gradient_log[iter,0],',',gradient_log[iter,1],')'
        print 'The new norm of the gradient is ',normGradient_log[iter]
        print ''


    # Plot the iterates
    x1 = np.linspace(-0.7,1.1,180)
    y1 = np.linspace(-0.7,1.2,180)

    xv, yv = np.meshgrid(x1, y1)

    def fplot(a,b):
        return pow(a,4) + 2*pow(a,3) + 2*pow(a,2) + pow(b,2) - 2*a*b

    func = fplot(xv,yv)

    # If both methods are use, wait with plotting and plot both results in one figure. Return all necessary variables back to the main function for central plotting
    if args.Method == 'both':
        return xv, yv, func, x_log
    else:   # Otherwise plot it right away and return nothing
        plt.contourf(xv, yv, func, 40, alpha=0.75, cmap='jet')
        plt.plot(x_log[:,0],x_log[:,1],'k')
        plt.plot(x_log[:,0],x_log[:,1],'.k')
        plt.title('GD with alpha is %.2f, momentum is %.2f and max. # of iterations is %d:' % (args.Stepsize, args.Momentum, args.Maximum_Iterations))
        plt.show()


def SGD():      # THIS IS THE NEW FUNCTION THAT PERFORMS STOCHASTIC GRADIENT DESCENT

    print 'Stochastic Gradient Descent: '

    args = parse_args()

    # Set initial parameters
    epsilon = args.Epsilon
    alpha = args.Stepsize
    max_iter = args.Maximum_Iterations
    momentum = args.Momentum

    # Set iteration to 0
    iter = 0

    # Starting point as as shared variable:
    xpoint = theano.shared(np.array([1.0,1.0]))

    # Define several functions that will be used later on:
    x = T.dvector('x')  # x = [x1, x2] is the input vector
    z =  [pow(x[0], 4), 2 * pow(x[0], 3), 2 * pow(x[0], 2), pow(x[1], 2), -2 * x[0] * x[1]] # z = [f1, f2, f3, f4, f5], the components of the actual function.
    y = theano.function([x], z)         # y is a function/mapping from R^2 --> R^5, it consists of f1, f2, f3, f4, f5. Defining y makes it easier later to calculate all gradients (for f1,2,3,4,5) by calculating the Jacobian
    f = theano.function([x], np.sum(z)) # f is the actual function value, the sum of all sub-functions: f = f1+f2+f3+f4+f5. This is purely used for to update the log with the function value, not for the gradients.
    gradf = T.grad(np.sum(z),x)             # gradf is the gradient of the actual function, f
    gradientf = theano.function([x],gradf) # Returns the gradient of f at any point x

    # Set up Jacobian function of y:
    J, updates = theano.scan(lambda i,z,x : T.grad(z[i], x), sequences=T.arange(len(z)), non_sequences=[z,x])
    Jacobian = theano.function([x], J, updates=updates)

    # Define the symbolic input for the step function:
    velocity = T.dvector('velocity')    # For momentum based SGD

    # Create initial log to save variables x, objective value f, the gradient, and the norm of the gradient after each iteration for printing and/or inspection later on:
    #Calculate the first log entries before starting the loop:
    x_log = np.array([xpoint.get_value()])
    f_log = np.array([f(x_log[iter])])
    gradient_log = np.array([gradientf(x_log[iter])])
    normGradient_log = np.array([np.linalg.norm(gradient_log[iter])])
    velocity_log = np.empty(shape=[0, xpoint.get_value().shape[0]])  # For velocity, we start with an empty array, as it lags 1 iteration behind the rest.The velocity for iteration 0 will automatically be added in the loop.

    # Define the function to take a step in theano, using shared variable, it updates x as a function of the velocity:
    # Here, velocity = momentum *velocity(iter-1) + alpha*gradient(iter)
    takeStep = theano.function([velocity], xpoint, updates=[(xpoint,xpoint-velocity)]) # If momentum=0, we have normal SGD.

    # Now start the algorithm main loop for optimization:
    while iter<max_iter and normGradient_log[iter]>epsilon:

        # Calculate gradients of all 5 functions and put in an array:
        gradients = Jacobian(xpoint.get_value())
        random.shuffle(gradients)

        for i in range(gradients.shape[0]):

            print('ITERATION ' + str(iter + 1))

            if iter == 0: # for the first iterate, velocity is just alpha*gradient
                currentVelocity = alpha*gradients[i]
            else:
                currentVelocity = momentum*velocity_log[iter-1] + alpha*gradients[i]

            takeStep(currentVelocity)   # Now take a step

            # Update log of the velocity:
            velocity_log = np.append(velocity_log, [currentVelocity], axis=0)

            # Increase the iteration counter and calculate the new gradient, objective f, and norm gradient for the next iteration, update logs:
            iter += 1
            x_log = np.append(x_log,[xpoint.get_value()],axis=0)    # Update new x-point to the log
            f_log = np.append(f_log,[f(x_log[iter])],axis=0)        # Update new objective value to the log
            gradient_log = np.append(gradient_log,[gradientf(x_log[iter])],axis=0)  # Calculate next gradient
            normGradient_log = np.append(normGradient_log, [np.linalg.norm(gradient_log[iter])], axis=0)  # Calculate next norm gradient

            # Print information about the steps to the screen:
            print 'The new point is (',x_log[iter,0],',',x_log[iter,1] ,')'
            print 'The new f is ',f_log[iter]
            print 'The new gradient is (',gradient_log[iter,0],',',gradient_log[iter,1],')'
            print 'The new norm of the gradient is ',normGradient_log[iter]
            print ''

    # Plot the iterates
    x1 = np.linspace(-0.7,1.1,180)
    y1 = np.linspace(-0.7,1.2,180)

    xv, yv = np.meshgrid(x1, y1)

    def fplot(a,b):
        return pow(a,4) + 2*pow(a,3) + 2*pow(a,2) + pow(b,2) - 2*a*b

    func = fplot(xv,yv)

    # If both methods are use, wait with plotting and plot both results in one figure. Return all necessary variables back to the main function for central plotting
    if args.Method == 'both':
        return xv, yv, func, x_log
    else:   # Otherwise plot it right away. Don't return anything.
        plt.contourf(xv, yv, func, 40, alpha=0.75, cmap='jet')
        plt.plot(x_log[:,0],x_log[:,1],'k')
        plt.plot(x_log[:,0],x_log[:,1],'.k')
        plt.title('SGD with alpha is %.2f, momentum is %.2f and max. # of iterations is %d:' % (args.Stepsize, args.Momentum, args.Maximum_Iterations))
        plt.show()

def main(): # Main method

    args = parse_args() # Get the input values

    # Depending on which method the user has chosen, perform different actions
    if args.Method == 'GD':
        print 'Gradient Descent (GD) is now being performed, please wait for a while'
        GD()
    elif args.Method == 'SGD':
        print 'Stochastic Gradient Descent (SGD) is now being performed, please wait for a while'
        SGD()
    else:
        print 'Both, Gradient Descent (GD) and Stochastic Gradient Descent (SGD) are now being performed, please wait for a while'
        # Prepare a shared plot:

        fig = plt.figure(figsize=(12,5))
        fig.suptitle('For alpha is %.2f, momentum is %.2f and max. # of iterations is %d we have:' % (args.Stepsize, args.Momentum, args.Maximum_Iterations), fontsize=15, fontweight="bold")

        xv, yv, func, x_log = GD()  # Call the GD function for the first subplot, the necessary parameters are returned

        ax = plt.subplot("121")
        ax.set_title("Gradient Descent (GD):")
        ax.contourf(xv, yv, func, 40, alpha=0.75, cmap='jet')
        ax.plot(x_log[:,0],x_log[:,1],'k')
        ax.plot(x_log[:,0],x_log[:,1],'.k')

        xv, yv, func, x_log = SGD()  # Call the SGD function for the first subplot, the necessary parameters are returned

        ax = plt.subplot("122")
        ax.set_title("Stochastic Gradient Descent (SGD):")
        ax.contourf(xv, yv, func, 40, alpha=0.75, cmap='jet')
        ax.plot(x_log[:,0],x_log[:,1],'k')
        ax.plot(x_log[:,0],x_log[:,1],'.k')
        plt.subplots_adjust(top=0.84)
        plt.show()


if __name__ == '__main__':
    main()
