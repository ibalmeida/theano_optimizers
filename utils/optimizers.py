import theano
import theano.tensor as T

def sgd(cost, params, learning_rate=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append((param, param - learning_rate * grad))
    return updates

def rmsprop(cost, params, learning_rate=1e-3, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        grad_avg = theano.shared(param.get_value() * 0.)
        grad_avg_new = rho * grad_avg + (1 - rho) * grad**2

        grad_scaling = T.sqrt(grad_avg_new + epsilon)
        grad = grad / grad_scaling

        updates.append((grad_avg, grad_avg_new))
        updates.append((param, param - learning_rate * grad))
    return updates

def adagrad(cost, params, learning_rate=1e-3, epsilon=1e-8):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        grad_square = theano.shared(param.get_value() * 0.)
        grad_square_new = grad_square + grad**2

        updates.append((grad_square, grad_square_new))
        updates.append((
                param,
                param - learning_rate * grad / T.sqrt(grad_square_new + epsilon)
        ))
    return updates

def adam2(cost, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-3):
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    t = theano.shared(0)
    t_new = t + 1
    updates.append((t, t_new))

    for param, grad in zip(params, grads):
        first_moment = theano.shared(param.get_value() * 0.)
        first_moment_new = beta1 * first_moment + (1 - beta1) * grad
        first_moment_new = first_moment_new / (1 - beta1**t_new)

        second_moment = theano.shared(param.get_value() * 0.)
        second_moment_new = beta2 * second_moment + (1 - beta2) * grad**2
        second_moment_new = second_moment_new / (1 - beta2**t_new)

        updates.append((first_moment, first_moment_new))
        updates.append((second_moment, second_moment_new))
        updates.append((
            param,
            param - learning_rate * first_moment_new / T.sqrt(second_moment_new + epsilon)
        ))
    return updates

def adam(cost, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    t = theano.shared(0)
    t_new = t + 1
    updates.append((t, t_new))

    for param, grad in zip(params, grads):
        first_moment = theano.shared(param.get_value() * 0.)
        first_moment_new = beta1 * first_moment + (1 - beta1) * grad

        second_moment = theano.shared(param.get_value() * 0.)
        second_moment_new = beta2 * second_moment + (1 - beta2) * grad**2

        learning_rate_norm = learning_rate * T.sqrt(1 - beta2**t_new) / (1 - beta1**t_new)

        updates.append((first_moment, first_moment_new))
        updates.append((second_moment, second_moment_new))
        updates.append((
            param,
            param - learning_rate_norm * first_moment_new / T.sqrt(second_moment_new + epsilon)
        ))
    return updates
