import types
class Optimizer(object):
    def __init__(self, model, defaults):
        self.defaults = defaults
        self.model = model
    
    def zero_grad(self):
        for param_dict in self.model.param():
            if isinstance(param_dict, types.GeneratorType): 
                param_dict = next(param_dict)
            for p in param_dict.values():
                if p.grad is not None: p.set_grad_zero()
                
    def step(self, closure):
        raise NotImplementedError
        
        
class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        defaults = dict(lr=lr)
        self.lr = lr
        super(SGD, self).__init__(model, defaults)
        
    def step(self, closure):
        loss= None
        if closure is not None:
            loss = closure    
        for param_dict in self.model.param():
            if isinstance(param_dict, types.GeneratorType):
                param_dict = next(param_dict)
            for p in param_dict.values():
                if p.grad is None:
                    continue
                d_p = p.grad
                p.data -= self.lr*d_p      
        return loss       