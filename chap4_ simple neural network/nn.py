import numpy as np

class Matmul:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem={'x': x, 'W':W}
        return h
    
    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']
        
        ####################
        '''计算矩阵乘法的对应的梯度'''
        ####################
        grad_x = np.matmul(grad_y,W.T)
        grad_W = np.matmul(x.T,grad_y)
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x']=x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        ####################
        x = self.mem['x']
        grad_x = np.where(x>0,grad_y,np.zeros_like(x))
        return grad_x


class Softmax:
    '''
    softmax over last dimension
    '''
    def __init__(self):
        self.epsilon = np.finfo(np.float64).eps
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        # 增加数值稳定性
        x = x - x.max(axis=1,keepdims=True)
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp/np.clip(partition,self.epsilon,None)
        
        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem["out"]
        return s*(grad_y - (s*grad_y).sum(axis=1,keepdims=True))


class Log:
    '''
    softmax over last dimension
    '''
    def __init__(self):
        self.epsilon = np.finfo(np.float64).eps
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(np.clip(x,self.epsilon,None))
        
        self.mem['x'] = x
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']
        
        return 1./np.clip(x,self.epsilon,None) * grad_y