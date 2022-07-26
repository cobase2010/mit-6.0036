from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        s = self.start_state
        output = []
        for x in input_seq:
            # print("State: ", s)
            # print("X:", x)
            s = self.transition_fn(s, x)
            # print("New State:", s)
            y = self.output_fn(s)
            output.append(y)
            # print("Output: ", output)
        
        return output


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0, 0)

    def transition_fn(self, s, x):
        # Your code here
        s=s[0]
        for ele in x:
           s+=ele
        return [s//2,s%2]

    def output_fn(self, s):
        # Your code here
        return s[1]


class Reverser(SM):
    start_state = [None,[]]

    def transition_fn(self, s, x):
        # Your code here
        if s[0] is None and x!='end':
            s=[s[0],[x]+s[1]]
        
        elif x=='end':
            s=s[1]
        
        elif s[1:]:
            s=s[1:]
        else:
            s=[None,[]]
        
        # print(s,x)
        return s

    def output_fn(self, s):
        # Your code here
        # print(s[0])
        return s[0]


class RNN(SM):
    start_state = None
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx=Wsx
        self.Wss=Wss
        self.Wo=Wo
        self.Wss_0=Wss_0
        self.Wo_0=Wo_0
        self.f1=f1
        self.f2=f2
        m, n = Wss_0.shape
        RNN.start_state = np.zeros((m, 1))
        

    def transition_fn(self, s, x):
        # Your code here
        # print(self.Wss)
        # print("s:",s)
        # print("x:", x)
        # Guarding against when x is just an int instead of array
        if isinstance(x, int):
            x = np.array([x])

        return self.f1(np.matmul(self.Wss, s) + np.matmul(self.Wsx, x) + self.Wss_0)

    def output_fn(self, s):
        # Your code here
        return self.f2(np.matmul(self.Wo, s) + self.Wo_0)
