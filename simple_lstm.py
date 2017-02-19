#encoding: utf-8
'''
@author: huangjin (Jeff)
@email: hj51yc@gmail.com
using LSTM to do char-rnn job
'''

import sys, os
import numpy as np
np.seterr(all='raise')

def sigmoid(a):
    return 1.0/(1+np.exp(-a))

def dsigmoid(a):
    return a*(1-a)

def tanh(a):
    return np.tanh(a)

def dtanh(a):
    return 1 - (a ** 2)

def softmax(y):
    p = np.exp(y)
    s = np.sum(p)
    final = p / s
    return final


def cross_entropy(prob, y_true):
    log_prob_neg = np.log(1 - prob)
    log_prob = np.log(prob)
    y_true_neg = 1 - y_true
    return -(np.sum(log_prob_neg * y_true_neg) + np.sum(log_prob * y_true))


##simple LSTM: N blocks with one cell in every block!
class LSTM(object):

    def __init__(self, x_dim, hidden_num, output_dim, eta, epsilon):
        self.eta = eta
        self.epsilon = epsilon
        self.adagrads_sum = {}

        ## concate input as [h, x]
        Z = x_dim + hidden_num
        H = hidden_num
        D = output_dim
        self.Z = Z
        self.H = H
        self.D = D
        self.Wi = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bi = np.zeros((1, H))
        self.Wf = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bf = np.zeros((1, H))
        self.Wo = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bo = np.zeros((1, H))
        self.Wc = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bc = np.zeros((1, H))

        self.Wy = np.random.randn(H, D) / np.sqrt(D / 2.0)
        self.by = np.zeros((1, D))

    
    def greedy_forward(self, x_seq_start, state_init, stop_indexes, stop_len):
        state = state_init
        for x in x_seq_start:
            prob, state, cache = self.forward(x, state)
        
        gen_indexes = []
        for k in xrange(stop_len):
            #index = np.random.choice(range(len(prob[0])), p=prob.ravel())
            #print 'prob', prob
            index = np.argmax(prob[0])
            gen_x = np.zeros(len(prob[0]))
            gen_x[index] = 1
            gen_indexes.append(index)
            if index in stop_indexes:
                break
            prob, state, cache = self.forward(gen_x, state)
        return gen_indexes


    def forward(self, x, state):
        h_prev, c_prev = state
        
        #X = np.column_stack((h_prev, x))
        #print 'x:', x
        #print 'h_prev:', h_prev
        X = np.hstack((h_prev, x.reshape(1, len(x))))
       
        hi = sigmoid(np.dot(X, self.Wi) + self.bi)
        hf = sigmoid(np.dot(X, self.Wf) + self.bf)
        ho = sigmoid(np.dot(X, self.Wo) + self.bo)
        
        hc = tanh(np.dot(X, self.Wc) + self.bc)
        c = hf * c_prev + hi * hc
        h = ho * tanh(c)

        y = np.dot(h, self.Wy) + self.by
        prob = softmax(y)
        cache = (hi, hf, ho, hc, h, c, y, c_prev, h_prev, X)
        state = (h, c)
        return prob, state, cache

    def grad_clip(self, dw, rescale=5.0):
        norm = np.sum(np.abs(dw))
        if norm > rescale:
            return dw * (rescale / norm)
        else:
            return dw

    def backward(self, prob, y_label, d_next, cache):
        hi, hf, ho, hc, h, c, y, c_prev, h_prev, X = cache
        dh_next, dc_next = d_next

        ## softmax loss gradient
        dy = prob.copy()

        y_index = np.argmax(y_label)
        dy[0, y_index] -= 1
        #print 'the dy:', dy
        
        dWy = np.dot(h.T, dy)
        dby = dy
        #print 'the dWy', dWy

        # Note we're adding dh_next here, because h is forward in next_step and make output y here: h is splited here!
        dh = np.dot(dy, self.Wy.T) + dh_next
        #print 'the dh', dh
        
        dho = tanh(c) * dh
        dho = dsigmoid(ho) * dho
        #print 'the dho', dho
        
        # Gradient for c in h = ho * tanh(c), note we're adding dc_next here! 
        #dc = ho * dh + dc_next
        #dc = dtanh(c) * dc
        
        ## i change dc below
        dc = dh * ho * dtanh(c) + dc_next
        #print 'the dc', dc

        dhc = hi * dc
        dhc = dhc * dtanh(hc)
        #print 'the dhc', dhc

        dhf = c_prev * dc
        dhf = dsigmoid(hf) * dhf
        #print 'the dhf', dhf

        dhi = hc * dc
        dhi = dsigmoid(hi) * dhi
        #print 'the dhi', dhi

        dWf = np.dot(X.T , dhf)
        dbf = dhf
        dXf = np.dot(dhf, self.Wf.T)
        #print 'the X.T', X.T
        #print 'the dWf', dWf
        #print 'the dbf', dbf

        dWi = np.dot(X.T, dhi)
        dbi = dhi
        dXi = np.dot(dhi, self.Wi.T)

        dWo = np.dot(X.T, dho)
        dbo = dho
        dXo = np.dot(dho, self.Wo.T)

        dWc = np.dot(X.T, dhc)
        dbc = dhc
        dXc = np.dot(dbc, self.Wc.T)

        dX = dXf + dXi + dXo + dXc
        new_dh_next = dX[:, :self.H]
        new_dh_next = self.grad_clip(new_dh_next)
        #print "the dh_next", new_dh_next

        # Gradient for c_old in c = hf * c_old + hi * hc
        new_dc_next = hf * dc
        new_dc_next = self.grad_clip(new_dc_next)
        #print 'the dc_next', new_dc_next

        grad = dict(Wf=dWf, Wi=dWi, Wo=dWo, Wc=dWc, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
        for key in grad:
            grad[key] = self.grad_clip(grad[key])
        new_d_next = (new_dh_next, new_dc_next) 

        return new_d_next, grad

    

    def train_step(self, x_seq, y_seq, state):
        probs = []
        caches = []
        loss = 0.0
        h, c = state

        for x, y in zip(x_seq, y_seq):
            prob, state, cache = self.forward(x, state)
            probs.append(prob)
            caches.append(cache)
            loss += cross_entropy(prob, y)
            
        loss /= len(x_seq[0])
        
        d_next = (np.zeros_like(h), np.zeros_like(c))
        grads = {}

        for prob, y_true, cache in reversed(list(zip(probs, y_seq, caches))):
            d_next, cur_grads = self.backward(prob, y_true, d_next, cache)
            for key in cur_grads:
                if key not in grads:
                    grads[key] = cur_grads[key]
                else:
                    grads[key] += cur_grads[key]

        return grads, loss, state

    
    def adagrad(self, grads, eta, epsilon):
        
        for w_name in grads:
            W = getattr(self, w_name)
            dW = grads[w_name]
            try:
                square_dW = dW * dW
            except:
                print 'overflow dW', w_name
                print 'overflow dW', dW
                for t in grads:
                    print 'w', t, getattr(self, t)
                    print 'dw', t, grads[t]
                raise
            if w_name in self.adagrads_sum:
                self.adagrads_sum[w_name] += square_dW
            else:
                self.adagrads_sum[w_name] = square_dW + epsilon

            W_step =  eta / np.sqrt(self.adagrads_sum[w_name])
            W -= W_step * dW


    def train_once(self, x_seq, y_seq, state):
        grads, loss, state = self.train_step(x_seq, y_seq, state)
        self.adagrad(grads, self.eta, self.epsilon)
        return loss, state

