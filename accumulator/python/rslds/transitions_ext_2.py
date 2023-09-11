## ----------------------------------------
# this script was taken from the rslds package: https://github.com/davidzoltowski/ssmdm
# and modified by Elaheh Imani
# You need to clone the main version of the code from
# the above repository and copy this file into the originial project
##

from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.stats import dirichlet
from autograd import hessian

from ssm.util import one_hot, logistic, relu, rle, ensure_args_are_lists, LOG_EPS, DIV_EPS
from ssm.regression import fit_multiclass_logistic_regression, fit_negative_binomial_integer_r
from ssm.stats import multivariate_normal_logpdf
from ssm.optimizers import adam, bfgs, lbfgs, rmsprop, sgd


from ssm.transitions import Transitions, InputDrivenTransitions


class StickyRecurrentTransitions2(InputDrivenTransitions):
    """
    Generalization of the input driven HMM in which the observations serve as future inputs
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0, l2_penalty=1e-8, l1_penalty=1e-8):

        # Parameters linking past observations to state distribution
        self.Rs = np.zeros((K, D))
        self.l2_penalty=l2_penalty
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return super(StickyRecurrentTransitions2, self).params + (self.Rs,self.Ss)

    @params.setter
    def params(self, value):
        self.Rs = value[-2]
        super(StickyRecurrentTransitions2, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(StickyRecurrentTransitions2, self).permute(perm)
        self.Rs = self.Rs[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]

        # Past observations effect

        #Off diagonal elements of transition matrix (state switches), from past observations
        log_Ps = log_Ps + np.tile(np.dot(data[:-1], self.Rs.T)[:,None,:],(1, self.K, 1))
        log_Ps = log_Ps + np.ones((self.K,self.K))*self.r

        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True) #Normalize

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # SWAP?
                
                #Switching terms
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))
        return -1 * hess


    def log_prior(self):
        #L2 penalty
        lp = np.sum(-0.5 * self.l2_penalty * self.Rs**2)

        #L1 penalty
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))

        return lp

class RecurrentOnlyTransitionPop(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=40):
        super(RecurrentOnlyTransitionPop, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        self.Rs = npr.randn(K, D)
        self.r = npr.randn(K)

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = np.tile(log_Ps, (1, self.K, 1))

        # log_Ps=log_Ps + log_Ps_tmp
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        T, D = data.shape
        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class StickyRecurrentOnlyTransitionsRace(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=20):
        super(StickyRecurrentOnlyTransitionsRace, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        # self.Rs = np.zeros([K, D])
        self.scale=scale
        
        self.Rs = np.vstack((np.zeros(D), scale*np.eye(D)))
        self.r = np.vstack((np.zeros(1), -scale*np.ones(1), -scale*np.ones(1)))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        scale=self.scale
        T, D = data.shape
        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        self.Rs = np.vstack((np.zeros(D), scale*np.eye(D)))
        self.r = np.vstack((np.zeros(1), -scale*np.ones(1), -scale*np.ones(1)))
        
        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = np.tile(log_Ps, (1, self.K, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess
        # return np.zeros([T-1,D,D])

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class singleAccumulatorMultipop(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=40):
        super(singleAccumulatorMultipop, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        self.scale = scale

        self.Rs = np.vstack((np.zeros(D), self.scale/D*np.ones(D), -self.scale/D*np.ones(D)))
        self.r = np.vstack((np.zeros(1), -self.scale*np.ones(1), -self.scale*np.ones(1)))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        self.Rs = np.vstack((np.zeros(D), self.scale/D*np.ones(D), -self.scale/D*np.ones(D)))
        self.r = np.vstack((np.zeros(1), -self.scale*np.ones(1), -self.scale*np.ones(1)))
        
        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = np.tile(log_Ps, (1, self.K, 1))

        # log_Ps=log_Ps + log_Ps_tmp
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        T, D = data.shape
        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class singleAccumulatorMultipop_collapsingBound(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=40,bound_scale=0.1):
        super(singleAccumulatorMultipop_collapsingBound, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.bound_scale=bound_scale
        self.Ws = self.bound_scale * scale * np.hstack((np.zeros((K,1)),np.ones((K,1))))
        self.Ws[0,1]=0
        self.scale = scale

        self.Rs = np.vstack((np.zeros(D), self.scale/D*np.ones(D), -self.scale/D*np.ones(D)))
        self.r = np.vstack((np.zeros(1), -self.scale*np.ones(1), -self.scale*np.ones(1)))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        self.Rs = np.vstack((np.zeros(D), self.scale/D*np.ones(D), -self.scale/D*np.ones(D)))
        self.r = np.vstack((np.zeros(1), -self.scale*np.ones(1), -self.scale*np.ones(1)))
        
        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = log_Ps + np.dot(input[1:,:], self.Ws.T)[:, None, :] # Input effect

        log_Ps = np.tile(log_Ps, (1, self.K, 1))

        # log_Ps=log_Ps + log_Ps_tmp
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        T, D = data.shape
        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class singleAccumulatorMultipop_nonlinearCollapsingBound(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=40,tau=100,bound_offset=0.5):
        super(singleAccumulatorMultipop_nonlinearCollapsingBound, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.tau=tau
        self.bound_offset=bound_offset
        self.Ws = -scale * np.hstack((np.zeros((K,1)),np.ones((K,1))))
        self.Ws[0,1]=0
        self.scale = scale

        self.Rs = np.vstack((np.zeros(D), self.scale*np.ones(D), -self.scale*np.ones(D)))
        self.r = np.zeros((K,1))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        def bound_func(times):
            return self.bound_offset+(1-self.bound_offset)*np.exp(-1000*times/self.tau)

        T, D = data.shape
        bound_input=bound_func(input[1:,:])

        self.Rs = np.vstack((np.zeros(D), self.scale*np.ones(D), -self.scale*np.ones(D)))
        self.r = np.zeros((self.K,1))
        self.Ws = -self.scale * np.hstack((np.zeros((self.K,1)),np.ones((self.K,1))))
        self.Ws[0,1]=0

        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = log_Ps + np.dot(bound_input, self.Ws.T)[:, None, :] # Input effect
        log_Ps = np.tile(log_Ps, (1, self.K, 1))

        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        T, D = data.shape
        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class dualAccumulatorMultipopRace(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=20):
        super(dualAccumulatorMultipopRace, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        # self.Rs = np.zeros([K, D])
        self.scale=scale
        msk=np.vstack((np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2))))), np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.vstack((np.zeros(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1)))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        scale=self.scale
        T, D = data.shape
        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        msk=np.vstack((np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2))))), np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.vstack((np.zeros(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1)))
        
        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = np.tile(log_Ps, (1, self.K, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess
        # return np.zeros([T-1,D,D])

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class dualAccumulatorMultipopRace_collpasBound(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=20,tau=100,bound_offset=0.5):
        super(dualAccumulatorMultipopRace_collpasBound, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.tau=tau
        self.bound_offset=bound_offset
        self.Ws = -scale * np.hstack((np.zeros((K,1)),np.zeros((K,1)),np.ones((K,1))))
        self.Ws[0,2]=0
        self.scale=scale
        msk=np.vstack((np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2))))), np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.zeros((K,1))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        def bound_func(times):
            return self.bound_offset+(1-self.bound_offset)*np.exp(-1000*times/self.tau)
        
        bound_input=bound_func(input[1:,:])
        scale=self.scale
        T, D = data.shape
        self.Ws = -scale * np.hstack((np.zeros((self.K,1)),np.zeros((self.K,1)),np.ones((self.K,1))))
        self.Ws[0,2]=0

        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        msk=np.vstack((np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2))))), np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.zeros((self.K,1))
        

        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = log_Ps + np.dot(bound_input, self.Ws.T)[:, None, :] # Input effect
        log_Ps = np.tile(log_Ps, (1, self.K, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess
        # return np.zeros([T-1,D,D])

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class dualAccumulatorMultipopCompete(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=20):
        super(dualAccumulatorMultipopCompete, self).__init__(K, D, M)
        self.x1_boundVal=1
        self.x2_boundVal=1
        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        # self.Rs = np.zeros([K, D])
        self.scale=scale
        x1=np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2)))))
        x2=np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))
        msk=np.vstack((x1, -1*x1, x2, -1*x2))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.vstack((np.zeros(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1)))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        scale=self.scale
        T, D = data.shape
        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        x1=np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2)))))
        x2=np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))
        msk=np.vstack((x1, -1*x1, x2, -1*x2))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.vstack((np.zeros(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1)))
        
        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = np.tile(log_Ps, (1, self.K, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        # Ez = np.sum(expected_joints, axis=0) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess
        # return np.zeros([T-1,D,D])

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class dualAccumulatorMultipopCompete_collapsBound(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=20,tau=100,bound_offset=0.5):
        super(dualAccumulatorMultipopCompete_collapsBound, self).__init__(K, D, M)
        self.x1_boundVal=1
        self.x2_boundVal=1

        self.tau=tau
        self.bound_offset=bound_offset
        self.Ws = -scale * np.hstack((np.zeros((K,1)),np.zeros((K,1)),np.ones((K,1))))
        self.Ws[0,2]=0

        # Parameters linking past observations to state distribution
        self.scale=scale
        x1=np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2)))))
        x2=np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))
        msk=np.vstack((x1, -1*x1, x2, -1*x2))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.zeros((K,1))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        def bound_func(times):
            return self.bound_offset+(1-self.bound_offset)*np.exp(-1000*times/self.tau)

        bound_input=bound_func(input[1:,:])
        scale=self.scale
        T, D = data.shape

        self.Ws = -scale * np.hstack((np.zeros((self.K,1)),np.zeros((self.K,1)),np.ones((self.K,1))))
        self.Ws[0,2]=0

        x1=np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2)))))
        x2=np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))
        msk=np.vstack((x1, -1*x1, x2, -1*x2))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.zeros((self.K,1))
        
        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = log_Ps + np.dot(bound_input, self.Ws.T)[:, None, :] # Input effect
        log_Ps = np.tile(log_Ps, (1, self.K, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        # Ez = np.sum(expected_joints, axis=0) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess
        # return np.zeros([T-1,D,D])

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class dualAccumulatorMultipopCompete_3state(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=20):
        super(dualAccumulatorMultipopCompete_3state, self).__init__(K, D, M)
        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        # self.Rs = np.zeros([K, D])
        self.scale=scale
        x1=np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2)))))
        x2=np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))
        msk=np.vstack((-1*x1, x2))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.vstack((np.zeros(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1)))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs,  self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        scale=self.scale
        T, D = data.shape
        #self.Rs=np.vstack((np.zeros(D),self.Rs[1:]))
        #Off diagonal elements of transition matrix (state switches), from past observations

        x1=np.hstack((np.ones((1, int(D/2))), np.zeros((1, int(D/2)))))
        x2=np.hstack((np.zeros((1, int(D/2))), np.ones((1, int(D/2)))))
        msk=np.vstack((-1*x1, x2))
        self.Rs = np.vstack((np.zeros(D), scale*msk))
        self.r = np.vstack((np.zeros(1), -D/2*scale*np.ones(1), -D/2*scale*np.ones(1)))
        
        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = np.tile(log_Ps, (1, self.K, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        # Ez = np.sum(expected_joints, axis=0) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))

        return -1 * hess
        # return np.zeros([T-1,D,D])

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        
        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp

class RecurrentTransitionsVariableLowerBoundary(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8, scale=20):
        super(RecurrentTransitionsVariableLowerBoundary, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.lb_loc=0.0
        self.lb_scale = 10.0
        self.Ws = np.zeros((K, M)) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        self.scale=scale
        self.Rs = np.vstack((np.zeros(D), scale*np.ones(D), -self.lb_scale*np.ones(D)))
        self.r = np.vstack((np.zeros(1), -scale*np.ones(1), -self.lb_scale*np.ones(1)))

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.lb_loc, self.lb_scale

    @params.setter
    def params(self, value):
        self.lb_loc, self.lb_scale = value
        self.Rs = np.array([0.0, self.scale, -self.lb_scale]).reshape((3, 1))

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        scale=self.scale
        T, D = data.shape

        self.Rs = np.array([0.0, self.scale, -self.lb_scale]).reshape((3, 1))
        self.r = np.vstack((np.zeros(1), -scale*np.ones(1), -self.lb_scale*np.ones(1)))

        mask = np.vstack((np.array([1.0,1.0,0.0]), np.ones((self.K-1,self.K))))
        log_Ps_tmp = -self.scale*np.ones((self.K,self.K)) + np.diag(np.concatenate(([self.scale],2.0*self.scale*np.ones(self.K-1))))
        log_Ps_tmp = mask * log_Ps_tmp + (1.0 - mask) * self.lb_loc * self.lb_scale * np.ones((self.K,self.K))

        log_Ps = np.dot(data[:-1], self.Rs.T)[:,None,:]
        log_Ps = log_Ps + self.r.T
        log_Ps = np.tile(log_Ps, (1, self.K, 1))

        # log_Ps=log_Ps + log_Ps_tmp
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                Rv = vtilde@self.Rs[k2:k2+1,:]
                hess += Ez[k1,k2] * \
                        ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                        + np.einsum('ti, tj -> tij', Rv, Rv))
        return -1 * hess

    def log_prior(self):
        loc_mean = 0.0
        loc_var = 0.5
        return np.sum(-0.5 * (self.lb_loc - loc_mean)**2 / loc_var)

        