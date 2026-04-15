from ..models import DiffusionModel
from .adjoint_matching_trajectory import AdjointMatchingTrajectoryFinetuningTrainer
import torch

class FlowExpanderTrainer(AdjointMatchingTrajectoryFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr, 
                 traj_samples_per_stage, 
                 data_shape, 
                 grad_constraint_fn=None,
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 rew_type='score-matching',
                 base_model=None,
                 pre_trained_model=None,
                 alpha_div_fn =1.0,
                 traj_len=1000,
                 lmbda_fn=None,
                 eta = 1.0, # constraint strength
                 gamma=1.0, # update step size
                 clip_grad_norm=None,
                 running_cost=False,
                 epsilon=0.1,
                 **kwargs):
        

        self.grad_constraint_fn = grad_constraint_fn  # log gradient of constraint function
        self.pre_trained_model = pre_trained_model
        self.alpha_div_fn = alpha_div_fn
        self.gamma = gamma
        self.eta = eta
        self.base_model = base_model
        self.traj_len = traj_len
        self.epsilon = epsilon
        self.lmbda_fn = lmbda_fn
        
        if rew_type == 'score-matching':
            # recall that here 0 is the data-level time-step 
            grad_reward_fn = lambda x:  self.gamma * self.lmbda_fn(0) * ( - (self.base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())) - self.alpha_div_fn(0) * (self.base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach()) - self.pre_trained_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())))
            # compute gradient of f_k along trajectory
            grad_f_k_trajectory = lambda x, t: self.gamma * self.lmbda_fn(t) * ( - (self.base_model.score_func(x, torch.tensor(t, device=x.device).float().detach())) - self.alpha_div_fn(t) * (self.base_model.score_func(x, torch.tensor(t, device=x.device).float().detach()) - self.pre_trained_model.score_func(x, torch.tensor(t, device=x.device).float().detach())))
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, grad_f_k_trajectory, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm, running_cost=running_cost,**kwargs)
    

    def set_grad_reward_fn_for_projection(self):
        self.running_cost = False
        self.grad_reward_fn = lambda x: self.gamma * self.eta * self.grad_constraint_fn(x)


    def set_grad_reward_fn_for_expansion(self):
        self.running_cost = True
        self.grad_reward_fn = lambda x:  self.gamma * self.lmbda_fn(0) * ( - (self.base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())) - self.alpha_div_fn(0) * (self.base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach()) - self.pre_trained_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())))
            

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
