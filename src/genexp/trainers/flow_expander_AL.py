from ..models import DiffusionModel
from .adjoint_matching_trajectory import AdjointMatchingTrajectoryFinetuningTrainer
import torch
from tqdm import tqdm
from .alm import AugmentedLagrangian
from .augmented_reward import AugmentedReward
from .adjoint_matching import sample_trajectories_ddpm
from tqdm import trange

class FlowExpanderALTrainer(AdjointMatchingTrajectoryFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr, 
                 traj_samples_per_stage, 
                 data_shape, 
                 constraint_fn=None,
                 grad_constraint_fn=None,
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 rew_type='score-matching',
                 base_model=None,
                 pre_trained_model=None,
                 alpha_div=1.0,
                 traj_len=100,
                 lmbda=1.0,
                 clip_grad_norm=None,
                 running_cost=False,
                 AL_lambda=1.0,
                 AL_iterations=10,
                 AM_num_iterations=20,
                 epsilon=0.1,
                 **kwargs):

        self.lmbda = lmbda
        self.alpha_div = alpha_div
        self.pre_trained_model = pre_trained_model
        self.grad_constraint_fn = grad_constraint_fn
        self.epsilon = epsilon
        self.AL_iterations = AL_iterations
        self.AL_lambda = AL_lambda
        self.constraint_fn = constraint_fn
        self.AM_num_iterations = AM_num_iterations
        
        if rew_type == 'score-matching':
            print("Using score-matching reward, lambda:", lmbda)
            grad_reward_fn = lambda x:  lmbda * (- (base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())) - alpha_div * (base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach()) - pre_trained_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())))
            # compute gradient of f_k along trajectory
            self.grad_reward_fn = grad_reward_fn
            grad_f_k_trajectory = lambda x, t: lmbda * (- (base_model.score_func(x, torch.tensor(t, device=x.device).float().detach())) - alpha_div * (base_model.score_func(x, torch.tensor(t, device=x.device).float().detach()) - pre_trained_model.score_func(x, torch.tensor(t, device=x.device).float().detach())))
            self.grad_f_k_trajectory = grad_f_k_trajectory
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, grad_f_k_trajectory, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm, running_cost=running_cost,**kwargs)
    

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()

    def run_AM_oracle(self, reward_augmentation, K=3, num_iters_AM=10):
        # set new reward to agumented reward = - reward_augmentation + original reward
        # only the reward_augmentation term changes across iterations of the Augmented Lagrangian scheme
        self.grad_reward_fn = lambda x: - reward_augmentation +  self.lmbda * (- self.gamma * self.grad_constraint(x) - (self.base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())) - self.alpha_div * (self.base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach()) - self.pre_trained_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())))

        for _ in tqdm(range(K)): #5
            for _ in tqdm(range(num_iters_AM)):
                dataset = self.get_dataset(1, verbose=False)
                self.finetune_stage(dataset, verbose=False)
            self.update_base_model()


    def constrained_fine_tuning_via_AL_method(self):

        augmented_reward = AugmentedReward(grad_reward_fn = self.grad_reward_fn, 
                                          grad_constraint_fn = self.grad_constraint_fn,
                                          constraint_fn = self.constraint_fn,
                                          reward_lambda = self.AL_lambda,
                                          device = self.device,
                                        )

        # initialize Augmented Lagrangian method clss
        alm = AugmentedLagrangian(constraint_fn=self.constraint_fn,
                                    rho_max=10.0, #10000
                                    eta=1.25,
                                    rho_init=0.5,
                                    lambda_min=-1e6, #-10
                                    tau=0.99,
                                    baseline=False,
                                    base_lambda=-10.0, #0
                                    device="cuda",
                                )
        
         #### execute AUGMENTED LAGRANGIAN SCHEME for iterations = AL_iterations ####
        for it in trange(1, self.AL_iterations + 1, desc="AL iterations", leave=False):
            lambda_, rho_ = alm.get_current_lambda_rho()
            augmented_reward.set_lambda_rho(lambda_, rho_)

            # Setup augmented reward for self=trainer (ie AM oracle)
            self.grad_reward_fn = augmented_reward.grad_reward_fn
            self.fine_model.to(self.device)

            # Run AM oracle finetuning loop
            for _ in range(1, self.AM_num_iterations + 1):
                dataset = self.get_dataset(1)
                self.finetune_stage(dataset, verbose=False)

            # Set the finetuned model as the new generator model
            self.update_base_model()

            # Generate samples for lambda update
            num_samples = 1000
            x0 = torch.randn(num_samples, 2, device=self.device)
            with torch.no_grad():
                trajs, _ = sample_trajectories_ddpm(self.base_model, x0, self.traj_len)
            x_new = trajs[:, -1, :]  

            alm.update_lambda_rho(x_new)
            

        
