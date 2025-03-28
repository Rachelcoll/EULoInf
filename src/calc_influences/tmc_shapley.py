import os
import numpy as np 
import pickle as pkl
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader
from torchvision import transforms
from utils import load_checkpoint

def influence_at_checkpoints_tmc_shapley(traindata, testdata, model, optimizer, predict_func, checkpoint_filepaths, device):
    scores = []
    for checkpoint in tqdm(checkpoint_filepaths, desc=f"Calculating TMC Shapley scores"):
        model, _, _, completion_rate = load_checkpoint(model, optimizer, checkpoint)
        shap = DShap(model, predict_func, traindata.dataset, testdata.dataset, device)
        shap.run(iterations=500, err=0.1, tolerance=0.01)
        score = shap.vals_tmc
        scores.append(score)
    return scores


def error(mem):
    if len(mem) < 100: # if the number of iterations is less than 100, need to continue
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:] # cumulative sum of the stored marginals (last 100)
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1) # how much the recent Shapley estimate deviates from most recent one
    errors = np.max(errors) # takes max error to have a stricter stopping criterion
    # print('Error: {}'.format(errors))
    return errors

def accuracy(predict_func, y_pred, labels):
    correct, total = 0, 0
    with torch.no_grad():
        # preds = F.softmax(y_pred, dim=1)
        # predicted = torch.argmax(preds, dim=1)
        predicted = predict_func(y_pred)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return correct / total

class DShap(object):
    
    def __init__(self, model, predict_func, train_dataset, test_dataset, device,
                 directory=None, seed=10):
        """
        Args:
            model: Torch model
            train_dataset: Training Dataset (torch.Dataset)
            test_dataset: Test Dataset (torch.Dataset)
            directory: Directory to save results and figures.
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
        """
            
        if seed is not None:
            np.random.seed(seed)

        self.directory = directory
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)  

        self.model = model
        self.predict_func = predict_func
        transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))
        self.train_set = [(transform(x), y) for x, y in train_dataset]
        self.test_set = [(transform(x), y) for x, y in test_dataset]
        self.train_len = len(self.train_set)
        self.device = device

        self.mem_tmc = np.zeros((0, self.train_len))
        self.idxs_tmc = np.zeros((0, self.train_len), int)
        test_classes = torch.tensor([label for _, label in self.test_set]).to(torch.int64)
        self.random_score = torch.max(torch.bincount(test_classes) / len(self.test_set) ).item()

        self.tmc_number = self._which_parallel(self.directory)

    def run(self, iterations, err, tolerance=0.01):
        """Calculates data sources(points) values.
        
        Args:
            iterations: number of iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
        """

        tmc_run = True 
        while tmc_run:
            if error(self.mem_tmc) < err: # to determine stopping criterion (i.e. convergence)
                tmc_run = False
            else:
                self.tmc_shap(
                    iterations, 
                    tolerance=tolerance, 
                )
                self.vals_tmc = np.mean(self.mem_tmc, 0)
            self.save_results()
        
        
    def save_results(self):
        """Saves results computed so far."""
        if self.directory is None:
            return
        tmc_dir = os.path.join(
            self.directory, 
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )

        raw_list = []
        for j in range(self.mem_tmc.shape[0]):
            row_dict = { self.idxs_tmc[j][i] : self.mem_tmc[j][i] for i in range(self.mem_tmc.shape[1]) }
            raw_list.append(row_dict)

        pkl.dump(raw_list, open(tmc_dir, 'wb'))
    
    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]      
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0' 
        return tmc_number

    def tmc_shap(self, iterations, tolerance=0.01):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
        """
        self._tol_mean_score()
        
        marginals, idxs = [], []
        for _ in range(iterations):

            marginals, idxs = self.one_iteration(
                tolerance=tolerance
            )
            self.mem_tmc = np.concatenate([
                self.mem_tmc, 
                np.reshape(marginals, (1,-1))  # dims: (some number, # train samples)
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc, 
                np.reshape(idxs, (1,-1))
            ])


    def one_iteration(self, tolerance):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs = np.random.permutation(self.train_len)
        marginal_contribs = np.zeros(self.train_len)

        truncation_counter = 0
        new_score = self.random_score
        self.model.train() # ensure dropout and batch norm act accordingly, no weight update updates

        #  Iterates through the entire Training dataset
        data_list = []
        label_list = []
        for i, idx in enumerate(idxs):
            old_score = new_score
            data_list.append(self.train_set[idx][0])
            label_list.append(torch.tensor(self.train_set[idx][1]))
            if i == 0:
                data = self.train_set[idx][0].unsqueeze(0)
                labels = torch.tensor([self.train_set[idx][1]])
            else:
                data = torch.stack(data_list, 0)
                labels = torch.stack(label_list, 0)

            data, labels = data.to(self.device), labels.to(self.device)
            new_score = accuracy(predict_func=self.predict_func, y_pred=self.model(data), labels=labels)

            marginal_contribs[idx] = (new_score - old_score)  
            distance_to_full_score = np.abs(new_score - self.mean_score)
            #  Performance Tolerance
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs


    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.model.eval()
        for _ in range(100):
            
            sampler = RandomSampler(self.test_set)
            loader = DataLoader(self.test_set, batch_size=64, sampler=sampler)

            # 1-pass
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                acc = accuracy(predict_func=self.predict_func, y_pred=self.model(data), labels=labels)
                scores.append(acc)
                break

        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)