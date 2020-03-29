import torch.nn as nn
import os

# const for all score functions
class ExperimentInfo(nn.Module):
    def __init__ (self, ndim_, total_n_, start_n_, end_n_, test_n_, name_, 
                  dataset_, draw_data_, draw_scores_, update_freq_, comparsion_):
        if ndim_ > 2 and (draw_data_ or draw_scores_):
            print("you cannot draw multidimensional data")
            1 / 0
            
        super(ExperimentInfo, self).__init__()
        
        self.ndim = ndim_
        
        self.total_n = total_n_
        self.start_n = start_n_
        self.end_n   = end_n_
        self.test_n  = test_n_
        
        self.name = name_
        
        os.mkdir("./Results/" + self.name)
        
        self.dataset = dataset_(self.total_n, self.start_n, self.end_n, self.test_n)   
        self.dataset.generate_data(draw_data_, self.name)
        
        self.draw_data = draw_data_
        self.draw_scores = draw_scores_
        
        self.update_freq = update_freq_
        self.comparsion = comparsion_

        # for every score save train dataset, accuracies and scores (on the grid)
        self.per_score_acc = {}
        
        for score_name in ("rand", "mvar", "sqsm", "RKHS", "Hvar", "l2fm"):
            self.per_score_acc[score_name] = []
                
        # create folder structure
        os.mkdir("./Results/" + self.name + "/accuracies")
        
        if self.draw_scores:
            os.mkdir("./Results/" + self.name + "/score_plots")
            os.mkdir("./Results/" + self.name + "/scores")
            
            for score_name in ("rand", "mvar", "sqsm", "RKHS", "Hvar", "l2fm"):
                os.mkdir("./Results/" + self.name + "/score_plots/" + score_name)
                os.mkdir("./Results/" + self.name + "/scores/"      + score_name)
                
        # save overall description
        out = open("./Results/" + self.name + "/description.txt", 'a')
        
        out.write(
              "total_n = " + str(self.total_n) + "\n"
            + "start_n = " + str(self.start_n) + "\n"
            + "end_n   = " + str(self.end_n)   + "\n"
            + "test_n  = " + str(self.test_n)  + "\n"
            + "ndim    = " + str(self.ndim)    + "\n")
            
        out.close()

    def save(self, score_name):
        acc_file = open("./Results/" + self.name + "/accuracies/acc_" + score_name + ".txt", 'a') 
        acc_file.write(str(self.per_score_acc[score_name][-1]) + "\n")
        acc_file.close()
        
        