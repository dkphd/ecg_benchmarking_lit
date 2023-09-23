import torch 
import numpy as np
import random

from collections import namedtuple

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

SampleTuple = namedtuple("SampleTuple", "id_sample id_start id_end")

"""
1. Stores X and y
2. Allow for chunking of X
3. Returns random crop of X
"""

"""

ds_train = TimeseriesDatasetCrops(
            df_train,
            self.input_size,
            num_classes=self.num_classes,
            chunk_length=self.chunk_length_train if self.chunkify_train else 0,
            min_chunk_length=self.min_chunk_length,
            stride=self.stride_length_train,
            transforms=tfms_ptb_xl,
            annotation=False,
            col_lbl="label",
            npy_data=X_train,
        )
ds_valid = TimeseriesDatasetCrops(
            df_valid,
            self.input_size,
            num_classes=self.num_classes,
            chunk_length=self.chunk_length_valid if self.chunkify_valid else 0,
            min_chunk_length=self.min_chunk_length,
            stride=self.stride_length_valid,
            transforms=tfms_ptb_xl,
            annotation=False,
            col_lbl="label",
            npy_data=X_val,
        )

"""





class TimeseriesDatasetCrops(torch.utils.data.Dataset):
    """timeseries dataset with partial crops."""

    def __init__(self, X, y, output_size, chunk_length, min_chunk_length, random_crop=True, stride=None, start_idx=0, transforms=[], time_dim=-1, batch_dim=0):
        """output_size

        """          
        self.X = X
        self.y = y
        self.transforms = transforms
        self.random_crop = random_crop 
        self.id_mapping = []
        self.output_size = output_size
        
        self.time_dim = time_dim
        self.batch_dim = batch_dim

        for i in range(X.shape[batch_dim]):
            data_length = X.shape[time_dim]# tutaj
            if(chunk_length == 0):#do not split
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx,data_length,chunk_length if stride is None else stride))
                idx_end = [min(l+chunk_length, data_length) for l in idx_start]

                #remove final chunk(s) if too short
                for j in range(len(idx_start)):
                    if(idx_end[j]-idx_start[j]< min_chunk_length):
                        del idx_start[j:]
                        del idx_end[j:]
                        break
                
            for j in range(len(idx_start)):
                sample_tuple = SampleTuple(i, idx_start[j], idx_end[j])
                self.id_mapping.append(sample_tuple)
                    
    def __len__(self):
        return len(self.id_mapping)

    def __getitem__(self, idx):

        id_sample = self.id_mapping[idx].id_sample
        start_idx = self.id_mapping[idx].id_start
        end_idx = self.id_mapping[idx].id_end
        #determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.output_size)
        
        if(self.random_crop):#random crop
            if(timesteps==self.output_size):
                start_idx_crop= start_idx
            else:
                start_idx_crop = start_idx + random.randint(0, timesteps - self.output_size -1)
        else:
            start_idx_crop = start_idx + (timesteps - self.output_size)//2
        end_idx_crop = start_idx_crop+self.output_size


        X_sample = torch.index_select(self.X[id_sample], dim=self.time_dim - 1 if self.time_dim > 0 else self.time_dim, index=torch.arange(start_idx_crop,end_idx_crop))
        y_sample = self.y[id_sample]
        for t in self.transforms:
            X_sample = t(X_sample)

        return X_sample, y_sample
    
    def get_id_mapping(self):
        return self.id_mapping