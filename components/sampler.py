import numpy as np
class BaseSampler:
    '''BaseSampler acts as an interface for all the samplers'''
    def __init__(self, all_data, clean_indices, dirty_indices, batch_size=50) -> None:
        self.all_data = all_data
        self.clean_indices = clean_indices
        self.batch_size = batch_size
        self.dirty_indices = dirty_indices

    def sample(self):
        return NotImplementedError

    def update_indices(self, new_clean_indices):
        self.clean_indices.extend(new_clean_indices)
        self.dirty_indices = [i for i in self.dirty_indices if i not in new_clean_indices]

class UniformSampler(BaseSampler):
    '''Uniform sampler samples the data uniformly'''
    def __init__(self, all_data, clean_indices, dirty_indices, batch_size=50) -> None:
        super().__init__(all_data, clean_indices, dirty_indices, batch_size)

    def sample(self):
        return np.random.choice(self.dirty_indices, self.batch_size)

#TODO fix code based on detector implementation
class DetectorSampler(BaseSampler):
    '''Detectorsampler uses the detector provided to 
    find the dirty samples and prioritizes sampling them'''
    def __init__(self, all_data, clean_indices, dirty_indices, detector, batch_size=50) -> None:
        super().__init__(all_data, clean_indices, dirty_indices, batch_size)
        self.detector = detector

    def sample(self):
        dirty_probability = self.detector.find_dirty(self.dirty_indices, self.all_data)
        return np.random.choice(self.dirty_indices, self.batch_size, p=dirty_probability)

