from abc import ABC, abstractmethod


class Model(ABC):
    """An abstract deep learning model.
    The abstract class functions as a facade for the backend. Although
    current framework currently uses tensorflow, it is possible that future releases
    may use different front- or backends. The Model ABC should represent the
    baseline for any such model.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def create_save_directories(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass


class TensorflowModel(Model):
    """ABC for Models that rely on Tensorflow.
    The ABC provides an implementation to generate callbacks to monitor the
    model and write the data to HDF5 files. The function 'fit' simply forwards
    to the tensorflow.keras' 'fit', but will enable monitoring if wanted.
    """
    def __init__(self, name):
        super().__init__(name)

    def fit(self, train_data, batch_size, epochs, save_model):
        pass