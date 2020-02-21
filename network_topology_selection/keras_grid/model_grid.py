from abc import abstractmethod, ABC
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import os
import json
import pickle

class ModelGrid(ABC):
    """
    This class wraps a 2D grid of keras models.
    """

    _json_suffix = '_grid.json'
    _history_suffix = '_grid_history.pickle'

    def __init__(self, num_inputs, num_outputs, range_units, range_layers, constant_num_weights=False):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.range_units = range_units
        self.range_layers = range_layers
        self.constant_num_weights = constant_num_weights
        self.models = {}
        self.history = {}
        self.unit_grid = None

    def __eq__(self, other): 
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.num_inputs == other.num_inputs and \
               self.num_outputs == other.num_outputs and \
               np.all(self.range_units == other.range_units) and \
               np.all(self.range_layers == other.range_layers) and \
               self.constant_num_weights == other.constant_num_weights \
               and np.all(self.unit_grid.flatten() == other.unit_grid.flatten()) if self.unit_grid is not None else True

    @classmethod
    def from_disk(cls, path):
        """
        Reconstructs an instance of this class from a folder of files written with its `save´-method.
        """
        with open(os.path.join(path, cls._name + ModelGrid._json_suffix)) as fp:
            mgrid = cls.from_json(json.load(fp))
        for unit_idx in range(len(mgrid.range_units)):
            mgrid[unit_idx] = {}
            for layer_idx in range(len(mgrid.range_layers)):
                mgrid[unit_idx][layer_idx] = load_model(
                    os.path.join(path, cls._name + '_units%i_layers%i.h5' % (unit_idx, layer_idx)))
        with open(os.path.join(path, cls._name + ModelGrid._history_suffix), 'rb') as fp:
            mgrid.history = pickle.load(fp)
        return mgrid

    def __getitem__(self, key):
        """
        :param key: A tuple (unit_idx, layer_idx)
        :return: Model with self.unit_grid[unit_idx][layer_idx] number of units and self.range_layers[layer_idx]
                number of layers.
        """
        return self.models[key]

    def __setitem__(self, key, item):
        self.models[key] = item

    def _initialize_unit_grid(self):
        self.unit_grid = np.repeat(self.range_units[:, np.newaxis], len(self.range_layers), axis=1)
        if self.constant_num_weights:
            for layer_idx in range(1, len(self.range_layers)):
                for unit_idx in range(len(self.range_units)):
                    self.unit_grid[unit_idx, layer_idx] = self.get_num_equiv_units(self.range_units[unit_idx],
                                                                                   self.range_layers[layer_idx])

    def _initialize_model_grid(self):
        for unit_idx in range(len(self.range_units)):
            self.models[unit_idx] = {}
            for layer_idx in range(len(self.range_layers)):
                self.models[unit_idx][layer_idx] = self._create_model(unit_idx, layer_idx)

    def initialize(self):
        self._initialize_unit_grid()
        self._initialize_model_grid()

    def compile(self, **kwargs):
        for units_idx in range(len(self.range_units)):
            for layer_idx in range(len(self.range_layers)):
                self.models[units_idx][layer_idx].compile(**kwargs)

    def fit(self, **kwargs):
        for unit_idx in range(len(self.range_units)):
            self.history[unit_idx] = {}
            for layer_idx in range(len(self.range_layers)):
                print("-"*30)
                print("Fitting with units_index=%i and layers_index=%i" % (unit_idx, layer_idx))
                print("-"*30)
                self.history[unit_idx][layer_idx] = self.models[unit_idx][layer_idx].fit(**kwargs).history
                print("DONE")
                print("")
                print("")
        return self.history

    def save(self, path):
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        for unit_idx in range(len(self.range_units)):
            for layer_idx in range(len(self.range_layers)):
                self.models[unit_idx][layer_idx].save(
                    os.path.join(path, type(self)._name + '_units%i_layers%i.h5' % (unit_idx, layer_idx)))
        with open(os.path.join(path, type(self)._name + ModelGrid._history_suffix), 'wb') as fp:
            pickle.dump(self.history, fp)
        with open(os.path.join(path, type(self)._name + ModelGrid._json_suffix), 'w') as fp:
            json.dump(self.to_json(), fp)


    def get_num_equiv_units(self, num_units, num_layers):
        if self.constant_num_weights:
            if num_layers == self.range_layers[0]:
                n_u = num_units
            else:
                n_w = type(self).num_weights(self.num_inputs, self.num_outputs, self.range_layers[0], num_units)
                n_u = type(self).num_units(self.num_inputs, self.num_outputs, num_layers, n_w)
        else:
            n_u = num_units
        return n_u

    @abstractmethod
    def _create_model(self, unit_idx, layer_idx):
        pass

    def to_json(self):
        return {'num_inputs': self.num_inputs,
                'num_outputs': self.num_outputs,
                'range_units': self.range_units.tolist(),
                'range_layers': self.range_layers.tolist(),
                'constant_num_weights': self.constant_num_weights,
                'unit_grid': self.unit_grid.tolist()}
    
    @classmethod
    @abstractmethod
    def from_json(self, jsondict):
        pass
    
    @staticmethod
    def jsondict_to_initdict(jsondict):
        return {'num_inputs': jsondict['num_inputs'],
                'num_outputs': jsondict['num_outputs'],
                'range_units': np.array(jsondict['range_units']),
                'range_layers': np.array(jsondict['range_layers']),
                'constant_num_weights': jsondict['constant_num_weights']}

    @classmethod
    @abstractmethod
    def num_weights(cls, n_i, n_o, n_L, n_u):
        """
        Computes the total number of parameters in the model assuming all layers have the same
        number of units (except the output layer).

        param n_i: number of inputs
        param n_o: number of outputs
        param n_L: number of layers
        param n_u: number of units per layer

        returns: total number of trainable weights

        """
        pass

    @classmethod
    @abstractmethod
    def num_units(cls, n_i, n_o, n_L, n_w):
        """
        Computes the total number ofunits per layer to achieve a given number of parameters assuming all layers
        have the same number of units (except the output layer).

        param n_i: number of inputs
        param n_o: number of outputs
        param n_L: number of weights
        param n_w: number of weights

        returns: number of units per layer needed

        """
        pass


class MLPGrid(ModelGrid):
    """
    This class wraps a 2D grid of keras sequential models with dense layers.
    """
    _name = 'mlp'

    @classmethod
    def num_weights(cls, n_i, n_o, n_L, n_u):
        if n_L == 2:
            return n_u * (n_i + n_o + 1) + n_o
        else:
            return (n_L - 2) * n_u ** 2 + (n_i + n_o + n_L - 1) * n_u + n_o

    @classmethod
    def num_units(cls, n_i, n_o, n_L, n_w):
        if n_L == 2:
            return int((n_w - n_o) / (n_i + n_o + 1))
        else:
            p = n_i + n_o + n_L - 1
            p /= n_L - 2
            q = n_o - n_w
            q /= n_L - 2
            return int(-p / 2 + np.sqrt(p ** 2 / 4 - q))
    
    @classmethod
    def from_json(cls, jsondict):
        initdict = ModelGrid.jsondict_to_initdict(jsondict)
        mlpg = MLPGrid(**initdict)
        mlpg.unit_grid=np.array(jsondict['unit_grid'])
        return mlpg

    def _create_model(self, unit_idx, layer_idx):
        num_layers = self.range_layers[layer_idx]
        n_u = self.unit_grid[unit_idx][layer_idx]
        model = Sequential()
        model.add(Dense(units=n_u, input_shape=(self.num_inputs,), activation='sigmoid'))
        for layer in range(1, num_layers - 1):
            model.add(Dense(units=n_u, activation='sigmoid'))
        model.add(Dense(units=self.num_outputs, activation='linear'))
        return model


class LSTMGrid(ModelGrid):
    """
    This class wraps a 2D grid of keras LSTM models.
    """
    _name = 'lstm'

    def __init__(self, num_inputs, num_outputs, range_units, range_layers, num_time_steps, constant_num_weights=False):
        super(LSTMGrid, self).__init__(num_inputs, num_outputs, range_units, range_layers, constant_num_weights)
        self.num_time_steps = num_time_steps

    @classmethod
    def num_weights(cls, n_i, n_o, n_L, n_u):
        return 4 * (2 * n_L - 3) * n_u**2  + (4 * n_i + n_o + 4 * n_L - 4 ) * n_u + n_o

    @classmethod
    def num_units(cls, n_i, n_o, n_L, n_w):
        a = 4 * (2 * n_L - 3)
        p = (4 * n_i + n_o + 4 * n_L - 4 )
        q = n_o - n_w
        p /= a
        q /= a
        return int(-p / 2 + np.sqrt(p ** 2 / 4 - q))

    def _create_model(self, unit_idx, layer_idx):
        num_units = self.range_units[unit_idx]
        num_layers = self.range_layers[layer_idx]
        n_u = self.get_num_equiv_units(num_units, num_layers)
        model = Sequential()
        model.add(LSTM(input_shape=(self.num_time_steps, self.num_inputs),
                       units=n_u,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       use_bias=True,
                       return_sequences=True))
        for layer in range(1, num_layers - 1):
            model.add(LSTM(units=n_u,
                           activation='tanh',
                           recurrent_activation='sigmoid',
                           use_bias=True,
                           return_sequences=True))
        model.add(Dense(self.num_outputs))
        return model

    def to_json(self):
        d = super().to_json()
        d['num_time_steps'] = self.num_time_steps
        return d

    @classmethod
    def from_json(cls, jsondict):
        initdict = ModelGrid.jsondict_to_initdict(jsondict)
        lstm = LSTMGrid(**initdict, num_time_steps=jsondict['num_time_steps'])
        lstm.unit_grid=np.array(jsondict['unit_grid'])
        return lstm
