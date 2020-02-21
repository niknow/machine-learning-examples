from unittest import TestCase
import numpy as np
import tempfile

from keras_grid import MLPGrid, LSTMGrid

import os
import json


class TestModelGrid(TestCase):
    """
    Tests abstract ModelGrid class via MLPGrid instantiation.
    """

    def setUp(self):
        self.mlpg = MLPGrid(num_inputs=3,
                            num_outputs=2,
                            range_units=np.array([8, 26, 32]),
                            range_layers=np.array([2, 3]))
        self.mlpgc = MLPGrid(num_inputs=3,
                             num_outputs=2,
                             range_units=np.array([8, 26, 32]),
                             range_layers=np.array([2, 3]),
                             constant_num_weights=True)
        self.mlpg_trained = MLPGrid(num_inputs=3,
                            num_outputs=2,
                            range_units=np.array([8, 26, 32]),
                            range_layers=np.array([2, 3]))
        self.mlpg_trained.initialize()
        self.mlpg_trained.compile(loss='mean_squared_error',
                          metrics=['mean_squared_error', 'mean_absolute_error'],
                          optimizer='Adam')
        np.random.seed(1)
        self.mlpg_trained.fit(
            x=np.random.normal(0, 1, (10, 3)),
            y=np.random.normal(0, 1, (10, 2)),
            validation_split=0.2,
            epochs=1,
        )



    def test_can_instantiate(self):
        self.assertTrue(isinstance(self.mlpg, MLPGrid))
        self.assertEqual(self.mlpg.num_inputs, 3)
        self.assertEqual(self.mlpg.num_outputs, 2)
        self.assertEqual(self.mlpg.constant_num_weights, False)
        self.assertEqual(self.mlpgc.constant_num_weights, True)
        np.testing.assert_array_equal(self.mlpg.range_units, np.array([8, 26, 32]))
        np.testing.assert_array_equal(self.mlpg.range_layers, np.array([2, 3]))

    def test__initialize_unit_grid(self):
        self.mlpg._initialize_unit_grid()
        for i in range(len(self.mlpg.range_layers)):
            np.testing.assert_array_equal(self.mlpg.unit_grid[:, i], self.mlpg.range_units)
        self.mlpgc._initialize_unit_grid()
        for i in range(len(self.mlpgc.range_units)):
            for j in range(len(self.mlpgc.range_layers)):
                self.assertEqual(self.mlpgc.unit_grid[i][j],
                                 self.mlpgc.get_num_equiv_units(self.mlpgc.range_units[i],
                                                                self.mlpgc.range_layers[j]))

    def test_initialize_model_grid(self):
        self.mlpg._initialize_unit_grid()
        self.mlpg._initialize_model_grid()
        for i in range(len(self.mlpg.range_units)):
            for j in range(len(self.mlpg.range_layers)):
                self.assertTrue(self.mlpg[i][j] is not None)

    def test_compile(self):
        self.mlpg.initialize()
        self.mlpg.compile(loss='mean_squared_error',
                          metrics=['mean_squared_error', 'mean_absolute_error'],
                          optimizer='Adam')

    def test_fit(self):
        for unit_idx in range(len(self.mlpg_trained.range_units)):
            for layer_idx in range(len(self.mlpg_trained.range_layers)):
                self.assertTrue(layer_idx in self.mlpg_trained.history[unit_idx])

    def test_json(self):
        self.mlpg.initialize()
        jd = json.dumps(self.mlpg.to_json())
        jl = json.loads(jd)
        mlpg2 = MLPGrid.from_json(jl)
        self.assertEqual(self.mlpg, mlpg2)
        jl2 = mlpg2.to_json()
        self.assertEqual(jl, jl2)

    def test_save(self):
        with tempfile.TemporaryDirectory() as dirpath:
            self.mlpg_trained.save(dirpath)
            expected_files = sorted(['mlp_grid.json',
                'mlp_grid_history.pickle',
                'mlp_units0_layers0.h5',
                'mlp_units0_layers1.h5',
                'mlp_units1_layers0.h5',
                'mlp_units1_layers1.h5',
                'mlp_units2_layers0.h5',
                'mlp_units2_layers1.h5'])
            self.assertEqual(sorted(list(os.walk(dirpath))[0][2]), expected_files)
            mlpg2 = MLPGrid.from_disk(dirpath)
            self.assertEqual(self.mlpg_trained, mlpg2)
            for unit_idx in range(len(self.mlpg_trained.range_units)):
                for layer_idx in range(len(self.mlpg_trained.range_layers)):
                    self.assertEqual(self.mlpg_trained.history[unit_idx][layer_idx], mlpg2.history[unit_idx][layer_idx])
                    for l1, l2 in zip(self.mlpg_trained[unit_idx][layer_idx].layers,  mlpg2[unit_idx][layer_idx].layers):
                        self.assertEqual(l1.get_config(), l2.get_config())


class TestMLPGrid(TestCase):

    def setUp(self):
        self.mlpg = MLPGrid(num_inputs=3,
                            num_outputs=2,
                            range_units=np.array([8, 26, 32]),
                            range_layers=np.array([2, 3, 4]))

    def test_num_weights(self):
        self.mlpg.initialize()
        n_i = self.mlpg.num_inputs
        n_o = self.mlpg.num_outputs
        n_w = np.array([[MLPGrid.num_weights(n_i, n_o, n_L, n_u) for n_L in self.mlpg.range_layers] for n_u in self.mlpg.range_units])
        n_params = np.array([[self.mlpg[unit_idx][layer_idx].count_params() 
                            for layer_idx in range(len(self.mlpg.range_layers))]
                            for unit_idx in range(len(self.mlpg.range_units))])
        np.testing.assert_array_equal(n_w.flatten(), n_params.flatten())
    
    def test_num_units(self):
        n_i = self.mlpg.num_inputs
        n_o = self.mlpg.num_outputs
        n_w = np.array([[MLPGrid.num_weights(n_i, n_o, n_L, n_u) for n_L in self.mlpg.range_layers] for n_u in self.mlpg.range_units])
        n_u = np.array([[MLPGrid.num_units(n_i, n_o, self.mlpg.range_layers[layer_idx], n_w[unit_idx][layer_idx]) for layer_idx in range(len(self.mlpg.range_layers))] for unit_idx in range(len(self.mlpg.range_units))])
        np.testing.assert_array_equal(n_u.flatten(), np.repeat(self.mlpg.range_units[:, np.newaxis], 3))


class TestLSTMGrid(TestCase):

    def setUp(self):
        self.lstm = LSTMGrid(num_inputs=3,
                             num_outputs=2,
                             num_time_steps=10,
                             range_units=np.array([8, 26, 32]),
                             range_layers=np.array([2, 3, 4]))
    def test_json(self):
        self.lstm.initialize()
        jd = json.dumps(self.lstm.to_json())
        jl = json.loads(jd)
        lstm2 = LSTMGrid.from_json(jl)
        self.assertEqual(self.lstm, lstm2)
        jl2 = lstm2.to_json()
        self.assertEqual(jl, jl2)

    def test_num_weights(self):
        self.lstm.initialize()
        n_i = self.lstm.num_inputs
        n_o = self.lstm.num_outputs
        n_w = np.array([[LSTMGrid.num_weights(n_i, n_o, n_L, n_u) 
                       for n_L in self.lstm.range_layers]
                       for n_u in self.lstm.range_units])
        n_params = np.array([[self.lstm[unit_idx][layer_idx].count_params() 
                            for layer_idx in range(len(self.lstm.range_layers))]
                            for unit_idx in range(len(self.lstm.range_units))])
        np.testing.assert_array_equal(n_w.flatten(), n_params.flatten())
    
    def test_num_units(self):
        n_i = self.lstm.num_inputs
        n_o = self.lstm.num_outputs
        n_w = np.array([[LSTMGrid.num_weights(n_i, n_o, n_L, n_u) 
                        for n_L in self.lstm.range_layers]
                        for n_u in self.lstm.range_units])
        n_u = np.array([[LSTMGrid.num_units(n_i, n_o, self.lstm.range_layers[layer_idx], n_w[unit_idx][layer_idx])
                        for layer_idx in range(len(self.lstm.range_layers))]
                        for unit_idx in range(len(self.lstm.range_units))])
        self.assertTrue(np.all(np.abs(n_u.flatten() - np.repeat(self.lstm.range_units[:, np.newaxis], 3))<=1))
        
