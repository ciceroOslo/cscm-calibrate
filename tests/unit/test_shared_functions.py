import pytest
import numpy as np
import json
import os
from unittest.mock import patch, mock_open

from cscm_calibrate.shared_functions import rmse, make_config_distro_json


def test_rmse_basic_calculation():
    """Test RMSE calculation with simple arrays."""
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])
    result = rmse(obs, mod)
    expected = np.sqrt(np.sum((obs - mod) ** 2) / len(obs))
    print(result)
    print(expected)
    assert np.isclose(result, expected)


def test_rmse_perfect_match():
    """Test RMSE when observed and modeled are identical."""
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.0, 2.0, 3.0])
    result = rmse(obs, mod)
    assert result == 0.0


def test_rmse_single_value():
    """Test RMSE with single values."""
    obs = np.array([5.0])
    mod = np.array([7.0])
    result = rmse(obs, mod)
    assert result == 2.0


def test_make_config_distro_json_basic():
    """Test basic functionality of make_config_distro_json without fixtures."""
    import builtins
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(matrix.shape)
    parameter_names = ['param1', 'param2']
    json_name = 'test_config.json'

    make_config_distro_json(matrix, parameter_names, json_name)


    assert os.path.exists('data/test_config.json')
    with open('data/test_config.json', 'r', encoding='utf-8') as rfile:
        config_list = json.load(rfile)
    assert len(config_list) == 2  # Two configurations
    print(config_list)
    assert 'pamset_udm' in config_list[0]
    assert 'pamset_emiconc' in config_list[0]
    assert 'pamset_carbon' in config_list[0]
    assert 'Index' in config_list[0]
    os.remove('data/test_config.json')