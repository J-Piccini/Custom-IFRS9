from typing import Any
import json
import numpy as np
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    def default(
        self, 
        obj
    ):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Save parameters to JSON
def save_risk_parameters(
    risk_parameters: dict, 
    filepath: str
) -> None:
    """
    Save as .json file the risk parameters dictionary, and all necessary variables to run the application engine.

    risk_parameters:      --- dictionary containing the risk parameers
    """
    with open(filepath, 'w') as f:
        json.dump(
            risk_parameters, 
            f, 
            cls = NumpyEncoder, 
            indent = 4
        )


# Load parameters from JSON
def load_risk_parameters(
    filepath: str
) -> dict:
    
    """
    Loads the .json file and restructure it as a dictionary. Convert to array the relevant variables.
    """
    with open(filepath, 'r') as f:
        params = json.load(f)

        # Convert lists back to numpy arrays where needed
        # params['migration_matrix'] = np.array(params['migration_matrix'])
        
        # params['lgd'] = np.array(params['lgd'])
        params['lgd_liq_cur'] = np.array(params['lgd_liq_cur'])
        
        # params['ccf'] = np.array(params['ccf'])
        # params['ccf_vintages'] = np.array(params['ccf_vintages'])
        # params['ccf_time_windows'] = np.array(params['ccf_time_windows'])
        # params['K'] = np.array(params['K'])
        if 'K' in params.keys():
            params['K'] = np.array(params['K'])
            
        params['satellite_models_coefficients'] = np.array(params['satellite_models_coefficients'])
        params['satellite_models_shocks'] = np.array(params['satellite_models_shocks'])

        return params