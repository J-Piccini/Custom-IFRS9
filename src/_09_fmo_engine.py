from typing import List, Any, Dict

import numpy as np
from scipy.stats import norm 

def shock_calculator(
    sat_model_coeffs: np.ndarray,
    sat_model_coeffs_shocks: np.ndarray,
    target_variance: float
) -> np.ndarray:
    """
    Given the satellite model's coefficients, the shocks to be applied, and the target variable variance, 
    this function calculates the notches to be applied to migration matrix.

    Arguments:
    sate_model_coeffs        -- np.ndarray: Coefficients of the satellite model
    sate_model_coeffs_shocks -- np.ndarray: Shocks to be applied
    target_variance          -- float: Variance of the target variable
    
    Returns:
        np.ndarray: Notches to be applied
    """
    
    asset_correlation = np.sqrt(target_variance)

    return (sat_model_coeffs_shocks @ sat_model_coeffs) * asset_correlation / ( np.sqrt(1 - asset_correlation**2) )

def pre_process_migration_matrix(
    matrix: np.ndarray,
    zero_substitute: float = 0.001
) -> np.ndarray:
    """
    Because the normal inverse is applied, each zero-element (if zero) is increased of 0.001. Each increase is taken out of the main diagonal element.

    Arguments:
    matrix           -- np.ndarray: Starting migration matrix
    zero_subsititute -- float: Value to be subsituted to zeros

    Returns:
        np.ndarray: Processed migration matrix
    """

    matrix_adj = matrix.copy()
    for id, row in enumerate(matrix[:-1, :]):
        zero_transitions = (row == 0).sum()

        if zero_transitions:
            zeros_index = np.where(row == 0)[0]
            matrix_adj[id, zeros_index] = zero_substitute
            matrix_adj[id, id] -= zero_substitute * zero_transitions

    return matrix_adj

def matrix_notcher(
    macro_shock: float,
    matrix: np.ndarray
) -> np.ndarray:
    """
    Single matrix notching.
    For a given matrix, and a given notch, this function notches the matrix as described in the documentation.
    Notching is applied from right -> left, therefore we only calculates the notching process on columns 1:4, and not considering
    the last one, as it is an absorbing state.
    
    Arguments:
    macro_shock -- float: Notching factor
    matrix      -- np.ndarray: Starting ma

    Returns:
        np.ndarray: Notched matrix
    """

    notched_matrix = np.zeros_like(matrix)
    notched_matrix[-1, -1] = 1

    matrix_adj = pre_process_migration_matrix(
        matrix = matrix,
        zero_substitute = 0.001
    )

    # Identification of rows/columns to notch
    short_matrix = matrix_adj[:-1, 1:]

    # Flipping as notching is applied right -> left
    dev_flip = np.fliplr(short_matrix)
    cumulative_flipped = np.cumsum(dev_flip, axis = 1)

    ppf = norm.ppf(cumulative_flipped)
    ppf_shocked = ppf + macro_shock

    cdf = norm.cdf(ppf_shocked)

    result = cdf.copy()

    for col_id in range(result.shape[1] - 1):
        result[:, col_id + 1] -= result[:, :col_id + 1].sum(axis = 1)

    # Flipping again to obtain the correct orientation
    result = np.fliplr(result)
    notched_matrix[:-1, 1:] = result

    # Updating the first column
    notched_matrix[:, 0] = 1  - notched_matrix[:, 1:].sum(axis = 1)

    return notched_matrix

def apply_shock_sequence(
    starting_matrix: np.ndarray,
    macro_shocks: np.ndarray,
) -> List[np.ndarray]:    
    """
    Applies a sequence of macroeconomic shocks to the first three years.
    
    Arguments:
    macro_shocks    -- np.ndarray: Sequence of macroeconomic shocks to apply
    starting_matrix -- np.ndarray: Initial probability transition matrix
        
    Returns:
        List[np.ndarray]: List of notched matrices, one for each shock
    """
    
    # Initialize output list
    notched_matrices = []
    
    # Apply shocks sequentially
    for shock in macro_shocks:
        # Choose base matrix based on cumulative parameter
        base_matrix = notched_matrices[-1] if notched_matrices else starting_matrix
        
        # Apply shock and store result
        notched_matrix = matrix_notcher(
            matrix = base_matrix,
            macro_shock = shock
            )
        
        notched_matrices.append(notched_matrix)
    
    return notched_matrices


def matrix_projections(
    starting_matrix: np.ndarray,
    notched_matrices: list,
    projection_years: int = 10
) -> List[np.ndarray]:
    """
    Projects transition matrices into the future using matrix multiplication.
    
    This function first applies the sequence of notched matrices in order through
    matrix multiplication, then continues projecting using the starting matrix
    until the desired number of years is reached.
    
    Arguments:
    starting_matrix  -- np.ndarray: Base transition matrix used for projections after notched periods
    notched_matrices -- list: Sequence of notched transition matrices for initial periods
    projection_years -- int: Total number of years to project, default is 10
        
    Returns:
        List[np.ndarray]: List of projected transition matrices
    """
    
    # Initialize projections list
    projections = []
    
    # First year is simply the first notched matrix
    if notched_matrices:
        projections.append(notched_matrices[0])
    
    # Apply notched matrices sequentially
    for i in range(1, len(notched_matrices)):
        next_projection = projections[i - 1] @ notched_matrices[i]
        projections.append(next_projection)
    
    # Continue projections with starting matrix
    for _ in range(len(notched_matrices), projection_years):
        next_projection = projections[-1] @ starting_matrix
        projections.append(next_projection)
    
    return projections


def cumulative_pd(
    matrix_projections: list,
    default_statuses: int = 2
) -> np.ndarray:
    """
    
    """
    n_class = matrix_projections[0].shape[0]
    n_years = len(matrix_projections)

    # Cumulative PD array initialization
    cum_pd_arr = np.zeros((n_years, n_class))

    for yr, matrix in enumerate(matrix_projections):
        for i in range(n_class):
            cum_pd_arr[yr, i] = matrix[i, -default_statuses:].sum()

    return np.vstack([np.zeros((1, n_class)), cum_pd_arr])


def interpolate_pd(
    cum_pd_arr: np.ndarray,
    datapoint_p_year: int = 12
) -> np.ndarray:
    
    """
    This function interpolates cumulative PDs to monthly level. Default value assumes datapoints are taken yearly.
    Interpolation is done for the first four classes. The final one, that is, the absorbing state, is forced to 100%,
    to reduce the computational burden.
    """
    
    n_class = cum_pd_arr.shape[1]
    n_years = cum_pd_arr.shape[0] - 1

    year_datapoints = np.arange(n_years + 1)
    single_interp_arr = np.linspace(
        start = 0,
        stop = n_years,
        num = datapoint_p_year *  n_years + 1
        )

    interp_pd = np.zeros((single_interp_arr.shape[0], n_class))
    for status in range(n_class - 1):
        interp_pd[:, status] = np.interp(
            x = single_interp_arr,
            xp = year_datapoints,
            fp = cum_pd_arr[:, status]
            )
    
    # Absorbing status is always 100%
    interp_pd[:, -1] = np.ones((interp_pd.shape[0], ))

    return interp_pd



def marginal_pd(
    cum_pd: np.ndarray,
    default_statuses: int = 2
) -> np.ndarray:
    
    """"
    Calculate the marginal PD, starting from the cumulative pd. A zero row is added on top to reflect the starting point
    """
    marginal_arr = np.vstack([
            np.zeros((cum_pd.shape[1], )), 
            np.diff(
                cum_pd, 
                axis = 0, 
                n = 1
                )
        ])
    
    marginal_arr[:, -default_statuses:] = np.ones((marginal_arr.shape[0], default_statuses))
    return marginal_arr

def main_pd_curves(
    starting_matrix: np.ndarray,
    risk_params: Any,
    params: Any
) -> np.ndarray:
    
    """
    Sequentially applies all needed step 
    """

    macro_shocks = risk_params['satellite_models_coefficients'] @ risk_params['satellite_models_shocks']

    notched_matrix = apply_shock_sequence(
        starting_matrix = starting_matrix,
        macro_shocks = macro_shocks
        )
    
    multi_year_matrix_projection_arr = matrix_projections(
        starting_matrix = starting_matrix,
        notched_matrices = notched_matrix,
        projection_years = params.n_projection_years
        )

    cumulative_pd_arr = cumulative_pd(
        matrix_projections = multi_year_matrix_projection_arr
    )

    interpolated_cum_pd_arr = interpolate_pd(
        cum_pd_arr = cumulative_pd_arr,
        datapoint_p_year = params.datapoint_p_year
    )

    marginal_pd_arr = marginal_pd(
        cum_pd = interpolated_cum_pd_arr
    )

    return marginal_pd_arr


def marginal_pd_arr_set_up(
    risk_params: dict,
    params: Any
) -> Dict[str, Dict[str, np.ndarray]]:
    
    marginal_pd_dict = {}
    for in_split_key, split_dict in risk_params['migration_matrix'].items():

        for group, group_list in split_dict.items():
            # Only initialize if group doesn't exist in the dictionary yet
            if group not in marginal_pd_dict:
                marginal_pd_dict[group] = {}
                
            group_arr = np.array(group_list)

            mar_pd_arr = main_pd_curves(
                starting_matrix = group_arr,
                risk_params = risk_params,
                params = params
                )
            
            marginal_pd_dict[group][in_split_key] = mar_pd_arr
    return marginal_pd_dict

def ccf_selection(
    ccf_matrix: np.ndarray,
    params: Any,
    portfolio_age: int
) -> np.ndarray:
    """
    
    """
    
    return ccf_matrix[np.where( portfolio_age >= np.array(params.vintages_start_list) )[0][-1], :]

def weighted_lgd(
    lgd_arr: np.ndarray,
    migration_matrix: np.ndarray,
    n_year: int = 3
) -> float:
    """
    Calculate the weighted Loss Given Default (LGD) using a migration matrix.

    This function computes the probability of curing and weights the LGD accordingly
    based on the progression of a migration matrix over a specified number of years.

    Arguments
    lgd_arr -- np.ndarray: An array of Loss Given Default (LGD) values, lgd_arr[0] = LGD for liquidation
                           lgd_arr[1]: LGD in case of cure
    migration_matrix -- np.ndarray: Migration matrix
    n_year -- int: optional Number of years to project the migration matrix forward. Default is 3 years.

    Returns
    float
        Weighted LGD calculated as:
        P(cured) * LGD_cured + P(default) * LGD_default
    """

    # Project migration matrix forward
    migration_matrix_arr = np.zeros((n_year, *migration_matrix.shape))
    migration_matrix_arr[0] = migration_matrix
    for i in range(1, n_year):
        migration_matrix_arr[i] = migration_matrix_arr[i - 1] @ migration_matrix

    # Extract probability of cure from the final projected matrix
    # Note: This assumes a specific matrix structure; adjust indexing if needed
    # print(f'{[mm[3, 0] for mm in migration_matrix_arr]}')
    p_cure = migration_matrix_arr[-1, 3, 0]
    # Calculate weighted LGD
    return p_cure * lgd_arr[1] + (1 - p_cure) * lgd_arr[0]


def maturity_based_lgd(
    lgd_arr: np.ndarray,
    geography: str,
    migration_matrix_dict: dict,
) -> Dict[str, float]:
    
    lgd_dict = {}

    for group, group_dict in migration_matrix_dict.items():
        group_mat = group_dict[geography]
        lgd_dict[group] = weighted_lgd(
            lgd_arr = lgd_arr,
            migration_matrix = np.array(group_mat)
        )

    return lgd_dict