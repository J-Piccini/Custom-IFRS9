from typing import Any, Dict, List, Tuple
import sys

import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

import matplotlib.pyplot as plt


def check_matrix_rows(
        matrix_bank, 
        tolerance = 1e-10
):
    
    row_sums = np.sum(matrix_bank, axis = 1)  # Sum along rows (axis=1)
    
    # Use np.isclose for floating-point comparisons
    if not np.all(np.isclose(row_sums, 1, atol = tolerance)):
        offending_rows = np.where(~np.isclose(row_sums, 1, atol = tolerance))[0]
        for row_index in offending_rows:
            print(f'Row {row_index}: Sum = {row_sums[row_index]}')
        raise ValueError('Matrix rows do not sum up to 1 within the specified tolerance.')


def bootstrap_curve_generator(
    bootstrap_dict: dict,
    n_it: int,
    years_projection: int = 10
) -> np.ndarray:
    
    tot_curve_arr = np.zeros((n_it, years_projection))

    for key_iteration, iteration_dict in bootstrap_dict.items():
        matrix_year_list = []
        for _, year_dict in iteration_dict.items():
            if year_dict.values():
                matrix_y_arr = np.array(list(year_dict.values()))
                if matrix_y_arr.size > 0:
                    matrix_year_list.append(np.mean(matrix_y_arr, axis = 0))

        final_matrix = np.mean(matrix_year_list, axis = 0)

        iteration_curve = matrix_projection(
            matrix = final_matrix[0],
            years_projection = 10
        )  

        tot_curve_arr[key_iteration, :] = iteration_curve
    
    return tot_curve_arr, final_matrix[0]


def matrix_projection(
    matrix: np.array,
    years_projection: int = 10
) -> np.ndarray:
    
    proj_arr = np.zeros((years_projection, ))
    proj_arr[0] = matrix[0, -2:].sum()
    final_matrix = matrix.copy()

    for i in range(1, years_projection):
        final_matrix = final_matrix @ matrix
        proj_arr[i] = final_matrix[0, -2:].sum()

    return proj_arr


def safe_sample(
    group,
    frac: float = 0.2,
    replace: bool = True,
    random_state: int = 42
):          
    if len(group) == 1:
        # If there's only one row, return it to include the presence of such status
        return group
    else:
        # Otherwise, sample with replacement
        return group.sample(frac = frac, replace = replace, random_state = random_state)


def migration_matrix_val(
    start_df: pd.DataFrame, 
    next_df: pd.DataFrame,
    params: Any
):
    
    ##### Matrix initialization #####
    nx = params.num_status
    num_labels = np.arange(nx)
    matrix = np.zeros((nx, nx))
    matrix[-1, -1] = 1

    for status in num_labels[:-1]:
        # Filter rows with current status
        start_mask = start_df['nPrommiseLoanStatus'] == status
        start_status_df = start_df[start_mask]
        
        # Get all loans in current status (including duplicates)
        loans_in_status = start_status_df['Loan_ID'].tolist()  # Changed from set to list
        total_loans_in_status = len(loans_in_status)  # Count including duplicates
        
        if total_loans_in_status == 0:
            matrix[status, status] = 1
            continue
        
        # Get corresponding loans in next_df, this time set is used as all unique Loan_ID will be retrieved
        common_df = next_df[next_df['Loan_ID'].isin(set(loans_in_status))]  

        # Count transitions including duplicates
        arrival_group_df = pd.DataFrame(
            {'Loan_ID': loans_in_status}
        ).merge(
            common_df[['Loan_ID', 'nPrommiseLoanStatus']], 
            on = 'Loan_ID', 
            how = 'left'
            )['nPrommiseLoanStatus'].value_counts(dropna = True)

        # Calculate transition probabilities
        for item, count in arrival_group_df.items():
            matrix[status, int(item)] = count / total_loans_in_status
        
        # Handle missing transitions
        if round(matrix[status, :].sum(), 4) != 1:
            if status == 0: 
                matrix[status, status] += 1. - matrix[status, :].sum()
            else:
                matrix[status, -1] += 1. - matrix[status, :].sum()

    return matrix


def get_sampled_snapshots(
    df: pd.DataFrame,
    time_resolution: int = 12
) -> Tuple[Dict[str, List[int]], Dict[str, np.ndarray]]:
    """
    The function takes as input a DataFrame, which is assumed to be the cleaned version of A&B, and time resolution, which must be equal to the time resolution 
    used for calculating migration matrices during calibration.

    Returns:
        Tuple[
            Dict[str, List[int]]  -- Dictionary where keys identify portfolios, and items identify the number of unique loans in a given portfolio, and snapshot
            Dict[str, np.ndarray] -- Dictionary where keys identify portfolios, and items identify the relevant snapshots used for calibration
    """
    sampled_snaps_dict = {}
    loans_samp_snap_dict = {}

    for ptf in df['Portfolio_u'].unique():
        loans_samp_snap_dict[ptf] = []

        mask_ptf = df['Portfolio_u'].eq(ptf)

        snaps_ptf = np.sort(df[mask_ptf]['Period'].unique())
        indices = np.concatenate([[0],  
                            np.arange(time_resolution - 1, len(snaps_ptf), time_resolution)  # Subsequent periods
                            ])
        valid_indices = indices[indices < len(snaps_ptf)]
        
        if valid_indices.size > 1: 
            # If only one, the performance window starts but does not complete. It can be disregarded
            sampled_snaps_dict[ptf] = snaps_ptf[valid_indices]

    # This needs to go once
    loans_samp_snap_dict = {}

    for ptf in sampled_snaps_dict:
        loans_samp_snap_dict[ptf] = []
        mask_ptf = df['Portfolio_u'].eq(ptf)
        
        for snap in sampled_snaps_dict[ptf]:
            mask_snap = df['Period'] == snap
            unique_loans = df[mask_ptf & mask_snap]['Loan_ID'].nunique()
            loans_samp_snap_dict[ptf].append(unique_loans)
            
    return loans_samp_snap_dict, sampled_snaps_dict


def benchmark_matrix_loop(
    df: pd.DataFrame, 
    params: Any
) -> dict:
    
    matrices_dict = {}
    time_granularity = params.time_resolution
    nx = params.num_status

    portfolio_list = df['Portfolio_u'].unique()
    for portfolio in portfolio_list:
        tmp_portfolio = df[df['Portfolio_u'].eq(portfolio)].copy()
        portfolio_snaps = np.sort(tmp_portfolio['Period'].unique())
        
        indices = np.concatenate([[0],  
                                np.arange(time_granularity - 1, len(portfolio_snaps), time_granularity)  # Subsequent periods
                                ])

        valid_indices = indices[indices < len(portfolio_snaps)]
        sampled_periods = portfolio_snaps[valid_indices].tolist()

        if len(sampled_periods) > 0:
            matrices_norm_j = np.zeros((len(sampled_periods) - 1 , nx, nx))

            for j, quarter in enumerate(sampled_periods):
                if j == len(sampled_periods) - 1:
                    # print(f'Done for portfolio: {portfolio}\n')
                    break

                start_df = tmp_portfolio[tmp_portfolio['Period'] == sampled_periods[j]]
                next_df = tmp_portfolio[tmp_portfolio['Period'] == sampled_periods[j + 1]]
                #print(f'Going from {start_df['mPeriod'].unique()} -> {next_df['mPeriod'].unique()}')

                matrix_norm = migration_matrix_val(
                    start_df = start_df, 
                    next_df = next_df, 
                    params = params
                )

                matrices_norm_j[j] = matrix_norm
            matrices_dict[portfolio] = matrices_norm_j
        else:
            continue

    return matrices_dict


def main_bootstrap_loop(
    df: pd.DataFrame,
    params: Any,
    frac_value: float = 0.2,
    n_it: int = 10,
    replace: bool = True
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:

    random_state = 42 # random_state start, then increases by random_increase_it
    random_step = 1
    it = 0
    full_dict = {}

    loans_samp_snap_dict, sampled_snaps_dict = get_sampled_snapshots(
        df = df,
        time_resolution = params.time_resolution
    )

    ptfs_list = list(loans_samp_snap_dict.keys())

    # Main iteration loop
    while it < n_it:
        full_dict[it] = {}
        for i in range(params.max_obs_per_ptf):
            #print(f'\n{i} \n')
            full_dict[it][i] = {}
            for ptf in ptfs_list:
                if i + 1 >= sampled_snaps_dict[ptf].size:
                    continue

                mask_ptf = df['Portfolio_u'].eq(ptf)
                mask_snp = df['Period'] == sampled_snaps_dict[ptf][i]
                mask_nxt_snp = df['Period'] == sampled_snaps_dict[ptf][i + 1]

                tmp_ptf_df = df[mask_ptf & mask_snp].copy()
                tmp_nxt_ptf_df = df[mask_ptf & mask_nxt_snp].copy()

                strat_sample_df = tmp_ptf_df.groupby('nPrommiseLoanStatus', 
                                                     group_keys = False
                                                     ).apply(lambda group: 
                                                             safe_sample(group, 
                                                                         frac = frac_value,
                                                                         replace = replace,
                                                                         random_state = random_state
                                                                         )
                                                                         ).reset_index(drop = True)
                
                matrix_bank = migration_matrix_val(
                    start_df = strat_sample_df,
                    next_df = tmp_nxt_ptf_df,
                    params = params
                )
                
                try:
                    check_matrix_rows(matrix_bank) #Check the matrix
                except ValueError as e:
                    print(f'Error in portfolio {ptf}, snapshot {i}: {e}')
                    break
                    
                # Store matrix_bank in the dictionary
                if ptf not in full_dict[it][i]:
                    full_dict[it][i][ptf] = []
                full_dict[it][i][ptf].append(matrix_bank)

        # print(f'Iteration numbere: {it}')
        it += 1
        random_state += random_step 

    return full_dict


def cum_pd_sum(
    multi_year_matrix: np.array, 
    starting_row: int
):
    return np.array([mat_ti[starting_row, -2:].sum() for mat_ti in multi_year_matrix])


def matrix_projection(
    matrix: np.array,
    years_projection: int
) -> np.ndarray:
    
    proj_arr = np.zeros( (years_projection, ) )
    proj_arr[0] = matrix[0, -2:].sum()
    final_matrix = matrix.copy()

    for i in range(1, years_projection):
        final_matrix = final_matrix @ matrix
        proj_arr[i] = final_matrix[0, -2:].sum()

    return proj_arr


def cum_pd_plot_val(
    benchmark_curve_arr: np.ndarray,
    bootstrap_curve_arr: np.ndarray,
    frac_value: float,
    n_it: int,
    y_max_plot: float = 20,
    years_projection: int = 10,
    y_lines_step: float = 2.5,
    add_title: str = '',
    short: bool = False
) -> tuple:

    fig, ax = plt.subplots(figsize = (12, 6), dpi = 300.)

    perc_bootstrap_curve = 100 * bootstrap_curve_arr
    perc_benchmark_curve = 100 * benchmark_curve_arr

    for i in range(bootstrap_curve_arr.shape[0]):
        if short:
            plt.plot(range(1, 3 + 1), perc_bootstrap_curve[i, :3])#, label = f'Iteration {iteration_key}')
        else:
            plt.plot(range(1, years_projection + 1), perc_bootstrap_curve[i, :])#, label = f'Iteration {iteration_key}')

    # Find the maximum and minimum values for each year
    max_values = np.max(perc_bootstrap_curve, axis = 0)
    min_values = np.min(perc_bootstrap_curve, axis = 0)

    # Shade the area between the maximum and minimum curves
    years = range(1, years_projection + 1)
    if short:
        years = years[:3]
        years_projection = 3
        plt.fill_between(
            years, 
            min_values[:3], 
            max_values[:3], 
            color = 'gray', 
            alpha = 0.3, 
            label = 'Range'
        )
    else:
        plt.fill_between(
            years, 
            min_values, 
            max_values, 
            color = 'gray', 
            alpha = 0.3, 
            label = 'Range'
        )

    y_lines = np.arange(0, y_max_plot, y_lines_step)  # Create y lines with a step size of 2.5
    for y in y_lines:
        plt.axhline(y = y, color = 'gray', linestyle = '--', linewidth = 0.5, alpha = 0.5)

    if short:
        plt.plot(
            range(1, years_projection + 1), 
            perc_benchmark_curve[:3], 
            linewidth = 3, 
            marker = 'D', 
            color = 'k', 
            label = 'Full model'
        )
    else:
        plt.plot(
            range(1, years_projection + 1), 
            perc_benchmark_curve, 
            linewidth = 3, 
            marker = 'D', 
            color = 'k', 
            label = 'Full model'
        )

    plt.title(f'Sub-sample percentage: {100 * frac_value}% | Number of iterations: {n_it}')
    plt.suptitle(f'Validation visualisation | {add_title}', 
                fontsize = 16)
    plt.xlabel('Year')
    plt.ylabel('Cumulative PD (%)')

    plt.show()

    return fig, ax

def cum_pd_plot_val2(
        benchmark_curve_arr: np.ndarray,
        bootstrap_curve_arr: np.ndarray,
        frac_value: float,
        n_it: int,
        years_projection: int = 10,
        add_title: str = ''
) -> None:
    # [Previous code remains the same until the histogram section]

    perc_bootstrap_curve = 100 * bootstrap_curve_arr
    perc_benchmark_curve = 100 * benchmark_curve_arr

    fig, axes = plt.subplots(
        nrows = years_projection, 
        ncols = 1, 
        figsize = (12, 2 * years_projection), 
        dpi = 300.,
        sharex = True
    )

    # Add overarching title
    fig.suptitle(f'Sub-sample percentage: {100 * frac_value:.0f}% | Number of iterations: {n_it} | {add_title}', fontsize = 16)
    
    def get_pvalue_color(pvalue):
        if pvalue > 0.1:
            return 'lightgreen'  # Not significant
        elif 0.05 < pvalue <= 0.1:
            return 'yellow'  # Marginally significant
        else:
            return 'salmon'  # Significant
    def get_cohens_color(pvalue):
        if pvalue <= 0.2:
            return 'lightgreen'  # Not significant
        elif 0.2 < pvalue <= 0.5:
            return 'yellow'  # Marginally significant
        else:
            return 'salmon'  # Significant


    for year_idx in range(years_projection):
        ax = axes[year_idx]
        year_values = perc_bootstrap_curve[:, year_idx]
        
        # Calculate statistics
        mean = np.mean(year_values)
        ci_low, ci_high = np.percentile(year_values, [2.5, 97.5])
        
        # Perform t-test
        benchmark_value = perc_benchmark_curve[year_idx]
        t_stat, p_value = ttest_1samp(year_values, benchmark_value)
        
        # Plot histogram
        ax.hist(year_values, bins = 20, color = 'blue', alpha = 0.7, edgecolor = 'black')
        
        # Add vertical lines
        ax.axvline(mean, color = 'red', linestyle = '--', label = f'Mean: {mean:.2f}%')
        ax.axvline(ci_low, color = 'green', linestyle = ':', label = f'95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]')
        ax.axvline(ci_high, color  ='green', linestyle = ':')
        ax.axvline(benchmark_value, color = 'black', label = f'Implemented curve: {benchmark_value:.2f}%')
        
        cohens_d = np.abs( (np.mean(year_values) - benchmark_value) / np.std(year_values) )
        
        # Add p-value box
        box_text_p = f"p-value: {100 * p_value:.2f}%"
        box_text_cohens = f"Cohen's d: {100 * cohens_d:.2f}%"
        
        box_props_p = dict(
            boxstyle = 'round', 
            facecolor = get_pvalue_color(p_value), 
            alpha = 0.5
        )
        box_props_c = dict(
            boxstyle = 'round', 
            facecolor = get_cohens_color(cohens_d), 
            alpha = 0.5
        )

        ax.text(
            0.99, 
            0.95, 
            box_text_p, 
            transform = ax.transAxes,
            verticalalignment = 'top',
            horizontalalignment = 'right',
            bbox = box_props_p
        )

        ax.text(
            0.99, 
            0.78, 
            box_text_cohens, 
            transform = ax.transAxes,
            verticalalignment = 'top',
            horizontalalignment = 'right',
            bbox = box_props_c
        )

        ax.set_title(f'Year {year_idx + 1}')
        ax.set_ylabel('Frequency')
        ax.legend()

    axes[-1].set_xlabel('Cumulative PD (%)')
    plt.tight_layout(rect = [0, 0, 1, 0.98])
    plt.show()
