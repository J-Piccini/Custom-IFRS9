import os
from typing import Any, List,Tuple

import pandas as pd
import numpy as np


def validate_df_columns(
    df: pd.DataFrame,
    required_columns_list: List[str]
) -> None:
    """
    Standardized way to check that all required columns exist in a given DataFrame.
    
    Arguments:
        df               -- pd.DataFrame: Input DataFrame
        required_columns -- list: List of column names to validate
    
    Raises:
        ValueError: If any required column is missing
    """
    missing_columns = [col for col in required_columns_list if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

def portfolio_loader(
    ptf_full_str: str,
    params: Any,
) -> pd.DataFrame:
    
    """
    Load and process portfolio CSV files from specified directory structure.
    
    This function traverses through portfolio folders, loads CSV files, and combines them
    into a single DataFrame. It handles special cases for mixed data types in Loan_IDs
    and provides options for calibration mode.
    
    Arguments:
    ptf_full_str -- str: Path to the main folder containing portfolios' sub-folders.The folder structure should be: main_folder/portfolio_folders/csv_files
    params       -- Any: Parameters dataclass
    
    Returns:
        pd.DataFrame -- Dataframe with all snapshots from all banks appended
    """
    ptf_list = [
        ptf_folder for ptf_folder in os.listdir(ptf_full_str) 
        if os.path.isdir(os.path.join(ptf_full_str, ptf_folder))
        ]
    folders_list = [
        os.path.join(ptf_full_str, ptf_folder) for ptf_folder in os.listdir(ptf_full_str) 
        if os.path.isdir(os.path.join(ptf_full_str, ptf_folder))
        ]

    data_frame_list = []
    for i, folder in enumerate(folders_list):
        files_in_folder = os.listdir(folder)
        for file_i in files_in_folder:
            if file_i.endswith('.csv'):
                ptf_snap_df = pd.read_csv(os.path.join(folder, file_i), 
                                        delimiter = ';', 
                                        low_memory = False 
                                        # Needed to handle different types in same column. For example, certain banks use numbers as Loan_IDs while other use alphanumerical values (strings)
                                        )

            if ptf_list[i] in (params.mixed_types_ids_portfs):
                ptf_snap_df['Loan_ID'] = ptf_snap_df['Loan_ID'].apply(
                    lambda x: str(int(float(x))) 
                    if pd.notna(x) and (isinstance(x, (int, float)) or 
                                        (isinstance(x, str) and x.replace('.', '', 1).isdigit())
                                        )
                                        else x              
                                        )
            ptf_snap_df = ptf_snap_df.dropna(subset = ['Loan_ID']) # Drop untrackable loans
            data_frame_list.append(ptf_snap_df)

    if data_frame_list:
        combined_df = pd.concat(data_frame_list, ignore_index = True)
        return combined_df
    return pd.DataFrame


def reedemed_removal(
    df: pd.DataFrame,
    redeemed_label: str = 'Redeemed'
) -> pd.DataFrame:
    """
    Remove loans that eventually get flagged as "Redeemed".

    Arguments: 
    df             -- pd.DataFrame: Input DataFrame
    redeemed_label -- str: string to be removed

    Returns:
        pd.DataFrame: Clean Dataframe
    """
    mask_redeemed = df['PrommiseLoanStatus'].eq(redeemed_label)
    redeemed_loans = df[mask_redeemed]['Loan_ID'].unique()

    return df[~df['Loan_ID'].isin(redeemed_loans)]


def vitas_removal(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove Vitas portfolio

    Arguments: 
    df -- pd.DataFrame: Input DataFrame

    Returns:
        pd.DataFrame: Clean Dataframe
    """
    return df[~df['Portfolio'].eq('Vitas Palestine')]


def missing_maturity_calculator(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate missing maturity at origination based on the start and end date.

    Arguments: 
    df -- pd.DataFrame: Input DataFrame

    Returns:
        pd.DataFrame: Clean Dataframe
    """
    mask_missing_mat = df['Loan_TenureMonths'].isna()

    df.loc[mask_missing_mat, 'Loan_TenureMonths'] = (
        (
            pd.to_datetime(df.loc[mask_missing_mat, 'Loan_EndDate'],
                           errors = 'coerce',
                           dayfirst = True
                           ) - 
            pd.to_datetime(df.loc[mask_missing_mat, 'Loan_StartDate'],
                           errors = 'coerce',
                           dayfirst = True
                           )
        ).dt.days / 365.25 * 12
    ).round().clip(lower = 0).astype(int)

    return df

def data_preprocess(
    application_bool: bool,
    df: pd.DataFrame,
    params: Any
) -> pd.DataFrame:
    """
    Function for standardizing information. The number of columns carried on is also reduced. The set of essential information is based on the phase of the program,
    i.e. application, rather than calibration.

    Arguments:
    application_bool -- bool: Value used to determine the set of essential information
    df               -- pd.DataFrame: Input DataFrame
    params           -- Any: Dataclass with relevant constants

    Returns:
        pd.DataFrame

    """
    
    
    if application_bool:
        # Extended set of columns for application purposes
        validate_df_columns(
            df = df,
            required_columns_list = params.ptf_class_columns_short_list
        )
        df_shortlist = df[params.ptf_class_columns_short_list].copy()
    else:
        validate_df_columns(
            df = df,
            required_columns_list = params.columns_short_list
        )
        df_shortlist = df[params.columns_short_list].copy()
    
    
    df_red = reedemed_removal(df = df_shortlist)    
    df_cln = vitas_removal(df = df_red)
    df_fnl = missing_maturity_calculator(df = df_cln)

    ##### Portfolios' names uniformation and Country Mapping ####
    df_fnl = df_fnl.assign(Portfolio_u = df_fnl['Portfolio'].map(params.ptf_mapping_dict))
    df_fnl = df_fnl.assign(Country_u = df_fnl['Portfolio_u'].map(params.ptf_to_country_mapping_dict))
    df_fnl = df_fnl.assign(Geography = df_fnl['Country_u'].map(params.geographical_grouping_dict))

    df_fnl['Loan_StartDate'] = pd.to_datetime(
        df_fnl['Loan_StartDate'], 
        errors = 'coerce', 
        dayfirst = True
    )

    df_fnl['Loan_EndDate'] = pd.to_datetime(
        df_fnl['Loan_EndDate'], 
        errors = 'coerce', 
        dayfirst = True
    )
    mask_missing_end_date = df_fnl['Loan_EndDate'].isna()

    if mask_missing_end_date.sum() > 0:
        print('missing end dates')
    df_fnl.loc[mask_missing_end_date, 'Loan_EndDate'] = df_fnl.loc[mask_missing_end_date].apply(
        lambda row: row['Loan_StartDate'] + pd.DateOffset(months=row['Loan_TenureMonths']), 
        axis = 1
    )

    # Loan_TenureMonths overriding
    df_fnl['Loan_TenureMonths'] = (( df_fnl['Loan_EndDate'] - df_fnl['Loan_StartDate']).dt.days / 365.12 * 12).round()

    # Tenor calculation
    df_fnl['months_remaining'] = (
        df_fnl['Loan_TenureMonths'] - 
        (
            (
                pd.to_datetime(df_fnl['Reporting_Date'], 
                               errors = 'coerce', 
                               dayfirst = True
                               ) - 
                pd.to_datetime(df_fnl['Loan_StartDate'], 
                               errors = 'coerce', 
                               dayfirst = True)
            ).dt.days / 365.25 * 12
        ).round()
    ).clip(lower = 0).astype(int)

    # Defnition of loans' statuses conditions
    conditions = [
        (
            (df_fnl['Loan_Arrear_days'].eq(0)| df_fnl['Loan_Arrear_days'].isna()) &
            (df_fnl['PrommiseLoanStatus'] != 'Arrears')
        ),
        (
            df_fnl['Loan_Arrear_days'].between(1, params.arr_threshold, inclusive = 'left')
        ),
        (
            df_fnl['Loan_Arrear_days'].between(params.arr_threshold, 90, inclusive = 'left')
        ),
        (
            (df_fnl['Loan_Arrear_days'].between(90, params.def_threshold, inclusive = 'left')) &
            (df_fnl['PrommiseLoanStatus'].isin(['Default with recovery', 'Default or Foreclosure']))
        ),
        (
            (df_fnl['Loan_Arrear_days'] >= params.def_threshold) &
            (df_fnl['PrommiseLoanStatus'].isin(['Default with recovery', 'Default or Foreclosure']))
        )
        ]
    
    # Default value should there be a loan breaking the defined conditions
    default_value_str = 'Unknown'
    default_value_num = 99

    # Definition of a numerical loan status (int) used in subsequent operations, and of a string loan status, used for debugging
    df_fnl = df_fnl.assign(
        sPrommiseLoanStatus = np.select(
            conditions, 
            params.str_labels, 
            default = default_value_str
        ),
        nPrommiseLoanStatus = np.select(
            conditions, 
            params.num_labels, 
            default = default_value_num
        )
    )

    uncategorized_mask = (
        (df_fnl['nPrommiseLoanStatus'].eq(default_value_num)) | (df_fnl['sPrommiseLoanStatus'].eq(default_value_str))
    )
    

    # Handling weird loans status format that could not be classified
    if uncategorized_mask.any():
        problematic_loans = df_cln[uncategorized_mask]
        error_msg = (
            f"Found {len(problematic_loans)} loans that could not be categorized.\n"
            f"First few problematic loans:\n"
            f"Loan_Arrear_days: {problematic_loans['Loan_Arrear_days'].head().tolist()}\n"
            f"PrommiseLoanStatus: {problematic_loans['PrommiseLoanStatus'].head().tolist()}"
        )
        raise ValueError(error_msg)
        
    return df_fnl

def portfolio_preprocess(
    df: pd.DataFrame,
    params: Any
) -> pd.DataFrame:
    """
    Ad-hoc function for prepocessing dataframe when building a Portfolio object.

    Arguments:
    df     -- pd.DataFrame(): Input DataFrame
    params -- Any: Parameters' dataclass

    Returns:
        pd.DataFrame: refined DataFrame
    """

    # Standard preprocessing
    df2 = data_preprocess(
        df = df,
        application_bool = True,
        params = params
    )

    ##### Portfolio Maps #####
    repayment_type_map = {
            'Lineair': 'linear', 
            'Interest Only': 'interest_only', 
            'Annuity': 'annuity',
            'Bullet': 'bullet', 
            'Irregular': 'bullet', 
            'Unknown': 'bullet',
    }

    payment_frequency_map = {
            'Infrequent': '3 Months',
    }

    df2['Repayment Type'] = np.where(df2['TypeRepayment'].isna(), 
                                    'bullet', 
                                    df2['TypeRepayment'].map(repayment_type_map)
                                    )
    
    df2['PaymentFrequency'] = df2['PaymentFrequency'].fillna('3 Months')
    df2['payment_frequency'] = df2['PaymentFrequency'].copy()

    df2['payment_frequency'] = df2['PaymentFrequency'].map(payment_frequency_map).fillna(df2['PaymentFrequency'])
    df2['payment_frequency'] = df2['payment_frequency'].str.split('Months').str[0].astype(int)
    
    df2['date'] = pd.to_datetime(
        df2['Reporting_Date'], 
        errors = 'coerce', 
        dayfirst = True
    )
    df2['StartDate'] = pd.to_datetime(
        df2['Loan_StartDate'], 
        errors = 'coerce', 
        dayfirst = True
    )

    df2['EndDate'] = pd.to_datetime(
        df2['Loan_EndDate'], 
            errors = 'coerce', 
            dayfirst = True
)
    
    df2['months_remaining'] = (
        (
            (df2['EndDate'] - df2['date']).dt.days / 365.25 * 12
        ).round()
    ).clip(lower = 0).astype(int)
    
    return df2

def portfolio_snapshot_loader(
    params: Any,
    ptf: str,
    reference_snapshot: int | str,
    origination: bool = False
) -> Tuple[pd.DataFrame, int]:
    """
    Builds the Portfolio object based at the reference snapshot.

    Arguments:
    path_to_ptf_folder -- str: Full path to the portfolio folder
    reference_snapshot -- str: Snapshot for which the Expected Credit Loss must be calculated. Must be in formmat yyyymm

    Returns:
        pd.DataFrame: Pre-processed DataFrame
        int         : Portfolio's age, later used in CCF's application
    """

    if origination:
        ptf_full_path = os.path.join(params.path_to_origination_folder, ptf)
    else:
        ptf_full_path = os.path.join(params.path_to_main_folder, ptf)

    ptf_list = [
        ptf_folder for ptf_folder in os.listdir(params.path_to_main_folder) 
        if os.path.isdir(os.path.join(params.path_to_main_folder, ptf_folder))
    ]
    
    if not ptf in ptf_list:
        raise ValueError('Portfolio not in portfolios name list')

    files_in_folder_list = os.listdir(ptf_full_path)
    sorted_files_in_folder_list = sorted(
        [file for file in files_in_folder_list], 
        key = lambda file: file[:6]
    )
    
    if not sorted_files_in_folder_list:
        raise ValueError(f'No snapshot files found in portfolio "{ptf}"')
    
    
       
    ref_snap = str(reference_snapshot) if isinstance(reference_snapshot, int) else reference_snapshot
    matched_item = next(((i, s) for i, s in enumerate(sorted_files_in_folder_list) if s[:6] == ref_snap), None)
    
    portfolio_age = matched_item[0] + 1 # Because of Python
    snapshot_name = matched_item[1]     # File name

    ########## Snapshot Loader ##########
    snap_full_path = os.path.join(ptf_full_path, snapshot_name)
    if origination:
        ptf_snap_df = pd.read_excel(snap_full_path)
    else:
        ptf_snap_df = pd.read_csv(snap_full_path, 
                                delimiter = ';', 
                                low_memory = False 
                                )
    
        if ptf in (params.mixed_types_ids_portfs):
            ptf_snap_df['Loan_ID'] = ptf_snap_df['Loan_ID'].apply(
                lambda x: str(int(float(x))) 
                if pd.notna(x) and (isinstance(x, (int, float)) or 
                                    (isinstance(x, str) and x.replace('.', '', 1).isdigit())
                                    )
                                    else x
            )
            
            ptf_snap_df = ptf_snap_df.dropna(subset = ['Loan_ID'])

    final_df = portfolio_preprocess(
        df = ptf_snap_df,
        params = params
    )
    
    return final_df, portfolio_age