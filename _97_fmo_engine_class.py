import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# Custom libraries
from src import _00_fmo_parameters
from src import _01_fmo_utils_fun
from src import _09_fmo_engine
from src import _10_fmo_json

# Custom classes
from ._98_fmo_portfolio_class import Portfolio

class IFRS9Engine:
    def __init__(
        self,
        portfolio_name: str | List[str],
        reference_snapshot: str |int | bool,
        origination: bool = False
    ):
        """
        Initialize IFRS9 Engine with portfolio name(s) and reference snapshot(s)
        
        Arguments:
            portfolio_name: Single portfolio name or list of portfolio names
            reference_snapshot: Reference snapshot identifier
        """
        self.params_application = _00_fmo_parameters.ApplicationParameters()
        self.params_ccf = _00_fmo_parameters.CreditConversionFactorParameters()
        self.params_data_prep = _00_fmo_parameters.DataPreparationParameters()
        self.params_risk = _10_fmo_json.load_risk_parameters('./src/risk_parameters6.json')

        self.portfolio_name = portfolio_name
        self.reference_snapshot = reference_snapshot

        # Calculate marginal pd
        self.marginal_pd_curves_dict =_09_fmo_engine.marginal_pd_arr_set_up(
            risk_params = self.params_risk,
            params = self.params_application
            )

        # Create Portfolio object(s)
        self.portfolio_list = self.create_portfolio(
            portfolio_name = self.portfolio_name,
            reference_snapshot = reference_snapshot,
            origination = origination
        )

    def _defaulted_redeemed_merging(
        self,
        ass_brr_df: pd.DataFrame,
        portfolio_name: str,
        reference_snapshot: int
    ) -> str | Tuple[str, pd.DataFrame]:
        """
        First, relevant Defaulted&Redeemed .csv file is found and loaded. Then merging of Loan_BalanceCurrent and BalanceAtDefault, for relevant cases, is performed. 
        If for a given Loan_ID, the BalanceAtDefault column is not nan, then BalanceAtDefault overrides the Loan_BalanceCurrent information; 
        otherwise, Loan_BalanceCurrent stays. 
        A check on the presence of the D&R file is performed, should it be missing, Loan_BalanceCurrent is considered
        
        Arguments:
        ass_brr_df         -- pd.DataFrame: DataFrame of the Assets&Borrowers .csv file for the given FI, and reference snapshot
        porfolio_name      -- str: FI's name
        reference_snapshot -- int: reference snapshot provided as yyyymm, for example December 2024 -> 202412

        Returns:
            str          -- name of the new column to be considered in future manipulations, this is only if merging actually occurs
            pd.DataFrame -- merged DataFrame
        """

        # Required columns in Defaulted&Redeemed files
        required_columns_list_dr = ['Loan_ID', 'Period', 'Loan_BalanceCurrent', 'BalanceAtDefault']

        path_to_portfolio_dr_files = os.path.join(self.params_ccf.path_to_def_red_folders, portfolio_name)
        def_red_files_list = os.listdir(path_to_portfolio_dr_files)

        # Search for the correct Defaulted&Redeemd report
        matched_item = next(
            ((i, s) for i, s in enumerate(def_red_files_list) if s[:6] == str(reference_snapshot)), None
        )

        if matched_item:
            def_red_df = pd.read_csv(
                os.path.join(path_to_portfolio_dr_files, matched_item[1]),
                delimiter = ';',
                low_memory = False 
            )

            # Check that Defaulted&Redeemed file has the needed columns
            _01_fmo_utils_fun.validate_df_columns(
                df = def_red_df,
                required_columns_list = required_columns_list_dr
            )

            merged_df = ass_brr_df.merge(
                right = def_red_df[required_columns_list_dr],
                on = ['Loan_ID', 'Period'],
                how = 'left',
                suffixes = ('', '_dr')
            )
    
            merged_df = merged_df.assign(
                BalanceCurrent = lambda x: np.where(
                    (x['Loan_BalanceCurrent'] == x['BalanceAtDefault']) | x['BalanceAtDefault'].isna(), 
                    x['Loan_BalanceCurrent'], 
                    x['BalanceAtDefault']
                )
            )
            return 'BalanceCurrent', merged_df

        else:
            print(f'Defaulted & Redeemed file for {reference_snapshot} is missing')

            return 'Loan_BalanceCurrent'


    def _create_single_portfolio(
        self,
        portfolio_name: str,
        reference_snapshot: int,
        origination: bool = False
    ) -> Portfolio:
        """
        Create a single portfolio instance. Origination is a boolean handling the tool's behaviour for origination purposes. If set to True, the engine
        searches the portfolio_name, and reference snapshot in a different folder, which is specified in _00_parameters.py.
        
        Arguments:
            origination        -- bool: boolean managing the use of the engine at origination
            portfolio_name     -- str: Name of the portfolio
            reference_snapshot -- int: Reference snapshot identifier
            
        Returns:
            Portfolio -- instance
        """

        # Load and pre-process reference snapshot
        ptf_df, ptf_age = _01_fmo_utils_fun.portfolio_snapshot_loader(
            ptf = portfolio_name,
            reference_snapshot = reference_snapshot,
            params = self.params_data_prep,
            origination = origination        
        )

        # Load cured and liquidated LGD values
        lgd_arr_liq_cur = self.params_risk['lgd']

        geography_group = ptf_df['Geography'].unique()[0]

        # Calculated final LGD value based on P(cure)
        weighted_lgd_dict = _09_fmo_engine.maturity_based_lgd(
            lgd_arr = lgd_arr_liq_cur,
            geography = geography_group,
            migration_matrix_dict = self.params_risk['migration_matrix'],
        )

        if not origination:
            exposure_column, ptf_df = self._defaulted_redeemed_merging(
                ass_brr_df = ptf_df,
                portfolio_name = portfolio_name,
                reference_snapshot = reference_snapshot
            )
        else:
            exposure_column = 'Loan_BalanceCurrent'

        return Portfolio(
            age = ptf_age,
            country = ptf_df['Country_u'].unique()[0],
            covered_interest_bool = self.params_application.covered_interest_dict[ptf_df['Portfolio_u'].unique()[0]],
            cover_stop_date = self.params_application.cover_stop_date_dict[ptf_df['Portfolio_u'].unique()[0]],
            currency = ptf_df['Loan_Currency'].unique()[0],
            date = ptf_df['date'].unique()[0],
            exposure = ptf_df[exposure_column].sum(),
            exposure_column = exposure_column,
            geography = ptf_df['Geography'].unique()[0],
            guarantee = self.params_application.guarantee_dict[ptf_df['Portfolio_u'].unique()[0]],
            lgd_application_dict = weighted_lgd_dict,
            portfolio = ptf_df['Portfolio_u'].unique()[0],
            ptf_marginal_pd_dict = self.marginal_pd_curves_dict[ptf_df['Geography'].unique()[0]],
            reference_snapshot = ptf_df,
            scheduled_sid = self.params_application.scheduled_sid_dict[ptf_df['Portfolio_u'].unique()[0]],
            sid = self.params_application.early_sid_dict[ptf_df['Portfolio_u'].unique()[0]]
        )

    def create_portfolio(
        self,
        portfolio_name: str | List[str],
        reference_snapshot: int,
        origination: bool = False
    ) -> Portfolio | List[Portfolio]:
        """
        Wrapper on _create_portfolio(), if portfolio_name is a list, then .create_single_portfolio is iterated through 
        each name in the list. 
        
        Arguments:
            portfolio_name     -- str | List[str]: Single portfolio name or list of portfolio names
            reference_snapshot -- int: Reference snapshot identifier
            
        Returns:
            Single Portfolio object or list of Portfolio objects
        """
        if isinstance(portfolio_name, list):
            # Looping through all portfolios
            return [
                self._create_single_portfolio(
                    portfolio, 
                    reference_snapshot,
                    origination = origination
                )
                for portfolio in portfolio_name
            ]
        return [
            self._create_single_portfolio(
                portfolio_name, 
                reference_snapshot,
                origination = origination
            )
        ]

    def new_portfolio_ccf_loop(
        self,
        ptf_obj: Portfolio,
        sid_threshold: float = 0.1
    ) -> None:
        """
        Main function for the application of CCF, and calculation of the early stop inclusion date (SID).
        Function directly updates the DataFrame cashflow_schedule

        Arguments:
        ptf_obj       -- Portfolio: portfolio class under analysis
        sid_threshold -- float:  threshold over which sid is activated, default value is 0.1 (10%)
        
        Returns:
            None
        """

        # Select CCF row to apply based on vintage (portfolio's age)
        ccf_selected = _09_fmo_engine.ccf_selection(
            portfolio_age = ptf_obj.age,
            ccf_matrix = self.params_risk['K'],
            params = self.params_ccf
        )

        # sid_first_run = self.params_application.early_sid_dict[ptf_obj.portfolio] if isinstance(self.params_application.early_sid_dict[ptf_obj.portfolio], pd.Timestamp) else None

        # Application of CCF, and computation of expected loss (EL)
        ptf_obj.cashflow_schedule = ptf_obj.ccf_application(
            ccf_row = ccf_selected,
            sid_date = None,
            params = self.params_application,
            lgd = self.params_risk['lgd'][0]
        )

        # Check whether sid trigger exceeds 10%
        sid_row = ptf_obj.sid_check(
            params_application = self.params_application,
            ccf_row = ccf_selected,
            threshold = sid_threshold,
            lgd = self.params_risk['lgd'][0]
        )

        old_condition_sid_check = (ptf_obj.cashflow_schedule.loc[:sid_row, 'Percentage Cum PD'] >= sid_threshold).sum()
        count = 0

        while old_condition_sid_check:
            loop_sid_row = ptf_obj.sid_check(
                params_application = self.params_application,
                ccf_row = ccf_selected,
                threshold = sid_threshold,
                lgd = self.params_risk['lgd'][0]
            )

            condition_new_sid_check = (ptf_obj.cashflow_schedule.loc[:loop_sid_row-1, 'Percentage Cum PD'] >= sid_threshold).sum()
            if old_condition_sid_check == condition_new_sid_check:
                break
            else:
                old_condition_sid_check = condition_new_sid_check
                count += 1
                if count >= 30:
                    print(count)
                    break
