import os
from typing import List, Tuple, Union

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
        portfolio_name: Union[str, List[str]],
        reference_snapshot: Union[str, int, bool],
        test: bool = False,
        perf_def_split: bool = False,
        sicr_active: bool = False
    ):
        """
        Initialize IFRS9 Engine with portfolio name(s) and reference snapshot
        
        Args:
            portfolio_name: Single portfolio name or list of portfolio names
            reference_snapshot: Reference snapshot identifier
        """

        self.portfolio_name = portfolio_name
        self.reference_snapshot = reference_snapshot

        # Parameters
        self.params_data_prep = _00_fmo_parameters.DataPreparationParameters()
        self.params_application = _00_fmo_parameters.ApplicationParameters()
        self.params_ccf = _00_fmo_parameters.CreditConversionFactorParameters()
        self.params_risk = _10_fmo_json.load_risk_parameters('./src/risk_parameters3.json')

        self.marginal_pd_curves_dict =_09_fmo_engine.marginal_pd_arr_set_up(
            risk_params = self.params_risk,
            params = self.params_application
            )
        
        self.portfolio_list = self.create_portfolio(
            portfolio_name = self.portfolio_name,
            reference_snapshot = reference_snapshot,
            test = test,
            perf_def_split = perf_def_split,
            sicr_active = sicr_active
        )


    def _defaulted_redeemed_merging(
        self,
        ass_brr_df: pd.DataFrame,
        portfolio_name: str,
        reference_snapshot: str | int
    ) -> Union[str, Tuple[str, pd.DataFrame]]:
        """
        """

        required_columns_list_dr = ['Loan_ID', 'Period', 'Loan_BalanceCurrent', 'BalanceAtDefault']
        
        path_to_portfolio_dr_files = os.path.join(self.params_ccf.path_to_def_red_folders, portfolio_name)
        def_red_files_list = os.listdir(path_to_portfolio_dr_files)

        matched_item = next(
            ((i, s) for i, s in enumerate(def_red_files_list) if s[:6] == str(reference_snapshot)), None
        )

        if matched_item:
            def_red_df = pd.read_csv(
                os.path.join(path_to_portfolio_dr_files, matched_item[1]),
                delimiter = ';',
                low_memory = False 
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
        reference_snapshot: Union[str, int],
        test: bool = False,
        perf_def_split: bool = False,
        sicr_active: bool = False
    ) -> Portfolio:
        """
        Create a single portfolio instance
        
        Args:
            portfolio_name: Name of the portfolio
            reference_snapshot: Reference snapshot identifier
            
        Returns:
            Portfolio instance
        """

        ptf_df, ptf_age = _01_fmo_utils_fun.portfolio_snapshot_loader(
            ptf = portfolio_name,
            reference_snapshot = reference_snapshot,
            params = self.params_data_prep,
            test = test        
        )

        # test = True
        if not test:
            exposure_column, ptf_df = self._defaulted_redeemed_merging(
                ass_brr_df = ptf_df,
                portfolio_name = portfolio_name,
                reference_snapshot = reference_snapshot
            )
        else:
            exposure_column = 'Loan_BalanceCurrent'

        return Portfolio(
            portfolio = ptf_df['Portfolio_u'].unique()[0],
            currency = ptf_df['Loan_Currency'].unique()[0],
            country = ptf_df['Country_u'].unique()[0],
            geography = ptf_df['Geography'].unique()[0],
            date = ptf_df['date'].unique()[0],
            
            age = ptf_age,
            exposure_column = exposure_column,
            exposure = ptf_df[exposure_column].sum(),
            guarantee = 0.,
            sid = self.params_application.early_sid_dict[ptf_df['Portfolio_u'].unique()[0]],
            scheduled_sid = self.params_application.scheduled_sid_dict[ptf_df['Portfolio_u'].unique()[0]],
            cover_stop_date = self.params_application.cover_stop_date_dict[ptf_df['Portfolio_u'].unique()[0]],
            
            reference_snapshot = ptf_df,
            ptf_marginal_pd_dict = self.marginal_pd_curves_dict[ptf_df['Geography'].unique()[0]],
            lgd_arr = self.params_risk['lgd'],
            perf_def_split = perf_def_split,
            sicr_active = sicr_active
        )

    def create_portfolio(
        self,
        portfolio_name: Union[str, List[str]],
        reference_snapshot: Union[str, int, bool],
        test: bool = False,
        perf_def_split: bool = False,
        sicr_active: bool = False
    ) -> Union[Portfolio, List[Portfolio]]:
        """
        Create portfolio(s) based on input name(s)
        
        Args:
            portfolio_name: Single portfolio name or list of portfolio names
            reference_snapshot: Reference snapshot identifier
            
        Returns:
            Single Portfolio instance or list of Portfolio instances
        """
        if isinstance(portfolio_name, list):
            # Looping through all portfolios
            return [
                self._create_single_portfolio(
                    portfolio, 
                    reference_snapshot,
                    test = test,
                    perf_def_split = perf_def_split,
                    sicr_active = sicr_active,
                )
                for portfolio in portfolio_name
            ]
        return [
            self._create_single_portfolio(
                portfolio_name, 
                reference_snapshot,
                test = test,
                perf_def_split = perf_def_split,
                sicr_active = sicr_active
            )]
    

    def portfolio_ccf_loop(
        self,
        ptf_obj: Portfolio,
    ) -> None:
        """
        
        """

        ccf_selected = _09_fmo_engine.ccf_selection(
            ptf_obj.age,
            ccf_matrix = self.params_risk['ccf'],
            params = self.params_ccf
        )

        ptf_obj.cashflow_schedule = ptf_obj.ccf_preprocessing(
            params = self.params_ccf
        )

        ptf_obj.cashflow_schedule = ptf_obj.ccf_application(
            ccf_selected = ccf_selected,
            marginal_pd_dict = ptf_obj.ptf_marginal_pd_dict,
            lgd_guarantee = self.params_risk['lgd'][1],
            params = self.params_application
        )
