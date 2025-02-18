from typing import List, Union
import pandas as pd

# Custom libraries
import _00_fmo_parameters
import _01_fmo_utils_fun
import _09_fmo_engine
import _10_fmo_json

# Custom classes
from _98_fmo_portfolio_class import Portfolio
from _99_fmo_classes import Loan

class IFRS9Engine:
    def __init__(
        self,
        portfolio_name: Union[str, List[str]],
        reference_snapshot: Union[str, int, bool]
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
        self.params_risk = _10_fmo_json.load_risk_parameters('risk_parameters2.json')
        
        self.marginal_pd_arr = _09_fmo_engine.main_pd_curves(
            risk_params = self.params_risk,
            params = self.params_application
        )

        self.portfolio_list = self.create_portfolio(
            portfolio_name = self.portfolio_name,
            reference_snapshot = self.reference_snapshot
        )


    def _create_single_portfolio(
        self,
        portfolio_name: str,
        reference_snapshot: Union[str, int, bool]
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
            params = self.params_data_prep
        )
        
        return Portfolio(
            portfolio = ptf_df['Portfolio_u'].unique()[0],
            currency = ptf_df['Loan_Currency'].unique()[0],
            country = ptf_df['Country_u'].unique()[0],
            geography = ptf_df['Geography'].unique()[0],
            date = ptf_df['date'].unique()[0],
            
            age = ptf_age,
            exposure = ptf_df['Loan_BalanceCurrent'].sum(),
            guarantee = 0.,
            
            sid = self.params_application.early_sid_dict[ptf_df['Portfolio_u'].unique()[0]],
            scheduled_sid = self.params_application.scheduled_sid_dict[ptf_df['Portfolio_u'].unique()[0]],
            cover_stop_date = self.params_application.cover_stop_date_dict[ptf_df['Portfolio_u'].unique()[0]],
            
            reference_snapshot = ptf_df,
            ptf_marginal_pd_arr = self.marginal_pd_arr
        )

    def create_portfolio(
        self,
        portfolio_name: Union[str, List[str]],
        reference_snapshot: Union[str, int, bool]
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
            return [
                self._create_single_portfolio(portfolio, reference_snapshot)
                for portfolio in portfolio_name
            ]
        return [self._create_single_portfolio(portfolio_name, reference_snapshot)]
    

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
            marginal_pd_arr = self.marginal_pd_arr,
            params = self.params_application
        )

    
    def check_threshold(
        self,
        ptf_obj: Portfolio,
        relative_threshold: float = .1
    ) -> pd.DataFrame:
        """
        """

        # Initialization of the NPL dataframe
        npl_df = pd.DataFrame({
            'Payment Date': ptf_obj.cashflow_schedule['Payment Date'],
            'Exposure': 0.0
        })
        
        if not ptf_obj.npl_list:
            return npl_df

        # Process each NPL loan
        for npl_index in ptf_obj.npl_list:
            npl_loan = ptf_obj.loan_list[npl_index]
            
            # Find matching payment dates between NPL loan and master schedule
            common_dates_mask = npl_df['Payment Date'].isin(npl_loan.el_df['Payment Date'])
            if not common_dates_mask.any():
                continue
                
            # Get matched dates and add exposures
            matched_dates = npl_df.loc[common_dates_mask, 'Payment Date']
            npl_exposures = (
                npl_loan.el_df.set_index('Payment Date').loc[matched_dates, 'Exposure']
                )
            
            npl_df.loc[common_dates_mask, 'Exposure'] += npl_exposures.values

        npl_df['Exposure'] = npl_df['Exposure']

        npl_df['Ratio'] = npl_df['Exposure'] / ptf_obj.exposure
        npl_df['Flag'] = npl_df['Ratio'] > relative_threshold

        return npl_df
