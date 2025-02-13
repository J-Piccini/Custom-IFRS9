from typing import Tuple, Any

import numpy as np
import pandas as pd

from _99_fmo_classes import Loan
"""
A Portfolio class. Contains all information needed to fully characterize the portfolio, and calculate the 
Expected Credit Loss (ECL). 
Keyword Arguments:

portfolio               -- str: Portfolio name
currency                -- str: Currency used for the transactions
country                 -- str: Country name
geography               -- str: Geographical group to which the portfolio belongs, needed to identify the PD curve to be used

age                     -- int: Age of the portfolio. It is used for applying the Credit Conversion Factor (CCF)
exposure                -- float: Current on-book exposure
guarantee               -- float: Guarantee sum
"""

class Portfolio:
    def __init__(
        self,
        ##### Portfolio descriptive characteristics #####
        portfolio: str,
        currency: str,
        country: str,
        geography: str,
        date: pd.Timestamp,
        
        ##### ECL information #####
        age: int,
        exposure: float,
        guarantee: float,

        ##### SID and Cover stop information #####
        sid: bool,
        scheduled_sid: pd.Timestamp,
        cover_stop_date: pd.Timestamp,

        reference_snapshot: pd.DataFrame,
        ptf_marginal_pd_arr: np.ndarray,
    ) -> None:
        
        self.portfolio = portfolio
        self.currency = currency
        self.country = country
        self.geography = geography
        self.date = date

        self.age = age
        self.exposure = exposure
        self.guarantee = guarantee
        
        self.sid = sid
        self.scheduled_sid = scheduled_sid
        self.cover_stop_date = cover_stop_date
        

        self.reference_snapshot = reference_snapshot
        self.ptf_marginal_pd_arr = ptf_marginal_pd_arr


        self.cashflow_schedule = self.future_payments()

        self.loan_list = self.underlying_loan_list(
            df = reference_snapshot
            )


        self.cashflow_schedule['Cumulative PD_Exposure'] = self.cashflow_schedule['PD_Exposure'][::-1].cumsum()

    ##################################################
    ########## Cashflow Calculation Section ##########
    ##################################################
        
    def underlying_loan_list(
        self,
        df: pd.DataFrame
    ) -> list:
        
        loans = []
        for _, row in df.iterrows():
            loan = Loan(
                # Identification attributes
                identifier = row['Loan_ID'],
                portfolio = self.portfolio,
                currency = self.currency,
                status = row['nPrommiseLoanStatus'],
                date = self.date,
                
                # Loan duration attributes
                start_date = row['StartDate'],
                end_date = row['EndDate'],
                maturity = row['Loan_TenureMonths'],
                tenor = row['months_remaining'],
                
                # Loan exposure attributes
                principal = row['Loan_BalanceCurrent'], #max(row['Loan_BalanceOriginal'], row['Loan_BalanceDisbursed']),

                # Loan repayment attributes
                effective_interest_rate = row['Loan_InterestRateEffective'],
                payment_type = row['Repayment Type'],
                payment_frequency = row['payment_frequency'],
                )
            
            loan.el_df = loan.calculate_pd_and_expected_loss(
                df = loan.expected_loss_df_setup(
                    self.cashflow_schedule
                    ),
                marginal_pd_arr = self.ptf_marginal_pd_arr
            )

            self.cashflow_schedule = self.total_monthly_cashflow(
                loan = loan
            )

            loans.append(loan)

        return loans
    

    def future_payments(
        self
    ) -> pd.DataFrame:
        """
        Initialize a DataFrame storing monthly payment dates from reporting date up to, not includign, the cover stop date. The cover stop date is manually added to
        account for those cases with cover stop date not at the end of the month
        """
        payment_df = pd.DataFrame({
            'Payment Date': pd.date_range(start = self.date,
                                          end = self.cover_stop_date,
                                          freq = '1ME',
                                          inclusive = 'left'
                                          )
        })
        
        # Add the final date (cover_stop_date) as a new row
        final_row = pd.DataFrame({
            'Payment Date': [self.cover_stop_date]
        })
        
        # Concatenate the original DataFrame with the final row
        payment_df = pd.concat([payment_df, final_row], ignore_index=True)
        
        return payment_df
    

    def total_monthly_cashflow(
        self,
        loan: Loan
    ) -> pd.DataFrame:
        """
        Computes the total monthly cashflow and updates the Portfolio Expected Loss.
        """

        df = self.cashflow_schedule

        # Identify the common 'Payment Date' values
        common_dates = df['Payment Date'].isin(loan.el_df['Payment Date'])
        # Get the exact row indices where 'Payment Date' matches
        matched_indices = df.index[common_dates]

        if 'PD_Exposure' not in df.columns:
            df['PD_Exposure'] = 0.  # Initialize if column does not exist

        # Ensure element-wise addition
        df.loc[matched_indices, 'PD_Exposure'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'PD_Exposure'].values
        )

        if 'Portfolio Exposure' not in df.columns:
            df['Portfolio Exposure'] = 0.

        df.loc[matched_indices, 'Portfolio Exposure'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'Exposure'].values
        )   

        if 'Portfolio Exposure' in loan.el_df.columns:
            loan.el_df.drop(columns = ['Portfolio Exposure'], inplace = True)
        return df
    

    ##########################################################
    ########## Credit Conversion Factor Application ##########
    ##########################################################

    def ccf_portfolio_preprocessing(
        self,
        params: Any
    ) -> pd.DataFrame:
        """
        
        """

        df = self.cashflow_schedule

        if self.portfolio == 'Ararat':
            guarantee = 20_000_000
        else:
            guarantee = params.ccf_guarantee_dict[self.portfolio]

        df['Available Guarantee'] = guarantee - df['Portfolio Exposure']

        return df


