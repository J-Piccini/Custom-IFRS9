from typing import Tuple, List, Any

import numpy as np
import pandas as pd

import _09_fmo_engine
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

        # Portfolio lifetime schedule 

        # print('Starting future payments')
        self.cashflow_schedule = self.future_payments()
        self.tt = self.future_payments()

        # print('Starting underlying loan list')
        self.loan_list, self.npl_list = self.underlying_loan_list(
            df = reference_snapshot
            )
        
        # print('Starting df post processsing')
        self.df_post_processing()

        self.current_exposure = reference_snapshot['Loan_BalanceCurrent'].sum()
        # )

        # self.cashflow_schedule['Cumulative PD_Exposure'] = self.cashflow_schedule['PD_Exposure'][::-1].cumsum()
        # self.cashflow_schedule['Portfolio Month'] = np.arange(len)

    ##################################################
    ########## Cashflow Calculation Section ##########
    ##################################################
    
    def underlying_loan_list(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[Loan], List[bool]]:
        
        loans = []
        npl_list = []
        for index, row in df.iterrows():
            # print(f'Loan_ID: {row['Loan_ID']}') # for debugging

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
            
            # print(f'Loan_ID: {loan.identifier}\n payment type: {loan.payment_type}')
            loan.test = loan.expected_loss_df_setup(
                portfolio_schedule = self.cashflow_schedule
            )

            loan.el_df = loan.calculate_pd_and_expected_loss(
                df = loan.expected_loss_df_setup(
                    portfolio_schedule = self.cashflow_schedule,
                    ),
                marginal_pd_arr = self.ptf_marginal_pd_arr
            )
            
            self.cashflow_schedule = self.total_monthly_cashflow(
                loan = loan
            )            

            loans.append(loan)
            if row['nPrommiseLoanStatus'] > 2:
                npl_list.append(index)

        return loans, npl_list
    

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
        payment_df = pd.concat(
            [payment_df, final_row], 
            ignore_index = True
        )
        
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
        matched_indices = df.index[common_dates]

        if 'PD*Exposure' not in df.columns:
            df['PD*Exposure'] = 0.  # Initialize if column does not exist
        
        if 'Portfolio Exposure' not in df.columns:
            df['Portfolio Exposure'] = 0.

        # Ensure element-wise addition
        df.loc[matched_indices, 'PD*Exposure'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'PD*Exposure'].values
        )
        
        df.loc[matched_indices, 'Portfolio Exposure'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'Exposure'].values
        )   
        
        if 'Portfolio Exposure' in loan.el_df.columns:
            loan.el_df.drop(columns = ['Portfolio Exposure'], inplace = True)


        return df
    

    def df_post_processing(
        self
    ) -> pd.DataFrame:
        """
        """
        df = self.cashflow_schedule

        df['Cumulative PD_Exposure'] = df['PD*Exposure'][::-1].cumsum()

        return df


    ##########################################################
    ########## Credit Conversion Factor Application ##########
    ##########################################################

    def ccf_preprocessing(
        self,
        params: Any
    ) -> pd.DataFrame:
        """
        
        """
        df = self.cashflow_schedule
        df['Portfolio Month'] = np.arange(len(df)) + 1
        
        df['Portfolio Month Flag'] = df['Portfolio Month'].apply(
            lambda x: min(
                sum([x > threshold for threshold in params.perf_window_list]), 3
                )
            )


        if self.portfolio == 'Ararat':
            # This assumes that EL calculations for Ararat are not performed before the increase
            # in guarantee
            guarantee = 20_000_000
        else:
            guarantee = params.ccf_guarantee_dict[self.portfolio]

        df['Available Guarantee'] = guarantee - df['Portfolio Exposure']

        output_columns = ['Payment Date', 'Portfolio Month', 'Portfolio Month Flag', 'Portfolio Exposure', 
                           'PD*Exposure', 'Cumulative PD_Exposure', 'Available Guarantee']
        return df[output_columns]
    

    def ccf_application(
        self,
        ccf_selected: np.ndarray,
        marginal_pd_arr: np.ndarray,
        params: Any
    ) -> pd.DataFrame:
        """
        """
        
        df = self.cashflow_schedule.copy()
        sid_date = params.scheduled_sid_dict[self.portfolio]

        # Initialize Applied CCF column with zeros
        df['Applied CCF'] = 0.

        # Create mask for payments before SID
        mask_before_scheduled_sid = (
            (df['Payment Date'].dt.year < sid_date.year) |
            ((df['Payment Date'].dt.year == sid_date.year) & 
                (df['Payment Date'].dt.month <= sid_date.month))
        )

        # Apply CCF values only to payments before SID using .loc
        df.loc[mask_before_scheduled_sid, 'Applied CCF'] = ccf_selected[df['Portfolio Month Flag']][mask_before_scheduled_sid]

        df['Estimated Guarantee'] = df['Available Guarantee'] * df['Applied CCF']

        # Assuming the guarantee behaves as a performing loan established at the current snapshot
        df['EG*PD'] = df['Estimated Guarantee'] * marginal_pd_arr[:len(df), 0]

        # Estimated Guarantee Loan Amortized Plan
        df['EGLAP'] = 0.

        mask_after_sid = ~mask_before_scheduled_sid
        if mask_after_sid.any() and mask_before_scheduled_sid.any():
            last_estimated_guarantee = df.loc[mask_before_scheduled_sid, 'Estimated Guarantee'].iloc[-1]
            remaining_periods = len(df) - mask_before_scheduled_sid.sum()
            if remaining_periods > 0:
                df.loc[mask_after_sid, 'EGLAP'] = last_estimated_guarantee / remaining_periods

        df['Total Exposure'] = df['PD*Exposure'] + df['EG*PD'] + df['EGLAP'] * marginal_pd_arr[:len(df), 0]

        ordered_columns = [
            'Payment Date', 'Portfolio Month', 'Portfolio Month Flag', 
            'Portfolio Exposure', 'PD*Exposure', 'Cumulative PD_Exposure', 
            'Available Guarantee', 'Applied CCF', 'Estimated Guarantee', 'EG*PD', 'EGLAP', 'Total Exposure'
        ]
        return df[ordered_columns]
    
