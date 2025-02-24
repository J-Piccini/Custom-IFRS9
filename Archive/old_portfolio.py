from typing import Tuple, List, Any, Dict

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
date                    -- pd.Timestamp: Reporting date of the portfolio snapshot

age                     -- int: Age of the portfolio. Used for applying the Credit Conversion Factor (CCF)
exposure                -- float: Current on-book exposure
guarantee               -- float: Guarantee sum

sid                     -- bool: Flag indicating whether Scheduled Implementation Date has been reached
scheduled_sid           -- pd.Timestamp: Date of the scheduled implementation
cover_stop_date         -- pd.Timestamp: Final date until which the cover/guarantee is valid

reference_snapshot      -- pd.DataFrame: DataFrame containing loan-level data used as reference
ptf_marginal_pd_dict    -- dict: Dictionary containing the marginal PD curves to be applied
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
        ptf_marginal_pd_dict: dict,
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
        self.ptf_marginal_pd_dict = ptf_marginal_pd_dict

        # Portfolio lifetime schedule 
        self.cashflow_schedule = self.future_payments()

        # Creation of a list of Loan objects 
        self.loan_list, self.npl_list = self.underlying_loan_list(
            df = reference_snapshot
            )

        # Further computations
        self.df_post_processing()

        self.lel = 0.
        self.tranche_loss_dict = {} 
        self.fmo_loss_bool = False
        self.stage = 1
        #self.current_exposure = reference_snapshot['Loan_BalanceCurrent'].sum()

    ##################################################
    ########## Cashflow Calculation Section ##########
    ##################################################
    def select_maturity_split(
        self,
        split_variable: int
    ) -> np.ndarray:
        """
        Selects the appropriate maturity curve based on loan term length.
        
        Determines which probability of default (PD) curve to use based on whether
        the loan maturity is shorter or longer than 36 months.
        
        Args:
            split_variable (int): The loan tenure in months
            
        Returns:
            np.ndarray: The appropriate marginal PD array for the given maturity
        """
        if split_variable <= 36:
            return self.ptf_marginal_pd_dict['maturity-lower']
        else:
            return self.ptf_marginal_pd_dict['maturity-upper']


    def underlying_loan_list(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[Loan], List[int]]:
        """
        Creates Loan objects for each loan in the reference snapshot and calculates expected losses.
        
        For each loan in the input DataFrame:
        1. Creates a Loan object with attributes from the snapshot
        2. Sets up the expected loss DataFrame 
        3. Calculates PD and expected loss for each loan
        4. Updates the portfolio's cashflow schedule with the loan's cashflows
        5. Identifies non-performing loans (NPLs)
        
        Args:
            df (pd.DataFrame): Reference snapshot containing loan-level data
            
        Returns:
            Tuple[List[Loan], List[int]]: A tuple containing:
                - List of Loan objects
                - List of indices representing non-performing loans
        """
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
                end_date = row['EndDate'], #max([row['EndDate'], row['date']]),
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
                    portfolio_schedule = self.cashflow_schedule,
                    ),
                marginal_pd_arr = self.select_maturity_split(
                    split_variable = row['Loan_TenureMonths']
                    )
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
        Creates a payment schedule DataFrame from reporting date to cover stop date.
        
        Generates a DataFrame with monthly payment dates starting from the portfolio
        reporting date up to the cover stop date. The cover stop date is explicitly
        added as a final row, even if it doesn't fall on the last day of a month.
        
        Returns:
            pd.DataFrame: DataFrame with 'Payment Date' column containing all monthly
                        payment dates including the final cover stop date
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
        Updates the portfolio cashflow schedule with individual loan contributions.
        
        Aggregates the expected loss contributions from an individual loan into the
        portfolio-level cashflow schedule. Specifically adds the loan's PD*Exposure
        and Exposure values to the corresponding portfolio totals.
        
        Args:
            loan (Loan): The loan object whose cashflows will be added to the portfolio
            
        Returns:
            pd.DataFrame: Updated portfolio cashflow schedule with the loan's contribution
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
        Performs post-processing calculations on the portfolio cashflow schedule.
        
        Currently calculates the cumulative sum of PD*Exposure values in reverse order
        to determine the cumulative expected loss at each payment date.
        
        Returns:
            pd.DataFrame: The processed portfolio cashflow schedule with additional metrics
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
        Prepares the cashflow schedule for Credit Conversion Factor (CCF) application.
        
        Adds portfolio timing information and calculates the available guarantee amount
        for use in CCF calculations. The function:
        1. Adds a 'Portfolio Month' counter
        2. Creates a 'Portfolio Month Flag' based on performance windows
        3. Calculates the 'Available Guarantee' based on portfolio-specific guarantee amounts
        
        Args:
            params (Any): Parameter object containing CCF-related configuration values including
                        performance windows and guarantee amounts by portfolio
            
        Returns:
            pd.DataFrame: Preprocessed cashflow schedule with CCF-related columns
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
        marginal_pd_dict: dict,
        params: Any
    ) -> pd.DataFrame:
        """
        Applies Credit Conversion Factor (CCF) to calculate guarantee-related exposures.
        
        This function:
        1. Applies appropriate CCF values to periods before the Scheduled Implementation Date (SID)
        2. Calculates the estimated guarantee amount based on available guarantee and CCF
        3. Determines the PD-weighted exposure from the estimated guarantee
        4. Creates an amortization plan for the estimated guarantee after SID
        5. Calculates the total exposure including on-book exposures and guarantee-related exposures
        
        Args:
            ccf_selected (np.ndarray): Array of CCF values to apply based on portfolio month flags
            marginal_pd_dict (dict): Dictionary of marginal PD arrays for different maturity buckets
            params (Any): Parameter object containing SID dates and other configuration values
            
        Returns:
            pd.DataFrame: Cashflow schedule with CCF applied and total exposure calculated
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
        if len(df) - mask_before_scheduled_sid.sum() <= 36:
            marginal_pd_arr = marginal_pd_dict['maturity-lower']
        else:
            marginal_pd_arr = marginal_pd_dict['maturity-upper']

        df['EG*PD'] = df['Estimated Guarantee'] * marginal_pd_arr[1 : len(df) + 1, 0]

        # Estimated Guarantee Loan Amortized Plan
        df['EGLAP'] = 0.

        mask_after_sid = ~mask_before_scheduled_sid
        if mask_after_sid.any() and mask_before_scheduled_sid.any():
            last_estimated_guarantee = df.loc[mask_before_scheduled_sid, 'Estimated Guarantee'].iloc[-1]
            remaining_periods = len(df) - mask_before_scheduled_sid.sum()
            if remaining_periods > 0:
                df.loc[mask_after_sid, 'EGLAP'] = last_estimated_guarantee / remaining_periods

        # Term related to actual expsoure on-book + Estimated guarantee used before the s.i.d. through the CCF + Amortized plan of the estimated guarantee loan after s.i.d.
        df['Total Exposure'] = df['PD*Exposure'] + df['EG*PD']

        # Then only add the third component where mask_after_sid is True
        # Ensure the slice length matches the number of True values in mask_after_sid
        num_true_in_mask = mask_after_sid.sum()
        pd_slice = marginal_pd_arr[1 : min(remaining_periods + 1, num_true_in_mask + 1), 0]

        # Only modify the rows where mask_after_sid is True
        if len(pd_slice) > 0:  # Make sure we have values to add
            df.loc[mask_after_sid, 'Total Exposure'] += (
                df.loc[mask_after_sid, 'EGLAP'].iloc[:len(pd_slice)] * pd_slice[:len(df.loc[mask_after_sid])]
            )
        
        ordered_columns = [
            'Payment Date', 'Portfolio Month', 'Portfolio Month Flag', 
            'Portfolio Exposure', 'PD*Exposure', 'Cumulative PD_Exposure', 
            'Available Guarantee', 'Applied CCF', 'Estimated Guarantee', 'EG*PD', 'EGLAP', 'Total Exposure'
        ]
        return df[ordered_columns]
    
    def marginal_el(
        self,
        lgd: float,
    ) -> None:
        
        self.cashflow_schedule['Marginal EL'] = self.cashflow_schedule['Total Exposure'] * lgd


    def tranching_and_lifetime_expected_loss_calculation(
        self,
        tranche_dict: Dict[str, float],
        guarantee_dict: dict
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Calculate the lifetime expected loss (LEL) and its distribution across tranches.
        
        This function computes the total lifetime expected loss from the cashflow schedule 
        and allocates losses to different tranches based on pre-defined rules:
        
        1. When LEL < junior_tranche threshold:
        - Financial Institutions (FIs) absorb losses equal to junior_tranche% of LEL
        - Remaining losses split between European Commission (90%) and MASSIF (10%)
        
        2. When LEL >= junior_tranche threshold:
        - FIs absorb losses up to maximum junior tranche capacity
        - Remaining losses allocated to mezzanine tranches (EC 90%, MASSIF 10%)
        - If losses exceed mezzanine capacity, additional losses go to senior tranche (FMO)
        
        Parameters:
        -----------
        tranche_dict : Dict[str, np.ndarrar]
            Dictionary containing portfolio tranching percentages as an array. The value associated
            with self.portfolio should be an array structured as follows:
            [junior, mezzanine_ec, mezzanine_massif, senior].
        
        guarantee_dict : dict
            Dictionary containing guarantee values for each portfolio, with portfolio
            names as keys and guarantee amounts as values.
        
        Returns:
        --------
        Tuple[float, Dict[str, float], bool]:
            - total_lel: Total lifetime expected loss amount.
            - loss_distribution: Dictionary with loss allocation across tranches:
            {'junior': float, 'mezzanine_ec': float, 'mezzanine_massif': float, 'senior': float}
            - fmo_loss_bool: Boolean flag indicating whether the FMO (senior tranche) 
            experiences any loss.
        """

        tranches_arr = tranche_dict[self.portfolio]
        total_lel = self.cashflow_schedule['Marginal EL'].sum()
        guarantee_val = guarantee_dict[self.portfolio] # To avoid repeated calls
        fmo_loss_bool = False

        # Loss values initialization
        fi_loss = 0.
        ec_loss = 0.
        massif_loss = 0.
        fmo_loss = 0.

        # Maximum junior and mezzanine tranche losses
        max_junior_loss = tranches_arr[0] * guarantee_val
        max_mezzanine_loss = (tranches_arr[1] + tranches_arr[2]) * guarantee_val

        if total_lel <= max_junior_loss:
            # LEL <= junior_tranche% of guarantee limitFIs take junior_tranche% of the LEL, not all of it
            fi_loss = total_lel * tranches_arr[0]

            ec_loss = (total_lel - fi_loss) * 0.9
            massif_loss = (total_lel - fi_loss) * 0.1
        
        else:
            # LEL > junior_tranche% of guarantee limit -> FIs take that, the rest is destributed accordingly
            fi_loss = max_junior_loss
            residual_loss = total_lel - fi_loss
            
            if residual_loss <= max_mezzanine_loss:
                # LEL's portion accounted for by the mezzanine tranch does not exceed the mezzanine tranche limit -> split 90-10
                ec_loss = residual_loss * 0.9
                massif_loss = residual_loss * 0.1

            else: 
                ec_loss = tranches_arr[1] * guarantee_val
                massif_loss = tranches_arr[2] * guarantee_val
                fmo_loss = residual_loss - (ec_loss + massif_loss)

                fmo_loss_bool = True
                # Carrying losses into the FMO tranche cause the portfolio to be automatically move into stage 3
                self.stage = 3

        return total_lel, {'junior': fi_loss,
                           'mezzanine_ec': ec_loss,
                           'mezzanine_massif': massif_loss,
                           'senior': fmo_loss}, fmo_loss_bool
    

    # def stage_expected_loss(
    #     self,
    # ) -> float:
    #     """
        
    #     """
    #     if self.stage == 1:
    #         return self.cashflow_schedule.loc[:12, 'Marginal EL'].sum()
    #     else
    
