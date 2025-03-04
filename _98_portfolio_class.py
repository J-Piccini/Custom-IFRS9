from typing import Tuple, List, Any, Dict, Union

import numpy as np
import pandas as pd

from ._99_fmo_classes import Loan
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
        exposure_column: str,
        exposure: float,
        guarantee: float,

        ##### SID and Cover stop information #####
        sid: bool,
        scheduled_sid: pd.Timestamp,
        cover_stop_date: pd.Timestamp,

        reference_snapshot: pd.DataFrame,
        ptf_marginal_pd_dict: dict,

        perf_def_split: bool = False,
        sicr_active: bool = False,
        lgd_arr: np.ndarray = np.array([0.8754, 0.6210])
    ) -> None:
        
        self.portfolio = portfolio
        self.currency = currency
        self.country = country
        self.geography = geography
        self.date = date

        self.age = age
        self.exposure_column = exposure_column
        self.exposure = exposure
        self.guarantee = guarantee
        
        self.sid = sid
        self.scheduled_sid = scheduled_sid
        self.cover_stop_date = cover_stop_date

        self.reference_snapshot = reference_snapshot
        self.ptf_marginal_pd_dict = ptf_marginal_pd_dict

        # Portfolio lifetime schedule 
        self.cashflow_schedule = self.future_payments()
        self.principal_exposure = 0.
        self.default_loss = 0.

        # Creation of a list of Loan objects 
        self.loan_list, self.npl_list = self.underlying_loan_list(
            df = reference_snapshot,
            perf_def_split = perf_def_split,
            lgd_arr = lgd_arr,
            sicr_active = sicr_active
            )

        # Further computations
        # self.df_post_processing()

        self.lel = 0.
        self.tranche_loss_dict = {} 

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
        df: pd.DataFrame,
        perf_def_split: bool = False,
        lgd_arr: np.ndarray = np.array([0.8754, 0.6210]),
        sicr_active: bool = False
    ) -> Tuple[List[Loan], List[Loan]]:
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
        npl_list = []
        loan_list = []

        for _, row in df.iterrows():
            # print(row['Loan_ID'])
            loan = Loan(
                # Identification attributes
                identifier = row['Loan_ID'],
                portfolio = self.portfolio,
                currency = self.currency,
                status = row['nPrommiseLoanStatus'],
                arrear_days = row['Loan_Arrear_days'],
                date = self.date,
                
                # Loan duration attributes
                start_date = row['StartDate'],
                end_date = row['EndDate'], #max([row['EndDate'], row['date']]),
                maturity = row['Loan_TenureMonths'],
                tenor = row['months_remaining'],
                
                # Loan exposure attributes
                principal = row[self.exposure_column], #max(row['Loan_BalanceOriginal'], row['Loan_BalanceDisbursed']),

                # Loan repayment attributes
                effective_interest_rate = row['Loan_InterestRateEffective'],
                payment_type = row['Repayment Type'],
                payment_frequency = row['payment_frequency'],
                )

            self.principal_exposure += loan.principal

            if row['nPrommiseLoanStatus'] > 2:
                # loan.stage = 3 # Defaulted loans

                el_df = loan.calculate_pd_and_expected_loss(
                    df = loan.expected_loss_df_setup(
                        portfolio_schedule = self.cashflow_schedule,
                        ),
                    marginal_pd_arr = self.select_maturity_split(
                        split_variable = row['Loan_TenureMonths']
                        ),
                    lgd = 1#lgd_arr[0]
                    )
                
                loan.el_df = el_df.iloc[[0]]
                self.default_loss += loan.el_df.loc[0, 'Exposure'] # Sum contributions
                npl_list.append(loan)
            else:

                loan.el_df = loan.calculate_pd_and_expected_loss(
                    df = loan.expected_loss_df_setup(
                        portfolio_schedule = self.cashflow_schedule,
                        ),
                    marginal_pd_arr = self.select_maturity_split(
                        split_variable = row['Loan_TenureMonths']
                        ),
                    sicr_active = sicr_active,
                    lgd = lgd_arr[1]
                )
                
            self.cashflow_schedule = self.total_monthly_cashflow(
                loan = loan
            )            

            loan_list.append(loan)
        return loan_list, npl_list
    

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
            'Payment Date': pd.date_range(
                start = self.date,
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

        # if 'PD*Exposure' not in df.columns:
        #     df['PD*Exposure'] = 0.  # Initialize if column does not exist
        
        # if 'Portfolio Exposure' not in df.columns:
        #     df['Portfolio Exposure'] = 0.

        if 'Principal Exposure' not in df.columns:
            df['Principal Exposure'] = 0.

        if 'Sum ELs' not in df.columns:
            df['Sum ELs'] = 0.

        # # Ensure element-wise addition
        # df.loc[matched_indices, 'PD*Exposure'] += (
        #     loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'PD*Exposure'].values
        # )
        
        # df.loc[matched_indices, 'Portfolio Exposure'] += (
        #     loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'PD*Exposure*LGD'].values
        # )   

        df.loc[matched_indices, 'Principal Exposure'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'Cum Principal Payments'].values
        )  

        df.loc[matched_indices, 'Sum ELs'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'PD*Exposure*LGD'].values
        )  
        
        if 'Portfolio Exposure' in loan.el_df.columns:
            loan.el_df.drop(columns = ['Portfolio Exposure'], inplace = True)

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

        df['Available Guarantee'] = (guarantee - df['Principal Exposure']).clip(lower = 0)

        output_columns = ['Payment Date', 'Portfolio Month', 'Portfolio Month Flag', 
                          'Principal Exposure', #'Portfolio Exposure',
                          'Available Guarantee', 'Sum ELs']
        return df[output_columns]
    

    def ccf_application(
        self,
        ccf_selected: np.ndarray,
        marginal_pd_dict: dict,
        lgd_guarantee: float,
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
        
        early_sid_hit = self.check_threshold(
            relative_threshold = .1
        )

        if (early_sid_hit) | isinstance(params.early_sid_dict[self.portfolio], pd.Timestamp):
            sid_date = self.date
            mask_before_scheduled_sid = pd.Series(
                False, 
                index = df.index
            )

            self.sid = True
        else:
            sid_date = params.scheduled_sid_dict[self.portfolio]

            # Create mask for payments before SID, where CCF will be applied
            mask_before_scheduled_sid = (
                (df['Payment Date'].dt.year < sid_date.year) |
                ((df['Payment Date'].dt.year == sid_date.year) & 
                    (df['Payment Date'].dt.month <= sid_date.month))
            )

        # Initialize Applied CCF column with zeros
        df['Applied CCF'] = 0.

        # Apply CCF values only to payments before SID using .loc
        df.loc[mask_before_scheduled_sid, 'Applied CCF'] = ccf_selected[df['Portfolio Month Flag']][mask_before_scheduled_sid]

        df['Estimated Guarantee'] = df['Available Guarantee'] * df['Applied CCF'] ###

        # Assuming the guarantee behaves as a performing loan established at the current snapshot
        if len(df) - mask_before_scheduled_sid.sum() <= 36:
            marginal_pd_arr = marginal_pd_dict['maturity-lower']

        else:
            marginal_pd_arr = marginal_pd_dict['maturity-upper']

        df['EG*PD'] = df['Estimated Guarantee'] * marginal_pd_arr[1 : len(df) + 1, 0] ###
        df['Guarantee EL'] = df['EG*PD'] * lgd_guarantee

        # Estimated Guarantee Loan Amortized Plan
        df['EGLAP'] = 0.

        mask_after_sid = ~mask_before_scheduled_sid
        remaining_periods = 0

        if mask_after_sid.any() and mask_before_scheduled_sid.any():
            last_estimated_guarantee = df.loc[mask_before_scheduled_sid, 'Estimated Guarantee'].iloc[-1]
            remaining_periods = len(df) - mask_before_scheduled_sid.sum()
            
        if remaining_periods > 0:
            df.loc[mask_after_sid, 'EGLAP'] = last_estimated_guarantee / remaining_periods

        # Term related to actual expsoure on-book + Estimated guarantee used before the s.i.d. through the CCF + Amortized plan of the estimated guarantee loan after s.i.d.
        df['Portfolio Marginal EL'] = df['Sum ELs'] + df['Guarantee EL']

        # Then only add the third component where mask_after_sid is True
        # Ensure the slice length matches the number of True values in mask_after_sid
        num_true_in_mask = mask_after_sid.sum()
        pd_slice = marginal_pd_arr[1 : min(remaining_periods + 1, num_true_in_mask + 1), 0]

        # Only modify the rows where mask_after_sid is True
        if len(pd_slice) > 0:  # Make sure we have values to add
            df.loc[mask_after_sid, 'Portfolio Marginal EL'] += (
                df.loc[mask_after_sid, 'EGLAP'].iloc[:len(pd_slice)] * pd_slice[:len(df.loc[mask_after_sid])] * lgd_guarantee
            )
        
        ordered_columns = [
            'Payment Date', 'Portfolio Month', 'Portfolio Month Flag', 
            'Sum ELs', 'Principal Exposure', 'Available Guarantee', 
            'Applied CCF', 'Estimated Guarantee', 'EG*PD', 'EGLAP', 'Portfolio Marginal EL',
        ]
        return df[ordered_columns]
    
    # def marginal_el(
    #     self,
    #     lgd: float,
    # ) -> None:
        
    #     self.cashflow_schedule['Marginal EL'] = self.cashflow_schedule['Total Exposure'] * lgd


    def tranching_and_lifetime_expected_loss_calculation(
        self,
        tranche_dict: Dict[str, float],
        guarantee_dict: dict
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
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

        tranches_arr = tranche_dict[self.portfolio].copy()
        
        total_lel = self.cashflow_schedule['Portfolio Marginal EL'].sum()
        guarantee_val = guarantee_dict[self.portfolio] # To avoid repeated calls

        if isinstance(guarantee_val, np.ndarray):
            if (self.date.year < 2023) | ( (self.date.year == 2023) & (self.date.month < 4) ):
                guarantee_val = guarantee_val[0]
            else:
                guarantee_val = guarantee_val[1]

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

        out_dictionary = {
            'junior': fi_loss, 
            'mezzanine_ec': ec_loss, 
            'mezzanine_massif': massif_loss, 
            'senior': fmo_loss
        }

        out_dictionary_normalized = {
            'junior': fi_loss / guarantee_val, 
            'mezzanine_ec': ec_loss / guarantee_val, 
            'mezzanine_massif': massif_loss / guarantee_val, 
            'senior': fmo_loss / guarantee_val
        }

        return total_lel, out_dictionary, out_dictionary_normalized


    def check_threshold(
        self,
        relative_threshold: float = .1
    ) -> bool:
        """
        Check that non-performing loans (NPLs) exposure does not exceed the criteria for an early stop inclusion date
        """
        
        if not self.npl_list:
            return False

        tot_def_exposure = 0.

        for npl_loan in self.npl_list:
            tot_def_exposure += npl_loan.principal

        npl_exposure_ratio = tot_def_exposure / self.exposure
        if npl_exposure_ratio >= relative_threshold:
            print('Early stop inclusion date has been hit')
            return True
        
        else:
            return False
        

    def tranche_level_sicr(
        self,
        tranche_dict: Dict[str, float],
        guarantee_dict: Dict[str, Union[float, np.ndarray]],
        mezzanine_threshold: float = 0.17
    ) -> dict:
        """
        """

        tranches_arr = tranche_dict[self.portfolio].copy()
        tranche_loss_dict = {
            'junior': 0.,
            'mezzanine EC': 0.,
            'mezzanine Massif': 0.,
            'senior': 0.
        }

        lifetime_el = self.cashflow_schedule['Portfolio Marginal EL'].sum()
        el_12m = self.cashflow_schedule.loc[:11, 'Portfolio Marginal EL'].sum()
        guarantee_val = guarantee_dict[self.portfolio] # To avoid repeated calls

        if isinstance(guarantee_val, np.ndarray):
            if (self.date.year < 2023) | ( (self.date.year == 2023) & (self.date.month < 4) ):
                guarantee_val = guarantee_val[0]
            else:
                guarantee_val = guarantee_val[1]

        fi_max_loss = tranches_arr[0] * guarantee_val

        if lifetime_el <= fi_max_loss:
            print('LEL does not fully utilise junior tranche -> stage 1')
            print(f'LEL = {lifetime_el:.2f}, \033[1m12 months EL = {el_12m:.2f}\033[0m')

            self.stage = 1 

            self.el = el_12m
            tranche_loss_dict['junior'] = el_12m
            return tranche_loss_dict

        else:
            print('LEL fully utilise junior tranche -> stage 1')
            remaining_loss = lifetime_el - fi_max_loss 
            
            ec_max_loss = tranches_arr[1] * guarantee_val
            massif_max_loss = tranches_arr[2] * guarantee_val
            
            if remaining_loss <= mezzanine_threshold * (ec_max_loss + massif_max_loss):
                print(f'LEL on the mezzanine tranche does not exceed {mezzanine_threshold * 100:.2f}% of the mezzanine tranche ({100 * remaining_loss / (ec_max_loss + massif_max_loss):.2f}%) -> stage 1 and moving to 12 months EL')
                print(f'LEL = {lifetime_el:.2f}, 12 months EL = {el_12m:.2f}')
                
                self.stage = 1
                self.el = el_12m

                remaining_loss_12m = el_12m - fi_max_loss
                if remaining_loss_12m < 0:
                    # Junior is not fully depleted
                    print('12 months EL does not fully utilise junior tranche')
                    tranche_loss_dict['junior'] = el_12m
                    return tranche_loss_dict
                
                else:
                    tranche_loss_dict['junior'] = fi_max_loss

                    tranche_loss_dict['mezzanine EC'] = remaining_loss_12m * 0.9
                    tranche_loss_dict['mezzanine Massif'] = remaining_loss_12m * 0.1
            else:
                if remaining_loss <= (ec_max_loss + massif_max_loss):
                    print(f'LEL exceeds {mezzanine_threshold * 100:.2f}% of the mezzanine tranche -> stage 2')
                    self.stage = 2
                    
                    tranche_loss_dict['mezzanine EC'] = remaining_loss_12m * 0.9
                    tranche_loss_dict['mezzanine Massif'] = remaining_loss_12m * 0.1

                    return tranche_loss_dict
                else:
                    print(f'LEL exceeds mezzanine tranche -> stage 3')
                    self.stage = 3
                    tranche_loss_dict['mezzanine EC'] = ec_max_loss
                    tranche_loss_dict['mezzanine Massif'] = massif_max_loss

                    tranche_loss_dict['senior'] = remaining_loss - (ec_max_loss + massif_max_loss)

                    return tranche_loss_dict

