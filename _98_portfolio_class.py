from typing import Tuple, List, Any, Dict

import numpy as np
import pandas as pd

from ._99_classes import Loan
"""
A Portfolio class. Contains all information needed to fully characterize the portfolio, and calculate the 
Expected Credit Loss (ECL).

Arguments:
age                   -- int: Age of the portfolio. Used for application of the Credit Conversion Factor (CCF)
country               -- str: Country name
covered_interest_bool -- bool: True if the contract foresees covering interests
cover_stop_date       -- pd.Timestamp: Final date until which the cover/guarantee is valid
currency              -- str: Currency used for the transactions
date                  -- pd.Timestamp: Reporting date of the portfolio snapshot
exposure              -- float: Current on-book exposure
exposure_column       -- str: Column used to gather current balance information
geography             -- str: Geographical group to which the portfolio belongs, needed to identify the PD curve to be used
guarantee             -- float: Guarantee sum
lgd_application_dict  -- dict: Dictionary containing weighted LGD values
portfolio             -- str: Portfolio name
ptf_marginal_pd_dict  -- dict: Dictionary containing the marginal PD curves to be applied
reference_snapshot    -- pd.DataFrame: DataFrame containing loan-level data used as reference
scheduled_sid         -- pd.Timestamp: Date of the scheduled implementation
sid                   -- bool: Flag indicating whether Scheduled Implementation Date has been reached
"""

class Portfolio:
    def __init__(
        self,
        age: int,
        country: str,
        covered_interest_bool: bool,
        cover_stop_date: pd.Timestamp,
        currency: str,
        date: pd.Timestamp,
        exposure: float,
        exposure_column: str,
        geography: str,
        guarantee: float,
        lgd_application_dict: dict,
        portfolio: str,
        ptf_marginal_pd_dict: dict,
        reference_snapshot: pd.DataFrame,
        scheduled_sid: pd.Timestamp,
        sid: bool,
    ) -> None:


        self.age = age
        self.country = country
        self.covered_interest_bool = covered_interest_bool
        self.cover_stop_date = cover_stop_date
        self.currency = currency
        self.date = date
        self.exposure = exposure
        self.exposure_column = exposure_column
        self.geography = geography
        self.guarantee = guarantee
        self.lgd_application_dict = lgd_application_dict
        self.portfolio = portfolio
        self.ptf_marginal_pd_dict = ptf_marginal_pd_dict
        self.reference_snapshot = reference_snapshot
        self.scheduled_sid = scheduled_sid
        self.sid = sid
        
        # Initialise empty/zero attributes
        self.principal_exposure = 0.
        self.default_loss = 0.
        self.lel = 0.
        self.tranche_loss_dict = {} 

        # Initialise monthly payments
        self.cashflow_schedule = self.future_payments()
        
        # Creation of list of Loans
        self.loan_list, self.npl_list = self.underlying_loan_list(
            df = reference_snapshot,
            lgd_applied = lgd_application_dict
        )

        # Carry on Principal exposure of defaulted loans
        self.cashflow_schedule.loc[1:, 'Principal Exposure'] = self.cashflow_schedule.loc[1:, 'Principal Exposure'] + self.default_loss
    

    def select_maturity_split(
        self,
        split_variable: int
    ) -> np.ndarray:
        """
        Selects the appropriate maturity curve based on loan term length.
        
        Determines which probability of default (PD) curve to use based on whether
        the loan maturity is shorter or longer than 36 months.
        
        Arguments:
        split_variable -- int: The loan tenure in months
            
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
        lgd_applied: dict,
    ) -> Tuple[List[Loan], List[Loan]]:
        """
        Creates Loan objects for each loan in the reference snapshot and calculates expected losses.
        
        For each loan in the input DataFrame:
        1. Creates a Loan object with attributes from the snapshot
        2. Sets up the expected loss DataFrame 
        3. Calculates PD and expected loss for each loan
        4. Updates the portfolio's cashflow schedule with the loan's cashflows
        5. Identifies non-performing loans (NPLs)
        
        Arguments:
        df          -- pd.DataFrame: Reference snapshot containing loan-level data
        lgd_applied -- dict: LGD values applied based on maturity, i.e. short/long

        Returns:
            Tuple[List[Loan], List[Loan]]: A tuple containing:
                - List of Loan objects
                - List of defaulted Loan objects
        """
        npl_list = []
        loan_list = []

        # Loops through each row in the DataFrame, and creates a Loan object based on the row information
        for _, row in df.iterrows():
            loan = Loan(
                arrear_days = row['Loan_Arrear_days'],
                currency = self.currency,
                date = self.date,
                effective_interest_rate = row['Loan_InterestRateEffective'],
                end_date = row['EndDate'], 
                identifier = row['Loan_ID'],
                maturity = row['Loan_TenureMonths'],
                payment_frequency = row['payment_frequency'],
                payment_type = row['Repayment Type'],
                portfolio = self.portfolio,
                principal = row[self.exposure_column],
                start_date = row['StartDate'],
                status = row['nPrommiseLoanStatus'],
                tenor = row['months_remaining']
            )

            # Used for check on total exposure, this should match (be very close to) reference_snapshot['BalanceCurrent'].sum()
            self.principal_exposure += loan.principal

            # Calculate the full repayment schedule for each loan
            if row['nPrommiseLoanStatus'] > 2:
                # No recovery assumed for already defaulted loans
                el_df = loan.calculate_pd_and_expected_loss(
                    df = loan.expected_loss_df_setup(
                        portfolio_schedule = self.cashflow_schedule,
                    ),
                    marginal_pd_arr = self.select_maturity_split(
                        split_variable = row['Loan_TenureMonths']
                    ),
                    lgd = 1,
                    interest_bool = self.covered_interest_bool
                )
                
                loan.el_df = el_df.iloc[[0]]
                self.default_loss += loan.el_df.loc[0, 'Cum Principal Payments']
                npl_list.append(loan)

            else:                
                # Select LGD to be used based on maturity
                lgd_applied_value = lgd_applied['maturity-lower'] if loan.maturity <= 36 else lgd_applied['maturity-upper']
                
                # Calculate loan's full amortization schedule, and EL contribution
                loan.el_df = loan.calculate_pd_and_expected_loss(
                    df = loan.expected_loss_df_setup(
                        portfolio_schedule = self.cashflow_schedule,
                    ),
                    marginal_pd_arr = self.select_maturity_split(
                        split_variable = row['Loan_TenureMonths']
                    ),
                    lgd = lgd_applied_value,
                    interest_bool = self.covered_interest_bool
                )
                
            # After calculating the repayment plan, sum each individual contribution in cashflow_schedule
            self.cashflow_schedule = self.total_monthly_cashflow(
                loan = loan
            )            
            loan_list.append(loan)
        
        return loan_list, npl_list

    def future_payments(
        self
    ) -> pd.DataFrame:
        """
        Initialise a DataFrame from reporting date to cover stop date. 
        
        Generates a DataFrame with monthly payment dates starting from the portfolio reporting date up to the cover stop date. 
        The cover stop date is explicitly added as a final row, even if it doesn't fall on the last day of a month.
        
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
        
        Arguments:
            loan -- Loan: The loan object whose cashflows will be added to the portfolio
            
        Returns:
            pd.DataFrame: Updated portfolio cashflow schedule with the loan's contribution
        """

        df = self.cashflow_schedule

        # Identify the common 'Payment Date' values
        common_dates = df['Payment Date'].isin(loan.el_df['Payment Date'])
        matched_indices = df.index[common_dates]

        ##### Initialise relevant columns
        if 'Principal Exposure' not in df.columns:
            df['Principal Exposure'] = 0.

        if 'Sum ELs' not in df.columns:
            df['Sum ELs'] = 0.

        if 'Sum PD*Exposure' not in df.columns:
            df['Sum PD*Exposure'] = 0.

        ##### Sum loan's contributions
        df.loc[matched_indices, 'Principal Exposure'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'Cum Principal Payments'].values
        )  

        df.loc[matched_indices, 'Sum ELs'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'PD*Exposure*LGD'].values
        )  

        df.loc[matched_indices, 'Sum PD*Exposure'] += (
            loan.el_df.set_index('Payment Date').loc[df.loc[matched_indices, 'Payment Date'], 'PD*Exposure'].values
        )
        
        if 'Portfolio Exposure' in loan.el_df.columns:
            loan.el_df.drop(columns = ['Portfolio Exposure'], inplace = True)

        return df
    
    def ccf_application(
        self,
        ccf_row: np.ndarray,
        lgd: float,
        params: Any,
        sid_date: None | pd.Timestamp
    ) -> pd.DataFrame:
        """
        This function applies the CCF, and calculates the estimated guarantee used. CCF is applied also according to the SID (as 
        described in the arguments section). Portfolio-level marginal expected loss (EL) is then calculated.

        Arguments:
        ccf_row  -- np.ndarray: Row containing CCF values (based on portfolio's age) 
        lgd      -- float: LGD value applied
        params   -- Parameter's dataclass 
        sid_date -- None | pd.Timestamp: A None value is provided when running this function the first time, so that the function applies 
                                         the scheduled SID. If a pd.Timestamp is provided, the timestamp is applied as an early SID
    
        Returns:
            pd.DataFrame: updated portfolio-level view
        """

        # As tuned in calibration phase, monthly schedule is assumed
        new_perf_window_list = np.arange(len(ccf_row))

        df = self.cashflow_schedule        
        df.loc[:, 'Portfolio Month'] = np.arange(len(df)) + 1
        
        # Define indices for applying the CCF, after month 
        df['Portfolio Month Flag'] = df['Portfolio Month'].apply(
            lambda x: min(
                sum([(x - 1) > threshold for threshold in new_perf_window_list]), len(new_perf_window_list) - 1
            )
        )

        # Payments' dates are at the end of the month; checks on stop inclusion date (SID) are performed on years and months
        if isinstance(sid_date, pd.Timestamp):
            # Entered in the early SID loop
            mask_before_scheduled_sid = (
                (df['Payment Date'].dt.year < sid_date.year) |
                ((df['Payment Date'].dt.year == sid_date.year) & 
                    (df['Payment Date'].dt.month <= sid_date.month))
            )
        else:
            # Entered during the first CCF application run when the given sid_date is None
            if isinstance(params.early_sid_dict[self.portfolio], pd.Timestamp):
                sid_date = self.date
                mask_before_scheduled_sid = pd.Series(
                    False, 
                    index = df.index
                )
                self.sid = True
            else:
                sid_date = params.scheduled_sid_dict[self.portfolio]

            # Create mask for payments before SID, where CCF is applied
            mask_before_scheduled_sid = (
                (df['Payment Date'].dt.year < sid_date.year) |
                ((df['Payment Date'].dt.year == sid_date.year) & 
                    (df['Payment Date'].dt.month <= sid_date.month))
            )

        # Initialisation of CCF columns
        df['Applied CCF'] = 0.
        df['Multiplying Factor'] = 0.

        # Apply CCF values only to payments before SID using .loc
        df.loc[mask_before_scheduled_sid, 'Applied CCF'] = ccf_row[df['Portfolio Month Flag']][mask_before_scheduled_sid]
        df.loc[mask_before_scheduled_sid, 'Multiplying Factor'] = 1 - df.loc[mask_before_scheduled_sid, 'Applied CCF']

        # Ararat had a top-up in guarantee, here it is assumed checks for snapshots before 202304 (excluded) are not performed
        if self.portfolio == 'Ararat':
            guarantee = 20_000_000
        else:
            guarantee = params.guarantee_dict[self.portfolio]
        
        if len(df) - mask_before_scheduled_sid.sum() <= 36:
            marginal_pd_arr = self.ptf_marginal_pd_dict['maturity-lower']
        else:
            marginal_pd_arr = self.ptf_marginal_pd_dict['maturity-upper']

        # Initialisation of exposure columns
        df['Total Exposure'] = 0.
        df['Guarantee Available Next'] = 0.
        df['EGU'] = 0.
        df['EGLAP'] = 0.

        """
        The first row of the dataframe (current snapshot report) cannot use guarantee as reports are generated at the end of the month. Therefore,
        an early sid activation has effect only if aplied from row 2
        """
        if mask_before_scheduled_sid.sum() > 1:
            df.loc[0, 'Total Exposure'] = df.loc[0, 'Principal Exposure']
            df.loc[0, 'Guarantee Available Next'] = guarantee - df.loc[0, 'Total Exposure']
            
            for i in range(1, mask_before_scheduled_sid.sum()):
                df.loc[i, 'Total Exposure'] = df.loc[i - 1, 'Total Exposure'] + df.loc[i - 1, 'Guarantee Available Next'] * df.loc[i, 'Multiplying Factor']
                df.loc[i, 'Guarantee Available Next'] = guarantee - df.loc[i, 'Total Exposure']
                df.loc[i, 'EGU'] = df.loc[i, 'Total Exposure'] - df.loc[i, 'Principal Exposure']

            mask_after_sid = ~mask_before_scheduled_sid
            remaining_periods = 0

            if mask_after_sid.any() and mask_before_scheduled_sid.any():
                last_estimated_guarantee = df.loc[mask_before_scheduled_sid, 'EGU'].iloc[-1]
                remaining_periods = len(df) - mask_before_scheduled_sid.sum()
                
            if remaining_periods > 0:
                df.loc[mask_after_sid, 'EGLAP'] = last_estimated_guarantee / remaining_periods
            
            df.loc[mask_after_sid, 'Total Exposure'] += df.loc[mask_after_sid, 'EGLAP'][::-1].cumsum() + df.loc[mask_after_sid, 'Principal Exposure']

        else:
            # Setting total exposure to principal exposure
            df.loc[:, 'Total Exposure'] = df.loc[:, 'Principal Exposure']

            mask_after_sid = ~mask_before_scheduled_sid
            remaining_periods = 0
            last_estimated_guarantee = 0
            remaining_periods = len(df) - mask_before_scheduled_sid.sum()

        # Expected guarantee loss calculations
        df['EG*PD'] = df['EGU'] * marginal_pd_arr[1 : len(df) + 1, 0] ###
        df['Guarantee EL'] = df['EG*PD'] * lgd

        # Term related to actual expsoure on-book + Estimated guarantee used before the s.i.d. through the CCF + Amortized plan of the estimated guarantee loan after s.i.d.
        df['Portfolio Marginal EL'] = df['Sum ELs'] + df['Guarantee EL']

        # Then only add the third component where mask_after_sid is True
        num_true_in_mask = mask_after_sid.sum()
        pd_slice = marginal_pd_arr[1 : min(remaining_periods + 1, num_true_in_mask + 1), 0]

        # Only modify the rows where mask_after_sid is True
        if len(pd_slice) > 0:  # Make sure we have values to add
            df.loc[mask_after_sid, 'Portfolio Marginal EL'] += (
                df.loc[mask_after_sid, 'EGLAP'].iloc[:len(pd_slice)] * pd_slice[:len(df.loc[mask_after_sid])] * lgd
        )

        return df

    def tranche_level_sicr(
        self,
        tranche_dict: Dict[str, float],
        guarantee_dict: Dict[str, float | np.ndarray],
        mezzanine_threshold: float = 0.17
    ) -> Dict[str, float]:
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
        
        Arguments:
        tranche_dict   -- Dict[str, np.ndarray]: Dictionary containing portfolio tranching percentages as an array. The value associated
                                                 with self.portfolio should be an array structured as 
                                                 follows:[junior, mezzanine_ec, mezzanine_massif, senior].
        guarantee_dict -- dict: Dictionary containing guarantee values for each portfolio, with portfolio names as keys and guarantee amounts as values.
        
        Returns:
            Dict[str, float]:
                {'junior': float, 'mezzanine_ec': float, 'mezzanine_massif': float, 'senior': float}
                - fmo_loss_bool: Boolean flag indicating whether the FMO (senior tranche) 
                experiences any loss.
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
        self.lel = lifetime_el
        self.el12 = el_12m
        guarantee_val = guarantee_dict[self.portfolio] # To avoid repeated calls
        unlocked_guarantee = np.max(self.cashflow_schedule['Total Exposure'])
        
        # Check for Ararat
        if isinstance(guarantee_val, np.ndarray):
            if (self.date.year < 2023) | ( (self.date.year == 2023) & (self.date.month < 4) ):
                guarantee_val = guarantee_val[0]
            else:
                guarantee_val = guarantee_val[1]

        # fi_max_loss = tranches_arr[0] * guarantee_val 
        fi_max_loss = tranches_arr[0] * unlocked_guarantee

        if lifetime_el <= fi_max_loss:
            print('LEL does not fully utilise junior tranche -> stage 1')
            print(f'LEL = {lifetime_el:.2f}, 12 months EL = {el_12m:.2f}')

            self.stage = 1 

            tranche_loss_dict['junior'] = el_12m
            return tranche_loss_dict

        else:
            print('Junior tranche fully utilised')
            remaining_loss = lifetime_el - fi_max_loss 
            
            ec_max_loss = tranches_arr[1] * guarantee_val
            massif_max_loss = tranches_arr[2] * guarantee_val
            
            if remaining_loss <= mezzanine_threshold * (ec_max_loss + massif_max_loss):
                print(f'LEL on the mezzanine tranche does not exceed {mezzanine_threshold * 100:.2f}% of the mezzanine tranche ({100 * remaining_loss / (ec_max_loss + massif_max_loss):.2f}%) -> stage 1 and moving to 12 months EL')
                print(f'LEL = {lifetime_el:.2f}, 12 months EL = {el_12m:.2f}')
                
                self.stage = 1

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

                    return tranche_loss_dict
            else:
                if remaining_loss <= (ec_max_loss + massif_max_loss):
                    print(f'LEL exceeds {mezzanine_threshold * 100:.2f}% of the mezzanine tranche ({100 * remaining_loss / (ec_max_loss + massif_max_loss):.2f}%), senior tranche is not affected -> stage 2')
                    self.stage = 2
                    
                    tranche_loss_dict['junior'] = fi_max_loss
                    tranche_loss_dict['mezzanine EC'] = remaining_loss * 0.9
                    tranche_loss_dict['mezzanine Massif'] = remaining_loss * 0.1

                    return tranche_loss_dict
                else:
                    self.stage = 3
                    fmo_loss = remaining_loss - (ec_max_loss + massif_max_loss)

                    print(f'LEL exceeds mezzanine tranche -> stage 3')

                    tranche_loss_dict['junior'] = fi_max_loss
                    tranche_loss_dict['mezzanine EC'] = ec_max_loss
                    tranche_loss_dict['mezzanine Massif'] = massif_max_loss

                    tranche_loss_dict['senior'] = fmo_loss

                    return tranche_loss_dict

    def post_check(
        self,
    ) -> None:
        """
        Creates needed columns for investigation purposes. Directly updates cashflow_schedule.
        """
        df = self.cashflow_schedule
        df['Cumulative Portfolio PD*Exposure'] = (df['Sum PD*Exposure'] + df['EG*PD']).cumsum()
        df['Percentage Cum PD'] = df['Cumulative Portfolio PD*Exposure'] / df['Total Exposure']

    def sid_check(
        self,
        ccf_row: np.ndarray,
        lgd: float,
        params_application: Any,
        threshold: float = 0.1
    ) -> int:
        """
        Evaluation of the sid, and re-run of the ccf_application() function.
        
        Arguments:
        ccf_row            -- np.ndarray: CCF's row used
        lgd                -- float: LGD value used
        params_application -- Any: Application's parameter dataclass
        threshold          -- float: Threshold used to evaluate early SID activation

        Returns:
            int --
        
        """
        df = self.cashflow_schedule

        df['Cumulative Portfolio PD*Exposure'] = (df['Sum PD*Exposure'] + df['EG*PD']).cumsum()
        df['Percentage Cum PD'] = df['Cumulative Portfolio PD*Exposure'] / df['Total Exposure']

        new_sid = df.loc[(df['Percentage Cum PD'] > threshold).idxmax(), 'Payment Date']
        scheduled_sid = params_application.scheduled_sid_dict[self.portfolio]
        early_sid_condition = (new_sid.year < scheduled_sid.year) | ( 
            (new_sid.year == scheduled_sid.year) & (new_sid.month < scheduled_sid.month)
        )

        if ( 
            (not isinstance(params_application.early_sid_dict[self.portfolio], pd.Timestamp)) and (early_sid_condition) 
            ):
            print(f'Early SID activated, new SID: {new_sid}, instead of {scheduled_sid}')
            df = self.ccf_application(
                ccf_row = ccf_row,
                sid_date = new_sid,
                params = params_application,
                lgd = lgd
            )

        self.post_check()

        return (df['Percentage Cum PD'] > threshold).idxmax()
