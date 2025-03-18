from typing import Literal, Union
import pandas as pd
import numpy as np
import numpy_financial as npf


class Loan:
    """ 
    A Loan class. Contains payment information related to a single loan and generates payment schedules for 
    loans bearing the following payment types: 'bullet', 'linear', 'interest_only', 'annuity'
    Keyword Arguments:
    identifier              -- Loan identifier
    portfolio               -- str: Portfolio to which the loan belongs
    currency                -- str: Currency code of the loan
    principal               -- float: Principal amount remaining on the loan
    status                  -- int: Loan's status, 0 (performing), 1 (arrear class 1), 2 (arrear class 2), 3 (default 1), 4 (default - absorbing status)
    arrear_days             -- int: Number of arrear days, used for SICR staging

    tenor                   -- int: Remaining tenor of the loan in months (e.g. tenor 24 suggests the loan will have another 24 months until it is fully repaid)
    maturity:               -- int: Maturity at origination
    payment_frequency       -- int: Frequency with which repayments occur in months (e.g. quarterly repayments would be 3)
    
    start_date              -- pd.Timestamp: Loan's start date
    end_date                -- pd.Timestamp: Loan's end date
    date                    -- pd.Timestamp: Reporting date

    effective_interest_rate -- float: The effective annual interest rate expressed in decimals, e.g. 10% 0.10
    payment_type            -- Literal['bullet', 'linear', 'interest_only', 'annuity']: The type of payment schedule meant for the loan
    """

    def __init__(
        self,

        identifier,
        portfolio: str, 
        currency: str,
        principal: float,
        status: int, 
        arrear_days: int,
        
        tenor: int, 
        maturity: int,
        payment_frequency: int,

        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        date: pd.Timestamp,
        effective_interest_rate: float,
        payment_type: Literal['bullet', 'linear', 'interest_only', 'annuity'],
    ) -> None:
        
        self.identifier = identifier
        self.portfolio = portfolio
        self.currency = currency
        self.principal = principal
        self.status = status
        self.arrear_days = arrear_days

        self.maturity = maturity
        self.tenor = tenor
        self.payment_frequency = payment_frequency

        self.start_date = start_date
        self.end_date = end_date
        self.date = date

        self.effective_interest_rate = 0.05 if effective_interest_rate == 0 else abs(effective_interest_rate) if abs(effective_interest_rate) < 1 else abs(effective_interest_rate) / 100
        self.payment_type = payment_type

        self.original_maturity = self.quality_check_on_def()
        
        payment_dates = self.generate_payment_schedule_dates(
            self.start_date,
            self.end_date,
            self.maturity,
            self.tenor,
            self.payment_frequency
        )

        self.payment_schedule = self.generate_payment_schedule(payment_dates)

        # Add identifying information 
        self.payment_schedule['Currency'] = self.currency
        self.payment_schedule['Portfolio'] = self.portfolio 
        self.payment_schedule['Identifier'] = self.identifier

    def quality_check_on_def(
        self,
    ) -> int:
        """
        Validates and fixes inconsistent loan end dates.
        
        Handles cases where a loan is still on-book past its scheduled end_date by 
        adjusting the end date to the current reporting date and recalculating maturity/tenor.
        
        Returns:
            original_maturity -- int: The original maturity value if changed, -1 if no change was needed
        """
        conditions = (
            ( (self.end_date.year == self.date.year) & (self.end_date.month < self.date.month) ) |
             (self.end_date.year < self.date.year)
             )
        
        if (conditions):    
            self.end_date = self.date
            self.maturity = max((round((self.date - self.start_date).days / 365.25 * 12)), 0)
            self.tenor = max(self.maturity - round((self.date - self.start_date).days / 365.25 * 12), 0)
        
            return self.maturity
        
        return -1
            
    def generate_payment_schedule_dates(
        self, 
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        maturity: int,
        tenor: int,
        payment_frequency: int,
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of payment dates based on loan characteristics.
        
        Creates a schedule of payment dates from loan start to end, with appropriate
        handling based on payment type. The DataFrame includes the origination month
        if the loan exists at the reference date.
        
        Arguments:
        start_date          -- pd.Timestamp: Start date of the loan
        end_date            -- pd.Timestamp: End date of the loan
        maturity            -- int: Total loan maturity in months
        tenor               -- int: Remaining tenor in months
        payment_frequency   -- int: Frequency of payments in months
        
        Returns:
            pd.DataFrame    -- DataFrame containing payment schedule dates with columns:
                               'Payment Date', 'Month Number', 'Payment Number', 'Payment Accumulation'
        """
        if self.original_maturity >= 0:
            # Means that end_date is in the past 
            date_series = pd.Series([end_date])
            return pd.DataFrame({
                    'Payment Date': date_series,
                    'Month Number': pd.Series([self.maturity]),
                    'Payment Number': pd.Series([1,] * date_series.shape[0]).cumsum(),
                    'Payment Accumulation': pd.Series([payment_frequency,] * date_series.shape[0]),
                })

        else:

            if self.payment_type == 'bullet':
                if tenor == 0:
                    date_series = pd.Series([end_date])

                    return pd.DataFrame({
                        'Payment Date': date_series,
                        'Month Number': pd.Series([self.maturity]),
                        'Payment Number': pd.Series([1,] * date_series.shape[0]).cumsum(),
                        'Payment Accumulation': pd.Series([payment_frequency,] * date_series.shape[0]),
                    })
                
                else:
                    date_series = pd.date_range(
                        start = start_date, 
                        periods = round(maturity / payment_frequency) + 1, 
                        freq = f'{payment_frequency}ME', 
                        inclusive = 'both'
                    )
                    
                    return pd.DataFrame({
                        'Payment Date': date_series,# + pd.DateOffset(months = payment_frequency),
                        'Month Number': (pd.Series([1,] * date_series.shape[0]).cumsum() - 1) * payment_frequency,
                        'Payment Number': pd.Series([1,] * date_series.shape[0]).cumsum() - 1,
                        'Payment Accumulation': pd.Series([payment_frequency,] * date_series.shape[0]), # This might be removed if utilisation of 'Payment Accumulation' is substituted by self.payment_frequency
                    })
            
            else:
                # if tenor / payment_frequency < 1: 
                #      # If tenor / payment frequency is less than 1 then the next payment is the final one. 
                #      date_series = pd.Series([end_date])

                #      lifetime_df = pd.DataFrame({
                #         'Payment Date': date_series,
                #         'Month Number': (pd.Series([1,] * date_series.shape[0]).cumsum()) * payment_frequency,
                #         'Payment Number': pd.Series([1,] * date_series.shape[0]).cumsum(),
                #         'Payment Accumulation': pd.Series([payment_frequency,] * date_series.shape[0]),
                #     })
                # else:
                date_series = pd.date_range(
                    start = start_date, 
                    periods = round(maturity/ payment_frequency) + 1,
                    freq = f'{payment_frequency}ME', 
                    inclusive = 'both',
                )
                    
                lifetime_df = pd.DataFrame({
                    'Payment Date': date_series,
                    'Month Number': (pd.Series([1,] * date_series.shape[0]).cumsum() - 1) * payment_frequency,
                    'Payment Number': pd.Series([1,] * date_series.shape[0]).cumsum() - 1,
                    'Payment Accumulation': pd.Series([payment_frequency,] * date_series.shape[0]),
                })
            
                # Payment dates are calculated from loan's origination to maintain alignment with origination date (for payment_frequency > 1).
                # Past payment dates are now dropped, and "Payment Number" columns is reset.
                mask_present_future_dates = (
                    (lifetime_df['Payment Date'].dt.year > self.date.year) |
                    ( (lifetime_df['Payment Date'].dt.year == self.date.year) & (lifetime_df['Payment Date'].dt.month >= self.date.month) )
                    )
                
                out_df = lifetime_df[mask_present_future_dates].copy()

                if len(out_df) > 0:
                    out_df['Payment Number'] = np.arange(len(out_df))
                    return out_df
            
                else:
                    return pd.DataFrame([{
                        'Payment Date': self.end_date,
                        'Month Number': int(self.maturity),
                        'Payment Number': 1,
                        'Payment Accumulation': self.payment_frequency,
                    }])
    
    def generate_payment_schedule(
        self, 
        payment_date_schedule: pd.DataFrame, 
    ) -> pd.DataFrame:
        """
        Calculates full payment schedule with principal and interest amounts.
        
        Based on the payment dates schedule and loan type (bullet, linear, interest_only, annuity),
        calculates the principal and interest payments for each payment date.
        
        Arguments:
        payment_date_schedule -- pd.DataFrame: DataFrame of payment dates generated by generate_payment_schedule_dates
        
        Returns:
            pd.DataFrame      -- Complete payment schedule with columns including:
                                 'Principal Current', 'Interest Rate Applied', 'Principal Payment',
                                 'Principal Outstanding', 'Interest Payment', 'Total Payment'
        """

        df = payment_date_schedule
        df['Principal Current'] = self.principal

        if self.payment_type == 'bullet':
            if self.original_maturity < 0:
                df['Interest Rate Applied'] = self.effective_interest_rate#((self.effective_interest_rate + 1)**(self.maturity / 12)) - 1
            else:
                df['Interest Rate Applied'] = ((self.effective_interest_rate + 1)**(self.original_maturity / 12)) - 1
        else:
            df['Interest Rate Applied'] = ((self.effective_interest_rate + 1)**(df['Payment Accumulation'] / 12)) - 1

        # The following section takes care of the logic for bullet, IO, and linear payments and those payments which have only one payment left
        if (self.payment_type in ['bullet', 'interest_only', 'linear']) or (len(df) == 1): 

            if len(df) == 1:
                df['Principal Payment'] = self.principal
            elif self.payment_type in ['bullet', 'interest_only',]:

                df['Principal Payment'] = [0,] * (len(df) - 1) + [self.principal] 
            
            elif (self.payment_type == 'linear'):
                if df['Month Number'].iloc[0] == 0:
                    df['Principal Payment'] = [0, ] + [round(self.principal / (len(df) - 1), 2)] * (len(df) - 1)
                else:
                    df['Principal Payment'] = round(self.principal / max(len(df), 1), 2)
            
            # elif len(df) == 1: 
            #     # One payment means we are in the final payment. Here the annuity payment will be the same as the following calculation since the remaining interest and principal will be repaid
            #     df['Principal Payment'] = self.principal

            # Once the payment goes through on the payment date the final principal will be the outstanding amount
            df['Principal Outstanding'] = (df['Principal Current'] - df['Principal Payment'].cumsum()).clip(lower = 0).round(2)

            # Interest was accumulated on the principal before the payment so we take a step back to calculate the interest rate
            # we need to do it like this because of the cumulative sum applied to principal current before
            if self.payment_type == 'bullet':
                df['Interest Payment'] = [0.,] * (len(df) - 1) + [ self.principal * self.effective_interest_rate ]

            else:
                df['Interest Payment'] = ((df['Principal Outstanding'] + df['Principal Payment']) * df['Interest Rate Applied']).round(2)
                if df['Month Number'].iloc[0] == 0:
                    df.loc[0, 'Interest Payment'] = 0
                    

            ## JP:  Modified to Total Payment = Principal Payment + Interest Payment instead of
            #             Total Payment = Principal Outstanding + Interest Payment
            # df['Total Payment'] = df['Principal Outstanding'] + df['Interest Payment']
            df['Total Payment'] = df['Principal Payment'] + df['Interest Payment']
        
        elif self.payment_type == 'annuity': 
            """
            For annuity loans we use the numpy_financial library to calculate the payments which are necessary

            There are cases for which the first row corresponds to the inclusion of the loan. For such months, no payment is expected and the 
            amortment plan should start at the following month.
            An inelegant but necessary adjustment.
            """
            if df['Month Number'].iloc[0] == 0:
                df['Total Payment'] = 0.

                df.loc[1:, 'Total Payment'] =-npf.pmt(
                    ((self.effective_interest_rate + 1)**(self.payment_frequency / 12)) - 1, 
                    len(df) - 1, 
                    self.principal
                    ).round(2)
                
                df['Principal Payment'] = 0.
                principal_payments = -npf.ppmt(
                    df.loc[1:, 'Interest Rate Applied'], 
                    df.loc[1:, 'Payment Number'], 
                    len(df) - 1, 
                    df.loc[1:, 'Principal Current']
                    ).round(2)
                
                df.loc[1:, 'Principal Payment'] = principal_payments

            else:
                df['Payment Number'] = np.arange(len(df)) + 1

                df['Total Payment'] = -npf.pmt(
                    ((self.effective_interest_rate + 1)**(self.payment_frequency / 12)) - 1, 
                    len(df) - 1, 
                    self.principal
                    ).round(2)
                
                df['Principal Payment'] = -npf.ppmt(
                    df['Interest Rate Applied'], 
                    df['Payment Number'], 
                    len(df), 
                    df['Principal Current']
                    ).round(2)
                
            df['Interest Payment'] = (df['Total Payment'] - df['Principal Payment']).round(2)
            df['Principal Outstanding'] = (df['Principal Current'] - df['Principal Payment'].cumsum()).clip(lower = 0).round(2)

        return df
    
    
    def expected_loss_df_setup(
        self,
        portfolio_schedule: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepares a simplified DataFrame for expected loss calculations.
        
        Loan-level payment schedule and portfolio-level payment schedule are misaligned when loans are given in the middle of the month. This function
        merged the two information based on the year and month information, this is possible since reports are always generated at the end of the month.
        All payments can therefore be assumed to have been made.
        
        Arguments:
        portfolio_schedule    -- pd.DataFrame: Portfolio-level payment date schedule
        
        Returns:
            pd.DataFrame      -- Simplified DataFrame with aligned payment dates and exposure calculations
        """
        portfolio_schedule_copy = portfolio_schedule.copy()

        # Portfolio schedule transformation in YYYY-MM
        portfolio_schedule_copy['Short Payment Date'] = pd.PeriodIndex(
            portfolio_schedule_copy['Payment Date'], 
            freq = 'M'
        )

        # Loan schedule transformation in YYYY-MM
        self.payment_schedule['Short Payment Date'] = pd.PeriodIndex(
            self.payment_schedule['Payment Date'], 
            freq = 'M'
        )

        payment_schedule_copy = self.payment_schedule.copy()

        ######################################
        cum_principal_df = portfolio_schedule_copy.merge(
            payment_schedule_copy[['Short Payment Date', 'Total Payment', 'Month Number', 'Principal Payment']],
            on = 'Short Payment Date',
            how = 'outer'
        )

        cum_principal_df['Total Payment'] = cum_principal_df['Total Payment'].fillna(0.)
        cum_principal_df['Exposure'] = cum_principal_df['Total Payment'][::-1].cumsum()
        cum_principal_df['Principal Payment'] = cum_principal_df['Principal Payment'].fillna(0.)
        cum_principal_df['Cum Principal Payments'] = cum_principal_df['Principal Payment'][::-1].cumsum()

        cum_principal_df.dropna(subset = ['Payment Date'])

        # Merge the two dataframes based on "Short Payment Date", add to result_df the needed columns from the loan payment schedule
        result_df = portfolio_schedule_copy.merge(
            self.payment_schedule[['Short Payment Date', 'Total Payment', 'Month Number', 'Principal Payment']],
            on = 'Short Payment Date',
            how = 'left'
        )
        
        final_df = result_df.merge(
           cum_principal_df[['Short Payment Date', 'Cum Principal Payments', 'Exposure']],
           on = 'Short Payment Date',
           how = 'inner'
        )
    
        # Fill NaNs as 0. NaNs are generated where payment are not being made
        final_df['Total Payment'] = final_df['Total Payment'].fillna(0.)
        # final_df['Exposure'] = final_df['Total Payment'][::-1].cumsum()
        final_df['Principal Payment'] = final_df['Principal Payment'].fillna(0)

        final_df.drop(columns = ['Short Payment Date'], inplace = True)
        
        self.payment_schedule.drop(columns = ['Short Payment Date'], inplace = True)
    
        # Drop Rows after maturity is hit
        try:
            max_month_id = final_df['Month Number'].idxmax()
            max_value = final_df.loc[max_month_id, 'Month Number']

            final_df = final_df.loc[:max_month_id].reset_index(drop = True)
            final_df.loc[::-1, 'Month Number'] = np.arange(max_value, max_value - len(final_df), -1)

        except Exception as e:
            print(self.identifier)
            print(self.payment_schedule.head())
            # system = platform.system()

        return final_df
    

    def calculate_pd_and_expected_loss(
        self,
        df: pd.DataFrame,
        marginal_pd_arr: np.ndarray,
        lgd: float,
        interest_bool: bool,
    ) -> pd.DataFrame:
        """
        Calculates probability of default and expected loss for each payment period.
        
        Assigns appropriate marginal PD values based on loan status and payment month,
        then calculates the expected loss (PD*Exposure) for each payment period.
        
        Arguments:
        df              -- pd.DataFrame: DataFrame from expected_loss_df_setup to append PD calculations
        marginal_pd_arr -- np.ndarray: Array of marginal PD values indexed by [month, status]
        
        Returns:
            pd.DataFrame -- DataFrame with added 'Marginal PD' and 'PD*Exposure' columns
        """
        # Assign marginal PD values
        df['Marginal PD'] = marginal_pd_arr[
            df['Month Number'].to_numpy(dtype = np.int32), self.status
            ]
        
        # Calculate PD exposure directly
        if interest_bool:
            df['PD*Exposure'] = df['Exposure'] * df['Marginal PD']
            df['PD*Exposure*LGD'] = df['PD*Exposure'] * lgd

        else:
            df['PD*Exposure'] = df['Cum Principal Payments'] * df['Marginal PD']
            df['PD*Exposure*LGD'] = df['PD*Exposure'] * lgd

        # Order columns for better visualization
        output_columns = ['Payment Date', 'Month Number', 'Cum Principal Payments', 'Total Payment', 
                          'Exposure', 'Marginal PD',
                          'PD*Exposure', 'PD*Exposure*LGD']
        return df[output_columns]
    

    def print_loan_info(
        self,
        output: bool = False
    ) -> Union[None, dict]:
        """
        Displays key information about the loan.
        
        Prints loan attributes including identifier, dates, payment characteristics,
        and principal amount to assist with debugging and inspection.
        """
        
        if not output:
            print(f'Loan_ID: {self.identifier}')
            print(f'Start date: {self.start_date}')
            print(f'End date: {self.end_date}')
            print(f'Reporting date: {self.date}')
            print(f'Payment type: {self.payment_type}')
            print(f'Payment frequency: {self.payment_frequency}')
            print(f'Maturity: {self.maturity}')
            print(f'Tenor: {self.tenor}')
            print(f'Status: {self.status}')
            print(f'Arrear days: {self.arrear_days}')
            # print(f'Stage: {self.stage}')
            print(f'Principal: {self.principal}')
        else:
            return {'Loan_ID': self.identifier,
                    'Start_Date': self.start_date,
                    'End_Date': self.end_date,
                    'Reporting_Date': self.date,
                    'Payment_Type': self.payment_type,
                    'Payment_Frequency': self.payment_frequency,
                    'Maturity': self.maturity,
                    'Tenor': self.tenor,
                    'Status': self.status,
                    'Arrear days': self.arrear_days,
                    # 'Stage': self.stage,
                    'Principal': self.principal
                    }
