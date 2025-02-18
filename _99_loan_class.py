from typing import Literal
import copy

import pandas as pd
import numpy as np
import numpy_financial as npf


class Loan:
    """ 
    A Loan class. Contains payment information related to a single loan and generates payment schedules for 
    loans bearing the following payment types: 'bullet', 'linear', 'interest_only', 'annuity'
    Keyword Arguments:
    identifier              -- Loan identifier
    portfolio               -- str: portfolio to which the loan belongs
    currency                -- str: Currency code of the loan
    principal               -- float: Principal amount remaining on the loan
    tenor                   -- int: Remaining tenor of the loan in months (e.g. tenor 24 suggests the loan will have another 24 months until it is fully repaid)
    payment_frequency       -- int: Frequency with which repayments occur in months (e.g. quarterly repayments would be 3)
    start_date              -- pd._libs.tslibs.timestamps.Timestamp: Start date of the payment schedule
    effective_interest_rate -- float: The effective annual interest rate expressed in decimals (e.g. 10% 0.10)
    payment_type            -- Literal['bullet', 'linear', 'interest_only', 'annuity']: The type of payment schedule meant for the loan
    
    """
    def __init__(
        self,

        identifier,
        portfolio: str, 
        currency: str,
        principal: float,
        status: int, 
        
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

        self.maturity = maturity
        self.tenor = tenor

        self.start_date = start_date
        self.end_date = end_date
        
        self.date = date
        self.effective_interest_rate = 0.001 if effective_interest_rate == 0 else abs(effective_interest_rate) if abs(effective_interest_rate) < 1 else abs(effective_interest_rate) / 100
        
        self.payment_type = payment_type
        self.payment_frequency = payment_frequency

        # self.payment_schedule = self.generate_payment_schedule_dates(
        #     start_date,
        #     end_date,
        #     maturity,
        #     tenor,
        #     payment_frequency
        # )

        self.payment_schedule = self.generate_payment_schedule(
            self.generate_payment_schedule_dates(
                start_date,
                end_date,
                maturity,
                tenor,
                payment_frequency
        )
            )

        # Save the current column list before we add identifying information
        # output_columns = list(self.payment_schedule.columns)

        # Add identifying information 
        self.payment_schedule['Currency'] = self.currency
        self.payment_schedule['Portfolio'] = self.portfolio 
        self.payment_schedule['Identifier'] = self.identifier
        # self.payment_schedule = self.payment_schedule[['Identifier', 'Portfolio', 'Currency'] + output_columns]



    def generate_payment_schedule_dates(
        self, 
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        maturity: int,
        tenor: int,
        payment_frequency: int,
    ) -> pd.DataFrame:
        """
        The DataFrame returned by this function correctly includes the origination month, should the loan be included at the reference date.
        """
        # print(self.identifier)

        if self.payment_type == 'bullet':
            date_series = pd.date_range(start = start_date, 
                                        periods = round(self.maturity / payment_frequency) + 1, 
                                        freq = f'{payment_frequency}ME', 
                                        inclusive = 'both',
                                        )
            return pd.DataFrame({
                'Payment Date': date_series,# + pd.DateOffset(months = payment_frequency),
                'Month Number': (pd.Series([1,] * date_series.shape[0]).cumsum() - 1) * payment_frequency,
                'Payment Number': pd.Series([1,] * date_series.shape[0]).cumsum() - 1,
                'Payment Accumulation': pd.Series([payment_frequency,] * date_series.shape[0]), # This might be removed if utilisation of 'Payment Accumulation' is substituted by self.payment_frequency
            })
        
        else:
            if tenor / payment_frequency < 1: # if tenor / payment frequency is less than 1 then the next payment is the final one. 
                 date_series = pd.Series([end_date]) # We report the start date of the payment period
            else:
                date_series = pd.date_range(start = start_date, # Start from the first payment date
                                            periods = round(maturity / payment_frequency) + 1, # Needs to be tenor to be compatible with 
                                            freq = f'{payment_frequency}ME', 
                                            inclusive = 'both',
                                            )
                
            lifetime_df = pd.DataFrame({
                'Payment Date': date_series, #+ pd.DateOffset(months = payment_frequency),
                'Month Number': (pd.Series([1,] * date_series.shape[0]).cumsum() - 1) * payment_frequency,
                'Payment Number': pd.Series([1,] * date_series.shape[0]).cumsum() - 1,
                'Payment Accumulation': pd.Series([payment_frequency,] * date_series.shape[0]), # This might be removed if utilisation of 'Payment Accumulation' is substituted by self.payment_frequency
            })
            
            mask_present_future_dates = (
                (lifetime_df['Payment Date'].dt.year > self.date.year) |
                ( (lifetime_df['Payment Date'].dt.year == self.date.year) & (lifetime_df['Payment Date'].dt.month >= self.date.month) )
                )
            
            out_df = lifetime_df[mask_present_future_dates].copy()
            out_df['Payment Number'] = np.arange(len(out_df))

        return out_df
    

    def generate_payment_schedule(
        self, 
        payment_date_schedule: pd.DataFrame, 
    ) -> pd.DataFrame:
        
        df = payment_date_schedule
        df['Principal Current'] = self.principal
        # n_payments = len(df)

        if self.payment_type == 'bullet':
            df['Interest Rate Applied'] = ((self.effective_interest_rate + 1)**(self.maturity / 12)) - 1
        else:
            df['Interest Rate Applied'] = ((self.effective_interest_rate + 1)**(df['Payment Accumulation'] / 12)) - 1

        # The following section takes care of the logic for bullet, IO, and linear payments and those payments which have only one payment left
        if (self.payment_type in ['bullet', 'interest_only', 'linear']) or (len(df) == 1): 

            if self.payment_type in ['bullet', 'interest_only',]:
                df['Principal Payment'] = [0,] * (len(df) - 1) + [self.principal] 
            
            elif self.payment_type == 'linear':
                ########## If origination month is included as row 0, should it be divided by (len(df) - 1) (?)
                df['Principal Payment'] = round(self.principal / len(df), 2)
            
            elif len(df) == 1: 
                # One payment means we are in the final payment. Here the annuity payment will be the same as the following calculation since the remaining interest and principal will be repaid
                df['Principal Payment'] = self.principal

            # Once the payment goes through on the payment date the final principal will be the outstanding amount
            df['Principal Outstanding'] = (df['Principal Current'] - df['Principal Payment'].cumsum()).clip(lower = 0).round(2)

            # Interest was accumulated on the principal before the payment so we take a step back to calculate the interest rate
            # we need to do it like this because of the cumulative sum applied to principal current before
            if self.payment_type == 'bullet':
                df['Interest Payment'] = [0.,] * (len(df) - 1) + [ round(self.principal * (-1 + (self.effective_interest_rate + 1)**(self.maturity / 12)), 2) ]
            else:
                df['Interest Payment'] = ((df['Principal Outstanding'] + df['Principal Payment']) * df['Interest Rate Applied']).round(2)

            ## JP:  Modified to Total Payment = Principal Payment + Interest Payment instead of
            #             Total Payment = Principal Outstanding + Interest Payment
            # df['Total Payment'] = df['Principal Outstanding'] + df['Interest Payment']
            df['Total Payment'] = df['Principal Payment'] + df['Interest Payment']
        
        elif self.payment_type == 'annuity':
            # For annuity loans we use the numpy_financial library to calculate the payments which are necessary
            """
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
        
        # print(self.identifier)
        """
        Creation of a simplified version of the payment_schedule DataFrame. This is created to only store essential information needed for the expect loss (EL)
        value, while keeping the payment_schedule intact, should debugging be necessary.
        """
        portfolio_schedule_copy = portfolio_schedule.copy()

        portfolio_schedule_copy['Short Payment Date'] = pd.PeriodIndex(
            portfolio_schedule_copy['Payment Date'], 
            freq = 'M'
        )
        
        self.payment_schedule['Short Payment Date'] = pd.PeriodIndex(
            self.payment_schedule['Payment Date'], 
            freq = 'M'
        )
    
        result_df = portfolio_schedule_copy.merge(
            self.payment_schedule[['Short Payment Date', 'Total Payment', 'Month Number']],
            on = 'Short Payment Date',
            how = 'left'
        )

        result_df['Total Payment'] = result_df['Total Payment'].fillna(0.)

        result_df['Exposure'] = result_df['Total Payment'][::-1].cumsum()
        result_df.drop(columns = ['Short Payment Date'], inplace = True)
        
        self.payment_schedule.drop(columns = ['Short Payment Date'], inplace = True)
 
        ##### Drop Rows after maturity is hit
        max_month_id = result_df['Month Number'].idxmax()
        max_value = result_df.loc[max_month_id, 'Month Number']
        
        result_df = result_df.loc[:max_month_id].reset_index(drop = True)
        result_df.loc[::-1, 'Month Number'] = np.arange(max_value, max_value - len(result_df), -1)

        return result_df
    

    def calculate_pd_and_expected_loss(
        self,
        df: pd.DataFrame,
        marginal_pd_arr: np.ndarray
    ) -> pd.DataFrame:
        """
        Combines PD alignment and expected loss calculation in a single function.
        First month is discarded, so 'Month Number' = 1 corresponds to first repayment month.
        
        Args:
            df (pd.DataFrame): Dataframe on which to append the new columns
            marginal_pd_arr (np.ndarray): Array of marginal PD values

        Returns:
            pd.DataFrame: DataFrame with added 'Marginal PD' and 'PD_Exposure' columns
        """
        # Get reference to DataFrame to avoid copies
        # df = self.el_df
        
        # Assign marginal PD values
        df['Marginal PD'] = marginal_pd_arr[
            df['Month Number'].to_numpy(dtype = np.int32), self.status
            ]
        
        # Calculate PD exposure directly
        df['PD*Exposure'] = df['Exposure'] * df['Marginal PD']
        
        output_columns = ['Payment Date', 'Month Number', 'Total Payment', 'Exposure', 'Marginal PD', 'PD*Exposure']
        return df[output_columns]
