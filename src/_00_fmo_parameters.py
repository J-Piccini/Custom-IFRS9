from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DataPreparationParameters:
      """
      Parameters used to load and pre-process the data
      """

      ##### Data Loading/Dumping #####
      path_to_main_folder: str = r'C:\Users\jacopo.piccini\OneDrive - PROMETEIA SPA\Desktop\FMO\_96_Final Application\Assets&Borrowers'
      path_to_origination_folder: str = r'C:\Users\jacopo.piccini\OneDrive - PROMETEIA SPA\Desktop\FMO\_96_Final Application\Test Set'

      ##### Portfolio re-naming #####
      ptf_mapping_dict: Dict[str, str] = field(default_factory = lambda: {
            'Access Bank PLC': 'Access Bank', 
            'Ameriabank': 'Ameriabank', 
            'Araratbank OJSC' : 'Ararat',
            'Ararat Bank': 'Ararat',
            'ArdshinBank Armenia': 'Ardshinbank', 
            'Bank Al Etihad': 'Bank Al Etihad', 
            'Capital Bank of Jordan': 'Capital Bank of Jordan',
            'CIB': 'Commercial International Bank', 
            'Equity Bank Kenya': 'Equity Bank Kenya', 
            'I&M Bank' : 'I&M Bank Kenya',
            'I&M Bank Kenya': 'I&M Bank Kenya', 
            'I&M Bank Limited' : 'I&M Bank Kenya',
            'I&M Bank Rwanda': 'I&M Rwanda',
            'Sidian Bank Limited' : 'Sidian',
            'Sidian Bank': 'Sidian', 
            'Tamweelcom': 'Tamweelcom', 
            'Terabank': 'Terabank', 
            'Vitas Palestine': 'Vitas',
            }
      )
      
      ptf_to_country_mapping_dict: Dict[str, str] = field(default_factory = lambda: {
           'Access Bank': 'nigeria',
           'Ameriabank': 'armenia',
           'Ararat': 'armenia',
           'Ardshinbank': 'armenia',
           'Bank Al Etihad': 'jordan',
           'Capital Bank of Jordan': 'jordan',
           'Commercial International Bank': 'egypt',
           'Equity Bank Kenya': 'kenya',
           'I&M Bank Kenya': 'kenya',
           'I&M Rwanda': 'rwanda',
           'Sidian': 'kenya',
           'Tamweelcom': 'jordan',
           'Terabank': 'georgia'
           }
      )

      geographical_grouping_dict: Dict[str, str] = field(default_factory = lambda: {
            'nigeria': 'africa',
            'armenia': 'near_europe',
            'jordan': 'middle_east',
            'egypt': 'middle_east',
            'kenya': 'africa',
            'rwanda': 'africa',
            'georgia': 'near_europe'
            }
      )

      mixed_types_ids_portfs: list = field(default_factory = lambda: [
           'Equity Bank Kenya'
           ]
      )

      arr_threshold: int = 45
      def_threshold: int = 180
      
      num_labels: list = field(default_factory = lambda: [0, 1, 2, 3, 4])
      str_labels: list = field(default_factory = lambda: [
            'Performing',                                      # DPD == 0
            f'DPD <{45}',                                      # 1  <= DPD < 45
            'DPD <90',                                         # 45 <= DPD < 90
            f'DPD <{180}',                                     # 90 <= DPD < 180
            f'DPD >= {180}'                                    # DPD >= 90
            ]
      )
      
      columns_short_list: list = field(default_factory = lambda: [
            'Period', 'Reporting_Date', 'Portfolio', 'Loan_ID', 'Originator_Country', # Loan identification columns
            'Loan_Arrear_days', 'PrommiseLoanStatus',                                 # Status identification columns 
            'Loan_TenureMonths', 'Loan_StartDate', 'Loan_EndDate',                    # Remaining maturity calculation
            'Loan_BalanceOriginal', 'Loan_BalanceDisbursed', 'Loan_BalanceCurrent'    # Exposure related columns
            ]
      )

      ptf_class_columns_short_list: list = field(default_factory = lambda: [
           'Period', 'Reporting_Date', 'Portfolio', 'Loan_ID', 'Originator_Country',            # Loan identification columns
           'Loan_Arrear_days', 'PrommiseLoanStatus',                                            # Status identification columns 
           'Loan_TenureMonths', 'Loan_StartDate', 'Loan_EndDate',                               # Remaining maturity calculation
           'Loan_BalanceCurrent',              # Exposure related columns
           'Loan_Currency', 'Loan_InterestRateEffective', 'TypeRepayment', 'PaymentFrequency'  # Additional columns for application purposes
            ]
      )


@dataclass
class ProbabilityOfDefaultParameters:
    """
    """
    time_resolution: int = 12              # Months between subsequent snapshot
    time_projection: int = 10              # Years for which migration matrices will be projected
    max_obs_per_ptf: int = 3               # Maximum 3 years per portfolio
    arr_days_btp: int = 0                  # Missing loans with less than or equal to these threshold will be considered as performing at closure
    
    num_status: int = 5                    # Number of status used to develop the model
    atol_check: float = 1e-10              # Absolute tolerance used to check that migration matrices' rows sum up to 1 (requirement for Markov's Chain)
    maturity_split_value: int = 36         # 

    ptf_macro_split: Dict[str, str] = field(default_factory = lambda: {
         'Access Bank': 'africa',
         'Ameriabank': 'near_europe',
         'Ararat': 'near_europe',
         'Ardshinbank': 'near_europe',
         'Bank Al Etihad': 'middle_east',
         'Capital Bank of Jordan': 'middle_east',
         'Commercial International Bank': 'middle_east',
         'Equity Bank Kenya': 'africa',
         'I&M Bank Kenya': 'africa',
         'I&M Rwanda': 'africa',
         'Sidian': 'africa',
         'Tamweelcom': 'middle_east',
         'Terabank': 'near_europe'
      })
    

@dataclass
class CreditConversionFactorParameters:
      """
      Expositions over time, and 
      """
      path_to_def_red_folders: str = r'C:\Users\jacopo.piccini\OneDrive - PROMETEIA SPA\Desktop\FMO\_96_Final Application\Defaulted&Redeemed'
     #  path_to_int_rep_folders: str = r'C:\Users\jacopo.piccini\OneDrive - PROMETEIA SPA\Desktop\FMO\New Internal Report Files'
      

      ptfs_to_be_excluded: list = field(default_factory = lambda: [
           ]                             
      )
      
      vintages_start_list: list = field(default_factory = lambda: [
           1, 4, 7, 13
           ]
      )

      perf_window_list: list = field(default_factory = lambda: [
            2, 4, 6, 8, 10, 12, 14, 16
            ]                            
      )

      ccf_guarantee_dict: Dict[str, float | np.ndarray] = field(default_factory = lambda: {
           'Access Bank': 10_250_000_000,
           'Ameriabank': 50_000_000,
           'Ararat': np.array([10_000_000, 20_000_000]),
           'Ardshinbank': 7_772_000_000,
           'Bank Al Etihad': 21_270_000,
           'Capital Bank of Jordan': 14_180_000,
           'Commercial International Bank': 50_000_000,
           'Equity Bank Kenya': 5_575_000_000,
           'I&M Bank Kenya': 1_765_543_500,
           'I&M Rwanda': 11_514_500_000 ,
           'Sidian': 1_650_000_000,
           'Tamweelcom': 7_090_000,
           'Terabank': 39_000_000,
           }
      )
      
      early_sid_dict: Dict[str, bool | pd.Timestamp] = field(default_factory = lambda: {
          'Access Bank': pd.to_datetime('30-11-2024', dayfirst = True),
          'Ameriabank': False,
          'Ararat': False,
          'Ardshinbank': False,
          'Bank Al Etihad': False,
          'Capital Bank of Jordan': pd.to_datetime('31-10-2024', dayfirst = True),
          'Commercial International Bank': False,
          'Equity Bank Kenya': pd.to_datetime('30-06-2024', dayfirst = True),
          'I&M Bank Kenya': False,
          'I&M Rwanda': False,
          'Sidian': pd.to_datetime('30-11-2023', dayfirst = True),
          'Tamweelcom': False,
          'Terabank': False,
          }
     )

      ccf_guarantee_change_snap_dict: Dict[str, float | np.ndarray] = field(default_factory = lambda: {
            'Access Bank': np.nan,
            'Ameriabank': np.nan,
            'Ararat': np.array([202304]),
            'Ardshinbank': np.nan,
            'Bank Al Etihad': np.nan,
            'Capital Bank of Jordan': np.nan,
            'Commercial International Bank': np.nan,
            'Equity Bank Kenya': np.nan,
            'I&M Bank Kenya': np.nan,
            'I&M Rwanda': np.nan ,
            'Sidian': np.nan,
            'Tamweelcom': np.nan,
            'Terabank': np.nan,
            }
      )


@dataclass
class ApplicationParameters:
     """
     Fixed parameters needed for the application phase
     """
     ################################################
     ########## Portfolio classes creation ##########
     ################################################
     ptf_age_dict: Dict[str, int] = field(default_factory = lambda:{
          'Access Bank': 15,
          'Ameriabank': 12,
          'Ararat': 41,
          'Ardshinbank': 4,
          'Bank Al Etihad': 38,
          'Capital Bank of Jordan': 32,
          'Commercial International Bank': 17,
          'Equity Bank Kenya': 47,
          'I&M Bank Kenya': 23,
          'I&M Rwanda': 9,
          'Sidian': 34,
          'Tamweelcom': 19,
          'Terabank': 12
          }
     )


     cover_stop_date_dict: Dict[str, pd.Timestamp] = field(default_factory = lambda: {
           'Access Bank': pd.to_datetime('31-03-2030', dayfirst = True),
           'Ameriabank': pd.to_datetime('08-01-2031', dayfirst = True),
           'Ararat': pd.to_datetime('18-08-2028', dayfirst = True),
           'Ardshinbank': pd.to_datetime('11-09-2031', dayfirst = True),
           'Bank Al Etihad': pd.to_datetime('07-07-2028', dayfirst = True),
           'Capital Bank of Jordan': pd.to_datetime('31-03-2030', dayfirst = True),
           'Commercial International Bank': pd.to_datetime('30-06-2030', dayfirst = True),
           'Equity Bank Kenya': pd.to_datetime('28-02-2028', dayfirst = True),
           'I&M Bank Kenya': pd.to_datetime('22-02-2030', dayfirst = True),
           'I&M Rwanda': pd.to_datetime('01-03-2031', dayfirst = True),
           'Sidian': pd.to_datetime('25-03-2028', dayfirst = True),
           'Tamweelcom': pd.to_datetime('12-05-2030', dayfirst = True),
           'Terabank': pd.to_datetime('30-01-2031', dayfirst = True),
           }
      )
     
     scheduled_sid_dict: Dict[str, pd.Timestamp] = field(default_factory = lambda: {     
          'Access Bank': pd.to_datetime('31-03-2027', dayfirst = True),
          'Ameriabank': pd.to_datetime('08-01-2027', dayfirst = True),
          'Ararat': pd.to_datetime('18-08-2025', dayfirst = True),
          'Ardshinbank': pd.to_datetime('11-09-2028', dayfirst = True),
          'Bank Al Etihad': pd.to_datetime('07-07-2025', dayfirst = True),
          'Capital Bank of Jordan': pd.to_datetime('31-03-2027', dayfirst = True),
          'Commercial International Bank': pd.to_datetime('30-06-2027', dayfirst = True),
          'Equity Bank Kenya': pd.to_datetime('28-02-2026', dayfirst = True),
          'I&M Bank Kenya': pd.to_datetime('22-02-2027', dayfirst = True),
          'I&M Rwanda': pd.to_datetime('01-03-2028', dayfirst = True),
          'Sidian': pd.to_datetime('25-03-2025', dayfirst = True),
          'Tamweelcom': pd.to_datetime('12-05-2026', dayfirst = True),
          'Terabank': pd.to_datetime('30-01-2027', dayfirst = True),
          }
      )
     
     early_sid_dict: Dict[str, bool | pd.Timestamp] = field(default_factory = lambda: {
          'Access Bank': pd.to_datetime('30-11-2024', dayfirst = True),
          'Ameriabank': False,
          'Ararat': False,
          'Ardshinbank': False,
          'Bank Al Etihad': False,
          'Capital Bank of Jordan': pd.to_datetime('31-10-2024', dayfirst = True),
          'Commercial International Bank': False,
          'Equity Bank Kenya': pd.to_datetime('30-06-2024', dayfirst = True),
          'I&M Bank Kenya': False,
          'I&M Rwanda': False,
          'Sidian': pd.to_datetime('30-11-2023', dayfirst = True),
          'Tamweelcom': False,
          'Terabank': False,
          }
      )

     datapoint_p_year: int = 12
     n_projection_years: int = 10

     portfolio_tranches_dict: Dict[str, np.ndarray] = field(default_factory = lambda: {
          'Access Bank': np.array([0.05, 0.09, 0.01, 0.85]),
          'Ameriabank': np.array([0.050, 0.198, 0.022, 0.730]),
          'Ararat': np.array([0.050000, 0.179775, 0.019975, 0.750250]),
          'Ardshinbank': np.array([0.050, 0.315, 0.035, 0.600]),
          'Bank Al Etihad': np.array([0.0500, 0.1200, 0.0133, 0.8167]),
          'Capital Bank of Jordan': np.array([0.0575, 0.12375, 0.01375, 0.805]),
          'Commercial International Bank': np.array([0.050, 0.135, 0.015, 0.800]),
          'Equity Bank Kenya': np.array([0.0600, 0.1116, 0.0124, 0.8160]),
          'I&M Bank Kenya': np.array([0.050, 0.198, 0.022, 0.730]),
          'I&M Rwanda': np.array([0.0500, 0.3660, 0.0407, 0.5433]),
          'Sidian': np.array([0.06, 0.18, 0.02, 0.74]),
          'Tamweelcom': np.array([0.050, 0.225, 0.0250, 0.700]),
          'Terabank': np.array([0.05, 0.18, 0.02, 0.75])
          }                                                          
      )
     
     guarantee_dict: Dict[str, float] = field(default_factory = lambda: {
           'Access Bank': 10_250_000_000,
           'Ameriabank': 50_000_000,
           'Ararat': 20_000_000,
           'Ardshinbank': 7_772_000_000,
           'Bank Al Etihad': 21_270_000,
           'Capital Bank of Jordan': 14_180_000,
           'Commercial International Bank': 50_000_000,
           'Equity Bank Kenya': 5_575_000_000,
           'I&M Bank Kenya': 1_765_543_500,
           'I&M Rwanda': 11_514_500_000 ,
           'Sidian': 1_650_000_000,
           'Tamweelcom': 7_090_000,
           'Terabank': 39_000_000,
           }
      )
     
     covered_interest_dict: Dict[str, bool] = field(default_factory = lambda: {
          'Access Bank': False,
          'Ameriabank': False,
          'Ararat': True,
          'Ardshinbank': False,
          'Bank Al Etihad': False,
          'Capital Bank of Jordan': False,
          'Commercial International Bank': False,
          'Equity Bank Kenya': False,
          'I&M Bank Kenya': True,
          'I&M Rwanda': False ,
          'Sidian': False,
          'Tamweelcom': False,
          'Terabank': True,
          }
     )   


@dataclass
class SatelliteModelsParameters:
     
     variables_meaning_dict: Dict[str, str] = field(default_factory = lambda: { 
          'NY.GDP.MKTP.CD': 'GDP', 
          'PA.NUS.FCRF': 'FX_rate', 
          'DT.DOD.DECT.CD': 'External_Debt'
          }
     )
     
     target_var: str = 'FB.AST.NPER.ZS'

     country_renaming_dict: Dict[str, str] = field(default_factory = lambda: {
          'Brunei Darussalam': 'Brunei',
          'Congo, Dem. Rep.': 'Congo, Democratic Republic of',
          'Congo, Rep.':'Congo',
          'Gambia, The':'Gambia',
          'Korea, Rep.':'Korea, North',
          'Kyrgyz Republic':'Kyrgyzstan',
          'Micronesia, Fed. Sts.':'Micronesia',
          'North Macedonia':'Macedonia',
          'Slovak Republic':'Slovakia',
          'St. Kitts and Nevis':'Saint Kitts and Nevis',
          'St. Lucia':'Saint Lucia',
          'St. Vincent and the Grenadines':'Saint Vincent and the Grenadines',
          'Turkiye':'Turkey',
          'United States':'US',
          'Viet Nam':'Vietnam'
          }
     )
     
     countries_to_be_excluded: list = field(default_factory = lambda: [
          'West Bank and Gaza',
          'Korea, Rep.',
          'China',
          'Hong Kong SAR, China',
          'Japan',
          'Macao SAR, China',
          'Russian Federation', 
          'Afghanistan', 
          'Iraq', 
          'United Arab Emirates',
          'Saudi Arabia',
          'Israel',
          'Canada', 
          'US'
          ]                             
      )    
     
     countries_in_africa: list = field(default_factory = lambda: [
          'Eswatini'
          ]
      )
     
     countries_in_europe: list = field(default_factory = lambda: [
          'Czechia',
          'Kosovo'
          ]
      )
     
     countries_in_asia: list = field(default_factory = lambda: [
          'Armenia',
          'Jordan',
          'Georgia'
          ]
      )
     
     columns_to_lag: list = field(default_factory = lambda: [
          'NPL_ratio_inv', 
          'GDP', 
          'FX_rate', 
          'External_Debt'
          ]
     )

@dataclass
class OriginationParameters:
     """
     Fixed parameters needed for the application phase
     """
     ################################################
     ########## Portfolio classes creation ##########
     ################################################

     cover_stop_date_dict: Dict[str, pd.Timestamp] = field(default_factory = lambda: {
          'Ameriabank': pd.to_datetime('08-01-2031', dayfirst = True),
          }
      )
     
     scheduled_sid_dict: Dict[str, pd.Timestamp] = field(default_factory = lambda: {     
          'Ameriabank': pd.to_datetime('08-01-2029', dayfirst = True),
          }
      )
     
     early_sid_dict: Dict[str, bool | pd.Timestamp] = field(default_factory = lambda: {
          'Ameriabank': False,
          }
      )

     datapoint_p_year: int = 12
     n_projection_years: int = 10

     portfolio_tranches_dict: Dict[str, np.ndarray] = field(default_factory = lambda: {
          'Ameriabank': np.array([0.050, 0.198, 0.022, 0.730]),
          }                                                          
      )
     
     guarantee_dict: Dict[str, float] = field(default_factory = lambda: {
          'Ameriabank': 50_000_000,
          }
      )
     
     covered_interest_dict: Dict[str, bool] = field(default_factory = lambda: {
          'Ameriabank': False,
          }
     )   