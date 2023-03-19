"""borsdata_client.py"""

import pandas as pd  # pandas is a data-analysis library for python (data frames)
import numpy as np
import matplotlib.pylab as plt  # matplotlib for visual-presentations (plots)
import datetime as dt  # datetime for date- and time-stuff
from dateutil.relativedelta import relativedelta

import os

from borsdata.borsdata_api import BorsdataAPI
from borsdata import constants as constants  # user constants
from borsdata.general_tools import list_grouper


# pandas options for string representation of data frames (print)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class BorsdataClient:
    def __init__(self):
        self._borsdata_api = BorsdataAPI()
        self._instruments_with_meta_data = pd.DataFrame()

    def get_fundamental_data(self, ins_id: {int, list}, reporting_item: {str, list}, report_type: str = 't12m',
                             method_missing_report_date: str = 'estimate')->{pd.DataFrame, dict}:
        """
        Returns a DataFrame (or a dictionary of DataFrames if a list of reporting items have been provided) with a
        specified reporting item for a given reporting type (yearly, quarterly or T12M)
        :param ins_id: list of int
        :param reporting_item: str or list of str
        :param report_type: str or list of str
        :param method_missing_report_date: str 'drop' will remove all rows where there is no report date. 'estimate'
        will sett 'report_date' to 2 months after 'report_end_date'
        :return: DataFrame or dict of DataFrames
        """
        # convert inputs to list
        if not isinstance(ins_id, list):
            ins_id = [ins_id]

        pre_pivot_result_df = self._get_fundamental_data(ins_id=ins_id, report_type=report_type)

        # handle missing report dates
        pre_pivot_result_df = self.handle_missing_report_date(report_df=pre_pivot_result_df,
                                                              method=method_missing_report_date)

        # pivot the result
        if isinstance(reporting_item, list):
            # if a list of reporting items have been specified return a dictionary with the reporting item name as key
            # and the result DataFrame as value
            result = {}
            for item in reporting_item:
                if item not in pre_pivot_result_df.columns:
                    raise ValueError(f"'{item}' is not a recognized report item\nChose from: '%s'" % "', '".join(pre_pivot_result_df.columns))
                result[item] = pre_pivot_result_df.pivot_table(values=item, index='report_date', columns='stock_id', aggfunc='sum')
            return result
        else:
            return pre_pivot_result_df.pivot_table(values=reporting_item, index='report_date', columns='stock_id', aggfunc='sum')

    @staticmethod
    def handle_missing_report_date(report_df: pd.DataFrame, method: str)->pd.DataFrame:
        """
        Returns a DataFrame that either drops the rows where 'report_date' is missing or replaces is with an estimated
        date as two months after 'reporting_end_date'
        :param report_df:
        :param method:
        :return:
        """

        if method.lower() == 'drop':
            return report_df[~report_df['report_date'].isin([0, '1899-12-31T00:00:00'])].copy()
        elif method.lower() == 'estimate':
            # assume that the report date is 2 months after the ending period
            months_after_end = 2
            report_df['report_end_date'] = pd.to_datetime(report_df['report_end_date'])
            index_missing_report_date = report_df[report_df['report_date'].isin([0, '1899-12-31T00:00:00'])].index
            report_df.loc[index_missing_report_date, 'report_date'] = report_df.loc[index_missing_report_date, 'report_end_date'].apply(lambda x: x + relativedelta(months=months_after_end))
            report_df['report_date'] = pd.to_datetime(report_df['report_date'])
            return report_df
        else:
            raise ValueError(f"'{method}' is not a recongized method\nchose between 'drop' and 'estimate'")

    def _get_fundamental_data(self, ins_id: list, report_type: str):
        """
        Returns a DataFrame with reporting items as columns for a specified reporting type (e.g. 'annual')
        :param ins_id: list of int
        :param report_type: str
        :return: DataFrame
        """
        result_df = pd.DataFrame()  # initialize the result DataFrame
        for ins_id_sub_list in list_grouper(ins_id, 50):  # max 50 instruments per call
            # download the reporting data
            if report_type.lower() in ['y', 'annual', '10k', '10-k', 'year', 'yearly']:
                api_result_df = self._borsdata_api.get_annual_reports(stock_id_list=ins_id_sub_list)
            elif report_type.lower() in ['q', 'quarterly', '10q', '10-q', 'quarter']:
                api_result_df = self._borsdata_api.get_quarterly_reports(stock_id_list=ins_id_sub_list)
            elif report_type.lower() in ['t12m', 'r12m']:
                api_result_df = self._borsdata_api.get_rolling_12_month_reports(stock_id_list=ins_id_sub_list)
            else:
                raise ValueError(f"'{report_type}' is not a recognized report type")

            # merge the result
            result_df = pd.concat([result_df, api_result_df])

        result_df.reset_index(drop=True, inplace=True)
        return result_df

    def instruments_with_meta_data_extended(self):
        """
        Returns a DataFrame with instrument id and extended meta data
        A filter can be applied to the results
        :return: DataFrame
        """
        # download the instruments and translation DataFrames
        eng_to_swe_df = self._borsdata_api.get_translation_metadata()
        instrument_df = self._borsdata_api.get_instruments()

        # for each type of meta data, map the values to the corrsponding data id
        meta_data_names_map = {'countries': 'country', 'branches': 'branch', 'sectors': 'sector', 'markets': 'market'}

        for meta_data_name in meta_data_names_map.keys():
            meta_df = self._borsdata_api._get_metadata(meta_data_name=meta_data_name)
            meta_df.reset_index(inplace=True)
            if meta_data_name == 'markets':
                # special case with no need for translation and the name is a combination of the exchange name (e.g OMX
                # Stockholm) and descriptive name (e.g. Small cap)
                translate = False
                meta_df[['name']] = meta_df[['exchangeName']] + ' ' + meta_df[['name']].values
            else:
                translate = True
            if translate:
                instrument_df[f'{meta_data_name}_name_swe'] = instrument_df[f'{meta_data_names_map[meta_data_name]}Id'].map(meta_df.set_index('id')['name'])
                instrument_df[f'{meta_data_name}_name_eng'] = instrument_df[f'{meta_data_name}_name_swe'].map(eng_to_swe_df.set_index('nameSv')['nameEn'])
            else:
                instrument_df[f'{meta_data_name}_name'] = instrument_df[f'{meta_data_names_map[meta_data_name]}Id'].map(meta_df.set_index('id')['name'])
        # map instrument type
        ins_type_id_map = {0: 'Stock', 1: 'Pref', 2: 'Index', 3: 'Stocks2', 4: 'SectorIndex', 5: 'IndustryIndex',
                           8: 'SPAC', 13: 'Index GI'}
        instrument_df['instrument_type'] = instrument_df['instrument'].map(ins_type_id_map)
        return instrument_df

    def instruments_with_meta_data(self):
        """
        creating a csv and xlsx of the APIs instrument-data (including meta-data)
        and saves it to path defined in constants (default ../file_exports/)
        :return: pd.DataFrame of instrument-data with meta-data
        """
        if len(self._instruments_with_meta_data) > 0:
            return self._instruments_with_meta_data
        else:
            self._borsdata_api = BorsdataAPI()
            # fetching data from api
            countries = self._borsdata_api.get_countries()
            branches = self._borsdata_api.get_branches()
            sectors = self._borsdata_api.get_sectors()
            markets = self._borsdata_api.get_markets()
            instruments = self._borsdata_api.get_instruments()
            # instrument type dict for conversion (https://github.com/Borsdata-Sweden/API/wiki/Instruments)
            instrument_type_dict = {0: 'Aktie', 1: 'Pref', 2: 'Index', 3: 'Stocks2', 4: 'SectorIndex',
                                    5: 'BranschIndex', 8: 'SPAC', 13: 'Index GI'}
            # creating an empty dataframe
            instrument_df = pd.DataFrame()
            # loop through the whole dataframe (table) i.e. row-wise-iteration.
            for index, instrument in instruments.iterrows():
                ins_id = index
                name = instrument['name']
                ticker = instrument['ticker']
                isin = instrument['isin']
                # locating meta-data in various ways
                # dictionary-lookup
                instrument_type = instrument_type_dict[instrument['instrument']]
                # .loc locates the rows where the criteria (inside the brackets, []) is fulfilled
                # located rows (should be only one) get the column 'name' and return its value-array
                # take the first value in that array ([0], should be only one value)
                market = markets.loc[markets.index == instrument['marketId']]['name'].values[0]
                country = countries.loc[countries.index == instrument['countryId']]['name'].values[0]
                sector = 'N/A'
                branch = 'N/A'
                # index-typed instruments does not have a sector or branch
                if market.lower() != 'index':
                    sector = sectors.loc[sectors.index == instrument['sectorId']]['name'].values[0]
                    branch = branches.loc[branches.index == instrument['branchId']]['name'].values[0]
                # appending current data to dataframe, i.e. adding a row to the table.
                df_temp = pd.DataFrame([{'name': name, 'ins_id': ins_id, 'ticker': ticker, 'isin': isin,
                                         'instrument_type': instrument_type,
                                         'market': market, 'country': country, 'sector': sector, 'branch': branch}])
                instrument_df = pd.concat([instrument_df, df_temp], ignore_index=True)
            # create directory if it do not exist
            if not os.path.exists(constants.EXPORT_PATH):
                os.makedirs(constants.EXPORT_PATH)
            # to csv
            instrument_df.to_csv(constants.EXPORT_PATH + 'instrument_with_meta_data.csv')
            # creating excel-document
            excel_writer = pd.ExcelWriter(constants.EXPORT_PATH + 'instrument_with_meta_data.xlsx')
            # adding one sheet
            instrument_df.to_excel(excel_writer, 'instruments_with_meta_data')
            # saving the document
            excel_writer.save()
            self._instruments_with_meta_data = instrument_df
            return instrument_df

    def plot_stock_prices(self, ins_id):
        """
        Plotting a matplotlib chart for ins_id
        :param ins_id: instrument id to plot
        :return:
        """
        # creating api-object
        # using api-object to get stock prices from API
        stock_prices = self._borsdata_api.get_instrument_stock_prices(ins_id)
        # calculating/creating a new column named 'sma50' in the table and
        # assigning the 50 day rolling mean to it
        stock_prices['sma50'] = stock_prices['close'].rolling(window=50).mean()
        # filtering out data after 2015 for plot
        filtered_data = stock_prices[stock_prices.index > dt.datetime(2015, 1, 1)]
        # plotting 'close' (with 'date' as index)
        plt.plot(filtered_data['close'], color='blue', label='close')
        # plotting 'sma50' (with 'date' as index)
        plt.plot(filtered_data['sma50'], color='black', label='sma50')
        # show legend
        plt.legend()
        # show plot
        plt.show()

    def top_performers(self, market, country, number_of_stocks=5, percent_change=1):
        """
        function that prints top performers for given parameters in the terminal
        :param market: which market to search in e.g. 'Large Cap'
        :param country: which country to search in e.g. 'Sverige'
        :param number_of_stocks: number of stocks to print, default 5 (top5)
        :param percent_change: number of days for percent change calculation
        :return: pd.DataFrame
        """
        # creating api-object
        # using defined function above to retrieve dataframe of all instruments
        instruments = self.instruments_with_meta_data()
        # filtering out the instruments with correct market and country
        filtered_instruments = instruments.loc[(instruments['market'] == market) & (instruments['country'] == country)]
        # creating new, empty dataframe
        stock_prices = pd.DataFrame()
        # looping through all rows in filtered dataframe
        for index, instrument in filtered_instruments.iterrows():
            # fetching the stock prices for the current instrument
            instrument_stock_price = self._borsdata_api.get_instrument_stock_prices(int(instrument['ins_id']))
            instrument_stock_price.sort_index(inplace=True)
            # calculating the current instruments percent change
            instrument_stock_price['pct_change'] = instrument_stock_price['close'].pct_change(percent_change)
            # getting the last row of the dataframe, i.e. the last days values
            last_row = instrument_stock_price.iloc[[-1]]
            # appending the instruments name and last days percent change to new dataframe
            df_temp = pd.DataFrame([{'stock': instrument['name'], 'pct_change': round(last_row['pct_change'].values[0] * 100, 2)}])
            stock_prices = pd.concat([stock_prices, df_temp], ignore_index=True)
        # printing the top sorted by pct_change-column
        print(stock_prices.sort_values('pct_change', ascending=False).head(number_of_stocks))
        return stock_prices

    def history_kpi(self, kpi, market, country, year):
        """
        gathers and concatenates historical kpi-values for provided kpi, market and country
        :param kpi: kpi id see https://github.com/Borsdata-Sweden/API/wiki/KPI-History
        :param market: market to gather kpi-values from
        :param country: country to gather kpi-values from
        :param year: year for terminal print of kpi-values
        :return: pd.DataFrame of historical kpi-values
        """
        # creating api-object
        # using defined function above to retrieve data frame of all instruments
        instruments = self.instruments_with_meta_data()
        # filtering out the instruments with correct market and country
        filtered_instruments = instruments.loc[(instruments['market'] == market) & (instruments['country'] == country)]
        # creating empty array (to hold data frames)
        frames = []
        # looping through all rows in filtered data frame
        for index, instrument in filtered_instruments.iterrows():
            # fetching the stock prices for the current instrument
            instrument_kpi_history = self._borsdata_api.get_kpi_history(int(instrument['ins_id']), kpi, 'year', 'mean')
            # check to see if response holds any data.
            if len(instrument_kpi_history) > 0:
                # resetting index and adding name as a column
                instrument_kpi_history.reset_index(inplace=True)
                instrument_kpi_history.set_index('year', inplace=True)
                instrument_kpi_history['name'] = instrument['name']
                # appending data frame to array
                frames.append(instrument_kpi_history.copy())
        # creating concatenated data frame with concat
        symbols_df = pd.concat(frames)
        # the data frame has the columns ['year', 'period', 'kpi_value', 'name']
        # show year ranked from highest to lowest, show top 5
        print(symbols_df[symbols_df.index == year].sort_values('kpiValue', ascending=False).head(5))
        return symbols_df

    def get_latest_pe(self, ins_id):
        """
        Prints the PE-ratio of the provided instrument id
        :param ins_id: ins_id which PE-ratio will be calculated for
        :return:
        """
        # creating api-object
        # fetching all instrument data
        reports_quarter, reports_year, reports_r12 = self._borsdata_api.get_instrument_reports(3)
        # getting the last reported eps-value
        reports_r12.sort_index(inplace=True)
        print(reports_r12.tail())
        last_eps = reports_r12['earningsPerShare'].values[-1]
        # getting the stock prices
        stock_prices = self._borsdata_api.get_instrument_stock_prices(ins_id)
        stock_prices.sort_index(inplace=True)
        # getting the last close
        last_close = stock_prices['close'].values[-1]
        # getting the last date
        last_date = stock_prices.index.values[-1]
        # getting instruments data to retrieve the name of the ins_id
        instruments = self._borsdata_api.get_instruments()
        instrument_name = instruments[instruments.index == ins_id]['name'].values[0]
        # printing the name and calculated PE-ratio with the corresponding date. (array slicing, [:10])
        print(f"PE for {instrument_name} is {round(last_close / last_eps, 1)} with data from {str(last_date)[:10]}")

    def breadth_large_cap_sweden(self, sma_lag: int = 50):
        """
        plots the breadth (number of stocks above moving-average 40) for Large Cap Sweden compared
        to Large Cap Sweden Index
        :param sma_lag: int
        :return None
        """
        # creating api-object
        # using defined function above to retrieve data frame of all instruments
        instruments = self.instruments_with_meta_data()
        # filtering out the instruments with correct market and country
        filtered_instruments = instruments.loc[
            (instruments['market'] == "Large Cap") & (instruments['country'] == "Sverige")]
        # creating empty array (to hold data frames)
        frames = []
        # looping through all rows in filtered data frame
        for index, instrument in filtered_instruments.iterrows():
            # fetching the stock prices for the current instrument
            instrument_stock_prices = self._borsdata_api.get_instrument_stock_prices(int(instrument['ins_id']))
            # using numpy's where function to create a 1 if close > ma40, else a 0
            instrument_stock_prices[f'above_ma40'] = np.where(
                instrument_stock_prices['close'] > instrument_stock_prices['close'].rolling(window=sma_lag).mean(), 1, 0)
            instrument_stock_prices['name'] = instrument['name']
            # check to see if response holds any data.
            if len(instrument_stock_prices) > 0:
                # appending data frame to array
                frames.append(instrument_stock_prices.copy())
        # creating concatenated data frame with concat
        symbols_df = pd.concat(frames)
        symbols_df = symbols_df.groupby('date').sum()
        # fetching OMXSLCPI data from api
        omx = self._borsdata_api.get_instrument_stock_prices(643)
        # aligning data frames
        omx = omx[omx.index > '2015-01-01']
        symbols_df = symbols_df[symbols_df.index > '2015-01-01']
        # creating subplot
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        # plotting
        ax1.plot(omx['close'], label="OMXSLCPI")
        ax2.plot(symbols_df[f'above_ma40'], label="number of stocks above ma40")
        # show legend
        ax1.legend()
        ax2.legend()
        plt.show()


if __name__ == "__main__":
    from borsdata.general_tools import apply_column_filter
    # Main, call functions here.
    # creating BorsdataClient-instance
    borsdata_client = BorsdataClient()
    ins_df = borsdata_client.instruments_with_meta_data_extended()
    selection_filter = {
        'markets_name': ['omx stockholm large cap', 'omx stockholm mid cap', 'omx stockholm small cap'],
        'instrument_type': 'stock',
        'countries_name_eng': 'sweden'
    }
    ins_id = list(apply_column_filter(ins_df, selection_filter).index)
    yahoo_tickers = ins_df.loc[ins_id]['yahoo'].values

    eps_df = borsdata_client.get_fundamental_data(ins_id=ins_id, reporting_item='earnings_per_share')
    eps_df.columns = yahoo_tickers
    eps_df.to_clipboard()

    # eps_df = borsdata_client.get_fundamental_data(ins_id=[3, 195], reporting_item=['earnings_per_share', 'revenues'])

    # borsdata_client._borsdata_api.get_countries()

    # # calling some methods
    # borsdata_client.breadth_large_cap_sweden()
    # borsdata_client.get_latest_pe(87)
    # borsdata_client.instruments_with_meta_data()
    # borsdata_client.plot_stock_prices(3)  # ABB
    # borsdata_client.history_kpi(2, 'Large Cap', 'Sverige', 2020)  # 2 == Price/Earnings (PE)
    # borsdata_client.top_performers('Large Cap', 'Sverige', 10,
    #                                5)  # showing top10 performers based on 5 day return (1 week) for Large Cap Sverige.
