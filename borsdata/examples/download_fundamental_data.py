"""download_fundamental_data.py"""

import os
from datetime import datetime

from borsdata.borsdata_client import BorsdataClient
from borsdata.general_tools import apply_column_filter
from borsdata.constants import EXPORT_PATH


def main():
    # parameters
    save_csv = True
    copy_to_clipboard = False
    reporting_item = 'earnings_per_share'
    selection_filter = {
        'markets_name': ['omx stockholm large cap', 'omx stockholm mid cap', 'omx stockholm small cap'],
        'instrument_type': 'stock',
        'countries_name_eng': 'sweden',
    }

    # call data from api
    borsdata_client = BorsdataClient()
    instrument_df = borsdata_client.instruments_with_meta_data_extended()

    # select the instrument ids and yahoo tickers
    filtered_instrument_df = apply_column_filter(instrument_df, selection_filter)
    instrument_id = list(filtered_instrument_df.index)
    yahoo_tickers = instrument_df.loc[instrument_id]['yahoo'].values

    # download fundamental data
    result_df = borsdata_client.get_fundamental_data(ins_id=instrument_id, reporting_item=reporting_item)
    result_df.columns = yahoo_tickers

    if save_csv:
        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        result_df.to_csv(EXPORT_PATH + f"{reporting_item}_{datetime.now().strftime('%Y%m%d')}.csv")

    if copy_to_clipboard:
        result_df.to_clipboard()


if __name__ == '__main__':
    main()

