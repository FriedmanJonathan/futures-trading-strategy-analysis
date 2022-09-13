def MarketOrder(trades_df, starting_time, units_to_execute, transaction_stage):
    import pandas as pd
    YesPrint = False
    # Function takes in bid df (for long buy order) or ask df (for short sell order)
    # which has been pre-filtered based on current time and executes a MARKET order.
    # Returns dataframe with all transaction details: time, entry price, number of units.
    if transaction_stage == "entry": entry_exit_column_name = "Entry Price"
    else: entry_exit_column_name = "Exit Price"
    transaction_entry_df = pd.DataFrame(columns=["Datetime", entry_exit_column_name, "Units"])  # "entry" = bought or shorted!
    trades_df = trades_df.loc[(trades_df['datetime'] > starting_time)]
    trade_index = 0
    # dataframe would be initialized in advance, input to function and output at the end.
    # (initially missing buy/sell price depending on buy/short)
    while units_to_execute > 0:  # find+replace units_to_execute
        if units_to_execute >= trades_df["volume"][trade_index]:  # input df would be bid/ask/trade
            units_executed = trades_df["volume"][trade_index]
        else:
            units_executed = units_to_execute
        units_to_execute -= units_executed
        transaction_price = trades_df["price"][trade_index]
        transaction = pd.DataFrame([[trades_df['datetime'][trade_index], transaction_price, units_executed]],
                                   columns=["Datetime", entry_exit_column_name, "Units"])
        transaction_entry_df = pd.concat([transaction_entry_df, transaction], axis=0)
        trade_index += 1
    current_time = trades_df['datetime'][trade_index]
    if YesPrint: print(f"{entry_exit_column_name}s")
    if YesPrint: print(transaction_entry_df)
    return transaction_entry_df, current_time


def ConvertToOHLC(df, sampling_number, sampling_unit_of_time):
    sampling_text = sampling_number + sampling_unit_of_time
    ticks = df.loc[:, ['price', 'volume']]
    df = df.drop(columns=["trade_type"])
    df = ticks.price.resample(sampling_text).ohlc()
    volumes = ticks.volume.resample(sampling_text).sum()
    df['volume'] = volumes
    return df


def UpdateToCurrentTime(df, current_time):
    current_time_index = df.index.searchsorted(current_time)
    current_df = df.iloc[current_time_index:]
    return current_df


def TradesInTimeframe(trades_df, start_time, duration):
    import pandas as pd
    from datetime import timedelta
    start_time = pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S.%f')
    end_time = start_time + timedelta(minutes = int(duration))
    start_time_index = trades_df.index.searchsorted(start_time)
    end_time_index = trades_df.index.searchsorted(end_time)
    try:
        filtered_trades_df = trades_df.iloc[start_time_index:end_time_index]
        open_price = filtered_trades_df["price"][0]
        close_price = filtered_trades_df["price"][-1]
        volume = filtered_trades_df["volume"].sum()
    except IndexError:
        open_price = 'NA'
        close_price = 'NA'
        volume = 0
    return open_price, close_price, volume, end_time


def WaitUntilExitPrice(df, low_value, high_value):
    i = 0  # Index to iterate over trades until selling price found.
    while low_value < df["price"][i] < high_value:
        i += 1
        # if current time is starting candle plus a day, break.
    current_time = df["datetime"][i]
    current_df = UpdateToCurrentTime(df, current_time)
    return current_df, current_time


def WaitForVolume(df, units_to_wait_for, trade_type, price_threshold):
    i = 0
    if trade_type == "long":
        while units_to_wait_for > 0:
            if df["price"][i] > price_threshold:
                units_to_wait_for -= df["volume"][i]
            i += 1
    else:
        while units_to_wait_for > 0:
            if df["price"][i] < price_threshold:
                units_to_wait_for -= df["volume"][i]
            i += 1

        # if current time is starting candle plus a day, break.
    current_time = df["datetime"][i]
    current_df = UpdateToCurrentTime(df, current_time)
    return current_df, current_time


def ExecuteTradingStrategy(strategy, trades_df, asks_df, bids_df, csv_year, csv_month):
    #columns = ["Time Candle", "Duration", "Momentum Margin", "Momentum Volume", "Desired Profit",
              #"Exit at Loss", "Units", "Open-close Difference"])
    from calendar import monthrange
    import pandas as pd
    global YesPrint
    YesPrint = False
    days_in_month = monthrange(int(csv_year), int(csv_month))[1]
    daily_transaction_df = pd.DataFrame(columns=["Month", "Day", "Units Sold", "Profit"])
    transactions_df = pd.DataFrame(columns=["Datetime", "Buy Price", "Sell Price", "Units Sold", "Profit per unit", "Total Profit"])
    for day_number in range(1, days_in_month):
        if YesPrint: print(f"Day number {day_number}:")
        # Using current candle, find open and close prices, volume, and current time.
        time_string = csv_year+'-'+csv_month+'-'+str(day_number).zfill(2)+" "+strategy["Time Candle"]+":00.000"
        open_price, close_price, volume, current_time = TradesInTimeframe(trades_df, time_string, strategy["Duration"])
        if YesPrint: print(f"Open Price: {open_price}, Close Price: {close_price}, Volume: {volume}")

        # We check the desired candle and make a buying/shorting decision based on it.
        if volume > 0:  # else non-trading day, except day where key doesn't exist.
            if close_price > open_price + strategy["Open-close Difference"]:  # buy order.
                buy_price = close_price + strategy["Momentum Margin"]
                if YesPrint: print(f"Buy price is {buy_price}.")
                #current_trades_df, current_time = WaitForVolume(trades_df, momentum_volume, "long", buy_price)

                current_asks_df = UpdateToCurrentTime(asks_df, current_time)
                current_bids_df = UpdateToCurrentTime(bids_df, current_time)
                transaction_entry_df, current_time = MarketOrder(current_asks_df, current_time, strategy["Units"], transaction_stage="entry")

                # After all units bought, determine sell price (both profit and stoploss) based on average buy price:
                average_buy_price = transaction_entry_df["Entry Price"].dot(transaction_entry_df["Units"]) / strategy["Units"]
                sell_profit_price = average_buy_price + strategy["Desired Profit"]  # don't forget to round
                sell_stoploss_price = average_buy_price - strategy["Stop Loss"]
                if YesPrint: print(f"Average buy price is: {average_buy_price}, sell for profit at {sell_profit_price} or for loss at {sell_stoploss_price}.")

                current_bids_df, current_time = WaitUntilExitPrice(current_bids_df, low_value = sell_stoploss_price, high_value = sell_profit_price)


                # Now we have to figure out how to use the data from the "buy" dataframe and sell those units.
                # Technically we can work with the average buy price, but doing this intelligently can let us
                # expand code later for FIFO, LIFO etc.
                transaction_exit_df, current_time = MarketOrder(current_bids_df, current_time, strategy["Units"],
                                                                transaction_stage="exit")
                average_sell_price = transaction_exit_df["Exit Price"].dot(transaction_exit_df["Units"]) / strategy["Units"]
                daily_profit = (average_sell_price - average_buy_price) * strategy["Units"]
                units_sold = strategy["Units"]  # BUT WHAT IF THERE'S A PARTIAL SALE???

            elif close_price < open_price - strategy["Open-close Difference"]: # short
                short_price = close_price - strategy["Momentum Margin"]
                if YesPrint: print(f"Short price is {short_price}.")
                current_asks_df = UpdateToCurrentTime(asks_df, current_time)
                current_bids_df = UpdateToCurrentTime(bids_df, current_time)
                transaction_entry_df, current_time = MarketOrder(current_bids_df, current_time, strategy["Units"],
                                                                 transaction_stage="entry")

                average_short_price = transaction_entry_df["Entry Price"].dot(transaction_entry_df["Units"]) / strategy["Units"]
                rebuy_profit_price = average_short_price - strategy["Desired Profit"]  # don't forget to round
                rebuy_stoploss_price = average_short_price + strategy["Stop Loss"]
                if YesPrint: print(
                    f"Average short price is: {average_short_price}, rebuy for profit at {rebuy_profit_price} or for loss at {rebuy_stoploss_price}.")

                current_asks_df, current_time = WaitUntilExitPrice(current_asks_df, low_value = rebuy_profit_price, high_value = rebuy_stoploss_price)

                transaction_exit_df, current_time = MarketOrder(current_asks_df, current_time, strategy["Units"],
                                                                transaction_stage="exit")
                average_rebuy_price = transaction_exit_df["Exit Price"].dot(transaction_exit_df["Units"]) / strategy["Units"]
                daily_profit = (average_short_price - average_rebuy_price) * strategy["Units"]
                units_sold = strategy["Units"]

            else:
                if YesPrint: print(f"No sale on month #{csv_month}, day {day_number}")
                units_sold = 0
                daily_profit = 0

            daily_transactions = pd.DataFrame([[csv_month, day_number, units_sold, daily_profit]],
                                              columns=["Month", "Day", "Units Sold", "Profit"])
            if YesPrint: print(daily_transactions)

        else:
            if YesPrint: print(f"No sale on month #{csv_month}, day {day_number}")
            daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]],
                                              columns=["Month", "Day", "Units Sold", "Profit"])

        daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)


    if YesPrint: print(daily_transaction_df)
    total_profit = daily_transaction_df["Profit"].sum()
    if YesPrint: print(f"Total profit is {total_profit}")


    #strategy_result = pd.DataFrame([[time_candle, sampling_number, momentum_margin, desired_profit, stop_loss,
    #                                 desired_units, open_close_difference, total_profit]],
    #                               columns=["Time Candle", "Duration", "Momentum Margin", "Desired Profit", "Exit at Loss",
    #                                        "Units", "Open-close Difference", "Total Profit"])

    #return strategy_result
    return total_profit

def main():

    # Parameters and reading the data

    csv_year = "2021"
    csv_month = "10"
    csv_name = "F.US.NGEF22_" + csv_year + csv_month + ".csv"
    csv_name_filtered = "F.US.NGEF22_" + csv_year + csv_month + "_filtered.csv"
    futures_df = pd.read_csv(csv_name)

    #parameters = {"candle_duration": 10, "sampling_unit_of_time": 'min'}
    #candle_duration = '10'

    #candle_durations = [5,10]
    candle_durations = ['10','15']
    #sampling_unit_of_time = 'min' - currently in function
    #time_candles = ["09:00", "12:00", "14:00", "14:20"]
    time_candles = ["9:00","10:30"]
    #units_to_buy = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50]
    units_to_buy = [5]
    #momentum_margins = [0,3,5,7,10]
    momentum_margins = [1,2,3,4,5]
    momentum_volumes = [0]
    #desired_profits = range(5,25,5)
    desired_profits = [10]
    #stop_losses - defined using desired profit
    stop_losses = [10]

    #indicator_candle_strengths = [3,5,7,10]
    indicator_candle_strengths = [1,2,3,4,5]
    #time_candle = "10:30"
    #momentum_margin = 5
    #desired_profit = 10
    #stop_loss = 10
    #desired_units = 5
    #open_close_difference = 0
    #momentum_volume = 3

    strategies_df = pd.DataFrame(columns=["Time Candle", "Duration", "Momentum Margin", "Momentum Volume", "Desired Profit",
                                                 "Stop Loss", "Units", "Open-close Difference"])
    strategies = []

    for time_candle in time_candles:
        for candle_duration in candle_durations:
            for desired_units in units_to_buy:
                for momentum_margin in momentum_margins:
                    for momentum_volume in momentum_volumes:
                        for desired_profit in desired_profits:
                            for stop_loss in stop_losses:
                                for open_close_difference in indicator_candle_strengths:
                                    strategy = pd.DataFrame([[time_candle, candle_duration, momentum_margin, momentum_volume,
                                                              desired_profit, stop_loss, desired_units, open_close_difference]],
                                        columns=["Time Candle", "Duration", "Momentum Margin", "Momentum Volume", "Desired Profit",
                                                 "Stop Loss", "Units", "Open-close Difference"])
                                    strategies_df = pd.concat([strategies_df, strategy], axis=0)
    print(strategies_df)

    # 2. Extract the date from the relevant column and turn into a time series
    # + basic visualization

    # Naming columns, filtering by trades only (no bid/ask), changing date and time to Time format,
    # adjusting by an hour (CT to EST), and dropping unnecessary columns.
    futures_df.columns = ["future_type", "datetime", "session", "time", "price", "trade_type", "market", "correction", "volume"]
    future_type = futures_df["future_type"][0]
    futures_df["time"] = futures_df["time"].map(str).str.zfill(9)  # Adds zeros before time (fixes bug later with time comparisons)
    futures_df["datetime"] = futures_df["datetime"].map(str) + futures_df["time"]
    futures_df["datetime"] = pd.to_datetime(futures_df["datetime"], format='%Y%m%d%H%M%S%f') + timedelta(hours = 1)
    futures_df = futures_df.set_index(pd.DatetimeIndex(futures_df["datetime"]))
    futures_df = futures_df.drop(columns=["future_type", "session", "time", "market", "correction"])

    # Splitting the data into bids/asks/trades data, and switching to OHLC+volume format

    bids_df = futures_df[futures_df["trade_type"] == "B"]
    asks_df = futures_df[futures_df["trade_type"] == "A"]
    trades_df = futures_df[futures_df["trade_type"] == "T"]
    #bids_df = ConvertToOHLC(bids_df, candle_duration, sampling_unit_of_time)  # use map?
    #asks_df = ConvertToOHLC(asks_df, candle_duration, sampling_unit_of_time)
    trades_df_ohlc = ConvertToOHLC(trades_df, candle_duration, 'min')



    # Optional: exporting data, candlestick chart.
    #trades_df.to_csv(csv_name_filtered)
    #mpf.plot(trades_df_ohlc, type='candle', volume=True, title='Natural Gas - October')

    # Planning ahead: GUI selects commodity, time-frame, show visuals, executes strategy, real-time.
    start_time = time.time()
    strategies_df["Total Profit"] = strategies_df.parallel_apply(ExecuteTradingStrategy, args=(trades_df, asks_df, bids_df, csv_year, csv_month), axis=1)
    #strategy_result = ExecuteTradingStrategy(trades_df, asks_df, bids_df, time_candle, candle_duration, momentum_margin,
    #                                           momentum_volume, desired_profit, stop_loss, desired_units, open_close_difference)
    # strategies_df = pd.concat([strategies_df, strategy_result], axis=0)
    print(strategies_df)


    #freeze_support()
    #pool = Pool(processes=8)
    #rslt = pool.map(ExecuteTradingStrategy, strategies)
    #pool.close()
    #pool.join()

    #print(rslt)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    # Packages and importing raw data. (Future - year and month on loop, or determined by GUI.
    import pandas as pd
    import mplfinance as mpf
    #from calendar import monthrange
    from datetime import timedelta
    from multiprocessing import Pool, freeze_support
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    import time
    main()
# Features: momentum margin, profit, stoploss, start time, duration, units, day_of_week, month, volume_wait, how many units in volume wait, market or limit
# 6. Create a baseline model (all yes / all no)
# Definitely a good logistic regression just to understand correlation and importance
# Don't forget to normalize!!
# 7. Create a predictive model and compare to baseline:
# 8. Visualization of results + extras.