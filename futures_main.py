def MarketOrder(trades_df, starting_time, units_to_execute, transaction_entry_df, transaction_stage = 'entry'):
    # Function takes in bid df (for long buy order) or ask df (for short sell order)
    # which has been pre-filtered based on current time and executes a MARKET order.
    # Returns dataframe with all transaction details: time, entry price, number of units.

    import pandas as pd  # re-import necessary for running Pandarallel with Windows
    DebugMode = True  # Pandarallel doesn't do global variables.
    if DebugMode: print(f"Market Triggered with {units_to_execute} left to execute.")

    if transaction_stage == "entry": entry_exit_column_name = "Entry Price"
    else: entry_exit_column_name = "Exit Price"

    if transaction_entry_df.empty: # else exists (e.g. after limit sale) and we add to it.
        transaction_entry_df = pd.DataFrame(columns=["Datetime", entry_exit_column_name, "Units"])  # "entry" = bought or shorted!
    trades_df = trades_df.loc[(trades_df['datetime'] > starting_time)]
    if trades_df.empty: return 0, 0, 0  # = we've run out of data, so this final trading day shouldn't count.

    # The following loop analyzes each row
    trade_index = 0
    units_left_to_execute = units_to_execute
    try:
        while units_left_to_execute > 0:  # find+replace units_to_execute
            if units_left_to_execute >= trades_df["volume"][trade_index]:  # input df would be bid/ask/trade
                units_executed = trades_df["volume"][trade_index]
            else:
                units_executed = units_left_to_execute
            units_left_to_execute -= units_executed
            transaction_price = trades_df["price"][trade_index]
            transaction = pd.DataFrame([[trades_df['datetime'][trade_index], transaction_price, units_executed]],
                                       columns=["Datetime", entry_exit_column_name, "Units"])
            transaction_entry_df = pd.concat([transaction_entry_df, transaction], axis=0)
            trade_index += 1
    except IndexError:
        trade_index -= 1
        print ("Market Order ran out of data - Index Error")
        return 0,0,0

    current_time = trades_df['datetime'][trade_index]
    if DebugMode: print(f"{entry_exit_column_name}s")
    if DebugMode: print(transaction_entry_df)
    units_executed = units_to_execute - units_left_to_execute
    return transaction_entry_df, current_time, units_executed


def LimitOrder(trades_df, starting_time, units_to_execute, transaction_stage, bid_or_ask, limit_price, stoploss_price = None, time_to_wait = 30):
    import pandas as pd
    from datetime import timedelta
    DebugMode = True
    # Function takes in bid df (for long buy order) or ask df (for short sell order)
    # which has been pre-filtered based on current time and executes a LIMIT order.
    # Returns dataframe with all transaction details: time, entry price, number of units.
    if transaction_stage == "entry": entry_exit_column_name = "Entry Price"
    else: entry_exit_column_name = "Exit Price"
    transaction_entry_df = pd.DataFrame(columns=["Datetime", entry_exit_column_name, "Units"])  # "entry" = bought or shorted!
    time_limit = starting_time + timedelta(minutes=time_to_wait)
    print(f"Time limit: {time_limit}, Time to wait: {time_to_wait}")
    trades_df = trades_df.loc[(starting_time < trades_df['datetime'])]
    trade_index = 0
    # dataframe would be initialized in advance, input to function and output at the end.
    # (initially missing buy/sell price depending on buy/short)

    # Add timedelta to set end time for filter mentioned above, exception which returns units bought
    units_left_to_execute = units_to_execute
    try:
        while units_left_to_execute > 0 and trades_df['datetime'][trade_index] < time_limit:  # technically we might not be able to buy or sell all units, need a time limit.

            ExecuteTrade = False
            if transaction_stage == "entry":
                if bid_or_ask == 'bid':
                    if (trades_df["price"][trade_index] <= limit_price):
                        ExecuteTrade = True
                if bid_or_ask == 'ask':
                    if trades_df["price"][trade_index] >= limit_price:
                        ExecuteTrade = True

            else: # We're exiting the trade and would therefore also need to consider the stop-loss price.
                if bid_or_ask == 'bid':  # we've shorted and want to rebuy at lower price for profit
                    if trades_df["trade_type"][trade_index] == "A":
                        if trades_df["price"][trade_index] <= limit_price:
                            ExecuteTrade = True
                    else:   # transaction is a trade. We check if the stop-loss price is reached;
                        # if so, exit limit order function and sell all at market!
                        if trades_df["price"][trade_index] >= stoploss_price:
                            if DebugMode: print("Short stop-loss triggered.")
                            raise IndexError  # to exit loop, finish up Limit function and move on to Market function
                        # otherwise (a trade that didn't trigger stop-loss), continues without buying
                if bid_or_ask == 'ask':
                    if trades_df["trade_type"][trade_index] == "B":
                        if trades_df["price"][trade_index] >= limit_price:
                            ExecuteTrade = True
                    else:
                        if trades_df["price"][trade_index] <= stoploss_price:
                            if DebugMode: print("Buy stop-loss triggered.")
                            raise IndexError

            if ExecuteTrade:
                if units_left_to_execute >= trades_df["volume"][trade_index]:  # input df would be bid/ask/trade
                    units_executed = trades_df["volume"][trade_index]
                else:
                    units_executed = units_left_to_execute
                units_left_to_execute -= units_executed
                transaction_price = trades_df["price"][trade_index]
                transaction = pd.DataFrame([[trades_df['datetime'][trade_index], transaction_price, units_executed]],
                                           columns=["Datetime", entry_exit_column_name, "Units"])
                transaction_entry_df = pd.concat([transaction_entry_df, transaction], axis=0)
            trade_index += 1
    except IndexError:
        trade_index -= 1   # To allow update to current time afterwards

    current_time = trades_df['datetime'][trade_index]
    if DebugMode: print(f"{entry_exit_column_name}s")
    if DebugMode: print(transaction_entry_df)
    units_executed = units_to_execute - units_left_to_execute
    return transaction_entry_df, current_time, units_executed


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


def WaitForVolume(df, units_to_wait_for, trade_type, price_threshold, start_time):
    i = 0
    from datetime import timedelta
    import pandas as pd
    start_time = pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S.%f')
    day_later = start_time + timedelta(minutes=24*60)
    try:
        if trade_type == "long":
            while units_to_wait_for > 0:
                if df["datetime"][i] > day_later: raise IndexError
                if df["price"][i] >= price_threshold:
                    units_to_wait_for -= df["volume"][i]
                i += 1
        else:
            while units_to_wait_for > 0:
                if df["datetime"][i] > day_later: raise IndexError
                if df["price"][i] <= price_threshold:
                    units_to_wait_for -= df["volume"][i]
                i += 1

            # if current time is starting candle plus a day, break.
        current_time = df["datetime"][i]
        current_df = UpdateToCurrentTime(df, current_time)
        return current_df, current_time, True  # True indicates that
    except IndexError:
        current_time = df["datetime"][i-1]
        current_df = UpdateToCurrentTime(df, current_time)
        return current_df, current_time, False


def ExecuteTradingStrategy(strategy, trades_df, asks_df, bids_df, asks_and_trades_df, bids_and_trades_df, csv_year, csv_month):
    # This is essentially the main function which is run for many strategies in parallel.
    # A strategy consists of many parameters, the output of this function is the profit (or loss) from running said strategy on
    # the raw data.
    from calendar import monthrange   # Re-imports necessary for Pandarallel.
    import pandas as pd
    global DebugMode
    DebugMode = True
    days_in_month = monthrange(int(csv_year), int(csv_month))[1]
    daily_transaction_df = pd.DataFrame(columns=["Month", "Day", "Units Sold", "Profit"])
    transactions_df = pd.DataFrame(columns=["Datetime", "Buy Price", "Sell Price", "Units Sold", "Profit per unit", "Total Profit"])
    for day_number in range(1, days_in_month):
        if DebugMode: print(f"Day number {day_number}:")
        # Using current candle, find open and close prices, volume, and current time.
        time_string = csv_year+'-'+csv_month+'-'+str(day_number).zfill(2)+" "+strategy["Time Candle"]+":00.000"
        start_time = pd.to_datetime(time_string, format='%Y-%m-%d %H:%M:%S.%f')
        open_price, close_price, volume, current_time = TradesInTimeframe(trades_df, time_string, strategy["Duration"])
        if DebugMode: print(f"Open Price: {open_price}, Close Price: {close_price}, Volume: {volume}")

        # We check the desired candle and make a buying/shorting decision based on it.
        if volume > 0:  # else non-trading day, except day where key doesn't exist.
            if close_price > open_price + strategy["Open-close Difference"]:  # buy order.
                buy_price = close_price + strategy["Momentum Margin"]

                # STEP 1: WAITING FOR VOLUME - after buy price determined, wait for x number of trades to be bought at buy price. if a full day passes
                # without such a sale, we don't wait any longer and document that no trade was executed on that day. ("continue" moves on to next day)
                current_trades_df = UpdateToCurrentTime(trades_df, current_time)
                current_trades_df, current_time, less_than_day = WaitForVolume(current_trades_df, strategy["Momentum Volume"], "long", buy_price, start_time)
                if less_than_day == False:
                    daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]], columns=["Month", "Day", "Units Sold", "Profit"])
                    daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)
                    continue

                if DebugMode:
                    print(f"Buy price is {buy_price}.")
                    print(f"Current time is: {current_time}")

                # STEP 2: PLACING THE BUY ORDER - we are bidding, and comparing to everyone else's asks (current_asks_df)
                current_asks_df = UpdateToCurrentTime(asks_df, current_time)
                if strategy["Order Type"] == "Limit":
                    transaction_entry_df, current_time, units_bought = LimitOrder(current_asks_df, current_time, strategy["Units"],
                                                                transaction_stage="entry", bid_or_ask='bid',
                                                                limit_price=buy_price, time_to_wait=strategy["Buy Wait Time"])
                else:  # Market
                    transaction_entry_df = pd.DataFrame(columns=["Datetime", "Entry Price", "Units"])  # initialization
                    transaction_entry_df, current_time, units_bought = MarketOrder(current_asks_df, current_time, strategy["Units"], transaction_entry_df, transaction_stage="entry")

                current_bids_df = UpdateToCurrentTime(bids_df, current_time)
                current_bids_and_trades_df = UpdateToCurrentTime(bids_and_trades_df, current_time)

                # STEP 2b: IF NO BUY ORDER IS PLACED
                if units_bought == 0:  # If no sale executed, document and move on to next day.
                    daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]],
                                                      columns=["Month", "Day", "Units Sold", "Profit"])
                    daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)
                    continue

                # STEP 3 - After all units bought, determine sell price (both profit and stoploss) based on average buy price:
                # (consider implementing FIFO, LIFO, etc. later).
                average_buy_price = transaction_entry_df["Entry Price"].dot(transaction_entry_df["Units"]) / units_bought
                sell_profit_price = average_buy_price + strategy["Desired Profit"]  # don't forget to round
                sell_stoploss_price = average_buy_price - strategy["Stop Loss"] * strategy["Desired Profit"]  # stop-loss is % of profit
                if DebugMode:
                    print(f"Average buy price is: {average_buy_price}, sell for profit at {sell_profit_price} or for loss at {sell_stoploss_price}.")
                    print(f"Current time is: {current_time}")

                # STEP 4 - SELL ORDER (either limit until time expires and sell rest market, or flat out market once price reached.
                if strategy["Order Type"] == "Limit":
                    transaction_exit_df, current_time, units_sold = LimitOrder(current_bids_and_trades_df, current_time, units_bought,
                                                               transaction_stage="exit", bid_or_ask='ask',
                                                               limit_price=sell_profit_price,
                                                               stoploss_price=sell_stoploss_price, time_to_wait=strategy["Sell Wait Time"])
                    # If there are still things unsold after 24 hours, fire-sale.
                    units_left_to_sell = units_bought - units_sold
                    current_bids_df = UpdateToCurrentTime(bids_df, current_time)
                    if DebugMode: print(f"Current time after limit sale is: {current_time}")

                    transaction_exit_df, current_time, units_sold = MarketOrder(current_bids_df, current_time, units_left_to_sell, transaction_exit_df, transaction_stage="exit")
                    if DebugMode: print(f"Current time after market sale is: {current_time}")

                else:  # Market
                    current_bids_df, current_time = WaitUntilExitPrice(current_bids_df, low_value = sell_stoploss_price, high_value = sell_profit_price)
                    transaction_exit_df = pd.DataFrame(columns=["Datetime", "Exit Price", "Units"])  # initialization
                    transaction_exit_df, current_time, units_sold = MarketOrder(current_bids_df, current_time, units_bought,
                                                                 transaction_exit_df, transaction_stage="exit")

                if units_sold == 0:
                    daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]], columns=["Month", "Day", "Units Sold", "Profit"])
                    daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)
                    continue

                # STEP 5 - SUMMARY
                #print(transaction_exit_df)
                average_sell_price = transaction_exit_df["Exit Price"].dot(transaction_exit_df["Units"]) / units_bought
                daily_profit = (average_sell_price - average_buy_price) * units_bought
                units_sold = units_bought

            elif close_price < open_price - strategy["Open-close Difference"]: # short
                short_price = close_price - strategy["Momentum Margin"]
                if DebugMode:
                    print(f"Short price is {short_price}.")
                    print(f"Current time is: {current_time}")

                # STEP 1: WAITING FOR VOLUME
                current_trades_df = UpdateToCurrentTime(trades_df, current_time)
                current_trades_df, current_time, less_than_day = WaitForVolume(current_trades_df, strategy["Momentum Volume"], "short",
                                                                short_price, start_time)
                if less_than_day == False:
                    daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]],
                                                      columns=["Month", "Day", "Units Sold", "Profit"])
                    daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)
                    continue

                # STEP 2: PLACING THE BUY ORDER
                current_bids_df = UpdateToCurrentTime(bids_df, current_time)

                if strategy["Order Type"] == "Limit":
                    transaction_entry_df, current_time, units_shorted = LimitOrder(current_bids_df, current_time, strategy["Units"],
                                                                transaction_stage="entry", bid_or_ask='ask',
                                                                limit_price=short_price, time_to_wait=strategy["Buy Wait Time"])
                else:
                    transaction_entry_df = pd.DataFrame(columns=["Datetime", "Entry Price", "Units"])  # initialization
                    transaction_entry_df, current_time, units_shorted = MarketOrder(current_bids_df, current_time, strategy["Units"],
                                                                     transaction_entry_df, transaction_stage="entry")
                current_asks_df = UpdateToCurrentTime(asks_df, current_time)

                # STEP 2b: IF NO SHORT ORDER IS PLACED
                if units_shorted == 0:  # If no sale executed, document and move on to next day.
                    daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]],
                                                      columns=["Month", "Day", "Units Sold", "Profit"])
                    daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)
                    continue

                # STEP 3 - After all units shorted, determine rebuy price (both profit and stoploss) based on average short price:
                average_short_price = transaction_entry_df["Entry Price"].dot(transaction_entry_df["Units"]) / units_shorted
                rebuy_profit_price = average_short_price - strategy["Desired Profit"]  # don't forget to round
                rebuy_stoploss_price = average_short_price + strategy["Stop Loss"] * strategy["Desired Profit"]
                if DebugMode:
                    print(f"Average short price is: {average_short_price}, rebuy for profit at {rebuy_profit_price} or for loss at {rebuy_stoploss_price}.")
                    print(f"Current time is: {current_time}")

                # STEP 4 - SELL ORDER (either limit until time expires and sell rest market, or flat out market once price reached.
                current_asks_and_trades_df = UpdateToCurrentTime(asks_and_trades_df, current_time)
                if strategy["Order Type"] == "Limit":
                    transaction_exit_df, current_time, units_rebought = LimitOrder(current_asks_and_trades_df, current_time, units_shorted,
                                                    transaction_stage="exit", bid_or_ask='bid', limit_price=rebuy_profit_price,
                                                    stoploss_price=rebuy_stoploss_price, time_to_wait=strategy["Sell Wait Time"])
                    if DebugMode: print(f"Current time after limit sale is: {current_time}")

                    units_left_to_rebuy = units_shorted - units_rebought
                    current_asks_df = UpdateToCurrentTime(asks_df, current_time)
                    transaction_exit_df, current_time, units_rebought = MarketOrder(current_asks_df, current_time,
                                                                                    units_left_to_rebuy,
                                                                                    transaction_exit_df,
                                                                                    transaction_stage="exit")
                    if DebugMode: print(f"Current time after market sale is: {current_time}")

                else: # Market
                    current_asks_df, current_time = WaitUntilExitPrice(current_asks_df, low_value = rebuy_profit_price, high_value = rebuy_stoploss_price)
                    transaction_exit_df = pd.DataFrame(columns=["Datetime", "Exit Price", "Units"])  # initialization
                    transaction_exit_df, current_time, units_rebought = MarketOrder(current_asks_df, current_time, units_shorted,
                                                                     transaction_exit_df, transaction_stage="exit")

                if units_rebought == 0:
                    daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]], columns=["Month", "Day", "Units Sold", "Profit"])
                    daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)
                    continue

                # STEP 5 - SUMMARY
                average_rebuy_price = transaction_exit_df["Exit Price"].dot(transaction_exit_df["Units"]) / units_shorted
                daily_profit = (average_short_price - average_rebuy_price) * units_shorted
                units_sold = units_shorted

            else:  # very minor candle (low open-close difference), therefore no trade.
                if DebugMode: print(f"No sale on month #{csv_month}, day {day_number}")
                units_sold = 0
                daily_profit = 0

            daily_transactions = pd.DataFrame([[csv_month, day_number, units_sold, daily_profit]],
                                              columns=["Month", "Day", "Units Sold", "Profit"])
            if DebugMode: print(daily_transactions)

        else:  # no volume = non-trading day (such as a weekend)
            if DebugMode: print(f"No sale on month #{csv_month}, day {day_number}")
            daily_transactions = pd.DataFrame([[csv_month, day_number, 0, 0]],
                                              columns=["Month", "Day", "Units Sold", "Profit"])

        # For ALL cases, adding the daily transaction.
        daily_transaction_df = pd.concat([daily_transaction_df, daily_transactions], axis=0)


    if DebugMode: print(daily_transaction_df)
    total_profit = daily_transaction_df["Profit"].sum()
    if DebugMode: print(f"Total profit is {total_profit}")

    return total_profit

def main():

    # Parameters and reading the data, will be turned into a data concatenation function
    # if we need data from multiple months

    csv_year = "2021"
    csv_month = "10"
    csv_name = "F.US.NGEF22_" + csv_year + csv_month + ".csv"
    csv_name_filtered = "F.US.NGEF22_" + csv_year + csv_month + "_filtered.csv"
    futures_df = pd.read_csv(csv_name)
    candle_durations = ['5']
    time_candles = ["09:00"]
    units_to_buy = [50] # [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50]
    momentum_margins = [7]
    momentum_volumes = [1]
    desired_profits = [20]  # can go into greater resolution
    stop_losses = [3] # [0.5,1,2] and some absolute numbers to be implemented later
    indicator_candle_strengths = [3] # [3,5,7,10]
    order_types = ["Limit"]  # or Market
    wait_time_limit_buy_order = [30]  # in minutes
    wait_time_limit_sell_order = [23*30]  # in minutes

    # Creating a dataframe with the parameters of all strategies which will be checked:
    strategy_parameters = [time_candles, candle_durations, momentum_margins, momentum_volumes, desired_profits,
                           stop_losses, units_to_buy, indicator_candle_strengths, order_types,
                           wait_time_limit_buy_order, wait_time_limit_sell_order]
    strategy_parameters_df = pd.DataFrame(itertools.product(*strategy_parameters),
                                          columns=["Time Candle", "Duration", "Momentum Margin", "Momentum Volume", "Desired Profit",
                                                    "Stop Loss", "Units", "Open-close Difference", "Order Type", "Buy Wait Time", "Sell Wait Time"])
    print(strategy_parameters_df)

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

    # The bids/asks and trades dataframes allow us to add a trades-based stop-loss criteria when running
    # a limit order, without having to run a separate search with added time complexity.
    bids_and_trades_df = futures_df[futures_df["trade_type"].isin(("T","B"))]
    asks_and_trades_df = futures_df[futures_df["trade_type"].isin(("T","A"))]

    #bids_df = ConvertToOHLC(bids_df, candle_duration, sampling_unit_of_time)  # use map?
    #asks_df = ConvertToOHLC(asks_df, candle_duration, sampling_unit_of_time)

    # Optional: exporting data, candlestick chart.
    #trades_df_ohlc = ConvertToOHLC(trades_df, candle_duration, 'min')
    #trades_df.to_csv(csv_name_filtered)
    #mpf.plot(trades_df_ohlc, type='candle', volume=True, title='Natural Gas - October')

    # Calculating the profit for every strategy and exporting to CSV:
    start_time = time.time()
    strategy_parameters_df["Total Profit"] = strategy_parameters_df.parallel_apply\
        (ExecuteTradingStrategy, args=(trades_df, asks_df, bids_df, asks_and_trades_df, bids_and_trades_df, csv_year, csv_month), axis=1)
    print(strategy_parameters_df)
    #strategy_parameters_df.to_csv("Strategies Results Limits.csv")
    end_time = time.time()  # currently runs approx. 2700 strategies per hour. Did 18000 in 26082s = around 1.5s per strategy
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    # Packages and importing raw data. (Future - year and month on loop, or determined by GUI.
    import pandas as pd
    import itertools
    import cudf
    import mplfinance as mpf
    #from calendar import monthrange
    from datetime import timedelta
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    import time
    main()