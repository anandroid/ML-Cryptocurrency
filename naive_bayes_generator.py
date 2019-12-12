import MySQLdb
import pandas as pd

conn = MySQLdb.connect(host="remotemysql.com", user="6txKRsiwk3", passwd="nPoqT54q3m", db="6txKRsiwk3")
cursor = conn.cursor()

sql = "select * from fetcherhistory where coin='BNB' and market='BINANCE'"

df = pd.read_sql_query(sql, conn)
# disconnect from server
conn.close()




BuyPrices = df['buy_for']
SellPrices = df['sell_for']
Dates  = df['time']
Coins = df['coin']
Volumes = df['volume']


with open(r'bnb_naive_label.csv', 'w') as f:

    str_init="BuyFor,SellFor,Volume,PreviousBuyPrice,PreviousSellPrice,Label"+"\n"
    f.write(str_init)

    for loop_index in range(10):

        prev_buy_price=-1
        prev_sell_price=-1
        str_final=""
        first_buy_price=100
        for i in range(len(BuyPrices)):

            selldifference = ((SellPrices[i]-prev_sell_price)/prev_sell_price)*100

            buydifference = ((BuyPrices[i]-prev_buy_price)/prev_buy_price)*100

            firstselldifference = ((SellPrices[i]-first_buy_price)/first_buy_price)*100



            if prev_buy_price == -1 :
                label = "BUY"
            elif   selldifference > 5:
                label = "SELL"
                first_buy_price = 100
            elif firstselldifference > 0:
                label = "SELL"
            elif buydifference < 0:
                label = "BUY"
                if first_buy_price == 100:
                  first_buy_price= BuyPrices[i]
            else:
                label = "HOLD"



            str_row = str(BuyPrices[i])+","+str(SellPrices[i])+","+str(Volumes[i])+","+str(prev_buy_price)+","+str(prev_sell_price)+","+label
            str_final=str_final+"\n"+str_row

            prev_buy_price = BuyPrices[i]
            prev_sell_price = SellPrices[i]

        f.write(str_final)

print("Done")