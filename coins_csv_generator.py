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
Date  = df['time']
Coins = df['coin']
Volumes = df['volume']



with open(r'bnb_price.csv', 'w') as f:

    str_init="Date,Open,High,Low,Close,Volume,OpenInt"+"\n"
    f.write(str_init)

    str_final=""
    for i in range(len(BuyPrices)):
        zero_val=0
        str_row = str(Date[i])+","+str(SellPrices[i])+","+str(SellPrices[i])+","+str(SellPrices[i])+","+str(SellPrices[i])+","+str(Volumes[i])+","+str(zero_val)
        str_final=str_final+"\n"+str_row

    f.write(str_final)