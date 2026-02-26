import sqlite3
con = sqlite3.connect('/tmp/trader_optuna.db')
cur = con.cursor()
cur.execute('SELECT COUNT(*) FROM trials')
print('trials:', cur.fetchone()[0])
con.close()
