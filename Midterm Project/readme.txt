Platform: Jupyter

Required files:
	listed_company_2018.csv # 2018年上市公司股票
	listed_company_2017.csv # 2017年上市公司股票
	listed_company_2016.csv # 2016年上市公司股票
	counter_company_2017.csv # 2017年上櫃公司股票
	counter_company_2018.csv # 2018年上櫃公司股票
	counter_company_2016.csv # 2016年上櫃公司股票
	news.csv # 2016~2018年新聞文章 (encoding=big5)
	bbs-utf8.csv # 2016~2018年BBS文章 (encoding=utf8)
	forum-utf8.csv # 2016~2018年論壇文章 (encoding=utf8)

Parameter Lists: (For Training)
	pre_day_t # 預測的前n天
	test_cut_time # 文章集切割時間點, 小於=training set, 大於=testing set
	sigma # 股票漲跌幅度 (CP(D+n)-CP(D))/CP(D+n), CP=收盤價
	company_list # 測試個股
Parameter Lists: (For Testing)
	pre_day # 預測的前n天
	tcnt_ma_f_len # 文章統計快速平均長度
	tcnt_ma_s_len # 文章統計慢速平均長度

Description:
	1. 過濾文章
	2. 找到訓練漲跌文章集
	3. 特徵選擇
	4. 訓練模型測試 kfold=3
	5. 測試事件發生與預測漲跌並出手判斷