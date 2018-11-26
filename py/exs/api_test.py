import requests

origin_data = {
    'applyid': 1111,
    'dealerid': 1111,
    'dealer_name': '主干新建44',
    'carid': 1111,
    'car_name': '日产 玛驰 2010款 1.5 手动 XE易型版',
    'car_cityid': 1111,
    'car_city_name': '哈尔滨',
    'loan_recheck_time': 1542179072,
    'interest_start': 1536225440,
    'debug': 1,
}

data = []
for x in range(1, 3):
	tmp = {k: v + x if isinstance(v, int) else v + str(x) for k, v in origin_data.items() if k != 'debug'}
	tmp['debug'] = 1
	data.append(tmp)

# for params in data:
# 	r = requests.post("http://develop.allinapi.ceshi.youxinjinrong.com/api/order/create", data=params)
# 	print(r.text)