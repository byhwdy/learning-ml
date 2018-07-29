import requests
import re
from requests.packages import urllib3
import logging


# headers = {
#     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
# }

# r = requests.get('https://www.zhihu.com/explore', headers=headers)
# pattern = re.compile('explore-feed.*?question_link.*?>(.*?)</a>', re.S)
# titles = re.findall(pattern, r.text)
# print(titles)

# r = requests.get('https://github.com/favicon.ico')
# with open('favicon.ico', 'wb') as f:
# 	f.write(r.content)

# r = requests.get('http://www.jianshu.com')
# print(type(r.status_code), r.status_code)
# print(type(r.headers), r.headers)
# print(type(r.cookies), r.cookies)
# print(type(r.url), r.url)
# print(type(r.history), r.history)

# r = requests.get('http://www.jianshu.com')
# exit() if not r.status_code == requests.codes.ok else print('Request Successfully')

# files = {'file': open('favicon.ico', 'rb')}
# r = requests.post('http://httpbin.org/post', files=files)
# print(r.text)

# r = requests.get('https://www.baidu.com')
# print(r.cookies)
# for key, value in r.cookies.items():
# 	print(key + '=>' + value)

# headers = {
# 	'Cookie': 'q_c1=3e2c36e7877f4941a6fa6bfd03132e90|1515220431000|1515220431000; _zap=e1620947-0da0-49e1-92de-4b9f4a038bd7; d_c0="AGDj4c7P9gyPTqATm_QeCFQ8ggHVbe6DMOk=|1515501686"; __DAYU_PP=m7fFB2vZf7FU6NQnvnJbffffffff8a3217a39c41; __utmz=51854390.1528714104.1.1.utmcsr=zhihu.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmv=51854390.100-1|2=registration_date=20140223=1^3=entry_date=20140223=1; q_c1=3e2c36e7877f4941a6fa6bfd03132e90|1530267189000|1515220431000; l_cap_id="YTIwNmFlYTQzZDk1NDM2MzhjMDJhNjcyN2VlMmYyYjE=|1531376671|8e994a32b36e60d231eead3fefb1acbc96513b5f"; r_cap_id="ODEyN2JhNmE3MTQ5NDE5YjkyNTg3MGM4ZDVmYzEzYmQ=|1531376671|3ba28c7ac60cc150450cf274ff29fc130743c048"; cap_id="M2RlYTdiZTlmMWUyNDU3Njk2NTEwMjU4ZGE5Mjk5MWI=|1531376671|62be44f4db2c578945dc85da1780b151a179cfe6"; capsion_ticket="2|1:0|10:1531384330|14:capsion_ticket|44:M2YzMmQ4MmM2YTQwNDJiNThkZTJlMjEyNjIyYWJjY2Q=|3e6186b56b034b04f1f0969b458c22b229429058a1d4461991331de1222c574f"; z_c0="2|1:0|10:1531384338|4:z_c0|92:Mi4xNE5JNUFBQUFBQUFBWU9QaHpzXzJEQ1lBQUFCZ0FsVk5FbUEwWEFDU0IyNFBtSnVmOWlvQ0U3WWFodEZsYjgzRml3|9c98d5688d3c64fb55c91f79d39355675dac7d170890a14f33b38f1652ba09c9"; _xsrf=3ad6791b-644b-45ee-8a2c-bfde4d7bd13a; __utmc=51854390; tgw_l7_route=5bcc9ffea0388b69e77c21c0b42555fe; __utma=51854390.899529586.1528714104.1531796033.1531798081.3; __utmb=51854390.0.10.1531798081',
# 	'Host': 'www.zhihu.com',
# 	'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
# }

# r = requests.get('https://www.zhihu.com', headers=headers)
# print(r.text)

# s = requests.Session()
# s.get('http://httpbin.org/cookies/set/number/123456')
# r = s.get('http://httpbin.org/cookies')
# print(r.text)

# urllib3.disable_warnings()
logging.captureWarnings(True)
response = requests.get('https://www.12306.cn', verify=False)
print(response.status_code)