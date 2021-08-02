# import sys
# import os.path
#
# orig = sys.stdout
# with open(os.path.join("dir", "output.txt"), "wb") as f:
#     sys.stdout = f
#     try:
#         execfile("run.py", {})
#     finally:
#         sys.stdout = orig
import urllib.request
import http
import time
base = "http://192.168.10.100/"

def transfer(my_url):   #use to send and receive data
    time.sleep(20)
    try:
        n = urllib.request.urlopen(base + my_url).read()
        n = n.decode("utf-8")
        return n

    except http.client.HTTPException as e:
        return e


two = transfer("two")
one = transfer("one")
print(one,two)
_ = transfer("45")   #Send this data