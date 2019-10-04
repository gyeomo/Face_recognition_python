import os
import time
import threading

i = 0
def time_():
	global i
	while(True):
		print(i)
		time.sleep(1)
		i+=1
		

t1 = threading.Thread(target=time_)
t1.daemon = True
t1.start()
while(True):
	os.system('cls')
	s = input("입력 : ")
	print(s)