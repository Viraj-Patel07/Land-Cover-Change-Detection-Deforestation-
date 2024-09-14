# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:39:18 2023

@author: velan
"""

#file = open(r".\text\file1.txt", "a")
#written = file.write("\n this is a test line 16\n this is a test line 17")
#print(written)
#file.close()    

#with open(r".\text\file1.txt") as f:  
#    print(f.readlines())
    
#with open(r"./text/text.txt") as t:
 #   list1 = t.readlines()
#    print(list1)
#print(list1[0][-3::])    
#for l in list1:
 #   if int(l[-3::])%2 == 0:
#    #    with open(r"./text/text2.txt", "a") as f2:
   #         ap2 = f2.write(l)
  #  elif int(l[-3::])%2 != 0:
 #       with open(r"./text/text1.txt", "a") as f1:
#            ap1 = f1.write(l)
with open("text/Rplot01.jpeg", "r") as file:
    # file.seek(3,1)
    # file.seek(15,1)
    str = file.read(3)#.decode("utf-8")
    print(str)