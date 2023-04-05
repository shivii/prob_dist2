# -*- coding: utf-8 -*-
import time

def print_with_time(*argument):
    start_time = time.localtime( time.time() )  
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", start_time)
    print(start_time, argument)
