# -*- coding: utf-8 -*-
from datetime import datetime

def print_with_time(*argument):
    #start_time = time.localtime( time.time() )  
    start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    print(start_time, argument)
