total=int(input('Enter no. of bananas at starting:\t'))
distance=int(input('Enter distance you want to cover:\t'))
load_capacity=int(input('Enter max load capacity of your camel:\t'))
lose=0
start=total
for i in range(distance):   # the camel moves 1 km at a time 
    while start>0:
        
        start=start-load_capacity
        if start==1:   # to check if camel has 1b left then it moves forward only
            lose=lose-1 
        lose=lose+2     # as the camel travels 1 mile forward and backward
    
    #at the last destination the camel only moves forward
    
    lose=lose-1
    start=total-lose
    #print(start,lose)
    if start==0:
        break
print("Total bananas left : ",start)  
