from labquest import LabQuest

lq = LabQuest()

lq.open()
lq.select_sensors(ch1='lq_sensor')    
lq.start(100)

for x in range(1000):
    ch1_measurement = lq.read('ch1')
    if ch1_measurement == None: 
        break 
    print(ch1_measurement)

lq.stop()
lq.close()