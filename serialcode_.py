import serial

serial_ = 1

serial_port = 'COM8'
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate)


# send to arduino

# data / 1000 = carspeed, data % 1000 = steering_angle
def send_data(steering_angle, car_speed):
    data = str(steering_angle + car_speed * 1000) + '\n'
    ser.write(data.encode())


# real usage example
# global serial_
angle = 0
if (angle > 17):
    angle = 17
elif (angle < -16):
    angle = -16
else:
    angle = angle

if (angle < 1 and angle > -1):
    angle = 0
speed = 50
send_data(angle)
