uint8 OWNER_NONE=0
uint8 OWNER_MANUAL=1
uint8 OWNER_NAVIGATION=2
uint8 OWNER_STAIR_POLICY=3

std_msgs/Header header
uint8 owner
string lease_id
geometry_msgs/Twist twist
