uint8 OWNER_NONE=0
uint8 OWNER_MANUAL=1
uint8 OWNER_NAVIGATION=2
uint8 OWNER_STAIR_POLICY=3

uint8 BACKEND_NONE=0
uint8 BACKEND_SIMULATION=1
uint8 BACKEND_SPORT=2
uint8 BACKEND_POLICY=3

std_msgs/Header header
uint8 active_owner
uint8 active_backend
bool ready
bool stopped
bool emergency_stop
float32 command_age_seconds
string reason
