# Tag ID
int32 id

# Tag size
float64 size

# Indicator of if this tag is static
bool static_tag

# Pose in the camera frame, obtained from homography transform
geometry_msgs/Pose pose

geometry_msgs/Point center
geometry_msgs/Point[4] corners
