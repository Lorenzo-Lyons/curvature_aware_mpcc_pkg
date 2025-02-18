import rospy
from dynamic_reconfigure.client import Client


rospy.init_node("optuna_node")  # Initialize the node

# Create a client to communicate with the reconfigurable node
client = Client("/dart_simulator_node", timeout=5)  # Change "your_node_name" to the correct node name


config = client.get_configuration()
print(config)  # Print all available parameters

client.update_configuration({"reset_state_x": 0})