# DDoS_ML

# DDoS Mitigation for Using SDN and Machine Learning Approach in The Traffic of Modern Video Streaming Server
###### tags: SDN, Machine_Learning, Video_Streaming, s-Flow, wireshark 

##### *This is a technical documentation of paper*

## Introduction 
In this experiment we setup our own network using Software-Defined Network (SDN) and then we collected the packet information using wireshark. Then, we categorized the packet information into 4 types : *normal, icmp echo flood, tcp xmas flood, and udp flood*. 

We build our SDN network using a Raspberry Pi 3 converted to SDN switch. Besides we also implement sFlow-RT to do Packet Sampling from SDN switch to the controller. In this experiment we use ONOS as our SDN controller. 

This documentation only focus on 3 types of DDoS attack : 
    1. ICMP Echo Flood
    2. TCP XMas Flood
    3. UDP Flood

## System Design
In this experiment we setup two different network. The difference is only in connection to the internet. In the first network architecture, we made it closed so there is no connection to the internet. In the second network architecture, we establish the connection to the internet from our SDN switch. 
#### Scheme to collect the dataset and model testing
![](https://i.imgur.com/3NHq98T.png)
![](https://i.imgur.com/J1kQfrK.png)

## Workflow
[1. Setup the Raspberry Pi to make it works as SDN switch](https://hackmd.io/Jhgpr5eYTCKYnDW_j29ejg)
[2. Setup the ONOS controller and connect it to our SDN switch]()
3. Setup sFlow-RT to do packet sampling
4. Setup the wireshark in the controller to capture packet information and export those information into .csv file
5. Analyze the packet information using Machine Learning method

###### tags: `project` `sdn` 


