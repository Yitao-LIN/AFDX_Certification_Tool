################################################################@
"""

This file is a starting base for xml network parsing.
It is provided in the scope of the AFDX Project (WoPANets Extension)
The aim of such a file it to simplify the python coding, so that students focus on Network Calculus topics.

You have to update and complete this file in order to fit all the projects requirements.
Particularly, you need to complete the Station, Switch, Edge, Flow and Target classes.

"""
################################################################@

import xml.etree.ElementTree as ET
import os.path
import sys
from copy import deepcopy
from math import ceil

################################################################@
""" Global data """
################################################################@
nodes = [] # the nodes
edges = [] # the edges
flows = [] # the flows

################################################################@
""" Local classes """
################################################################@

class Node:
    """ Node

        The Node class is used to handle any node if the network
        It's an abstract class
    """
    def __init__(self, name):
        self.name = name
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)

class Station(Node):
    """ Station

        The Station class is used to handle stations
    """
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = parseCapacities(capacityStr=capacity)
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)
    def isSwitch(self):
        return False

class Switch(Node):
    """ Switch

        The Switch class is used to handle switches
    """ 
    def __init__(self, name, latency, capacity):
        self.name = name
        self.latency = latency
        self.capacity = parseCapacities(capacityStr=capacity)
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)
    def isSwitch(self):
        return True

class Edge:
    """ Edge

        The Edge class is used to handle edges
    """
    def __init__(self, name, frm, to, frmPort, toPort, capacity):
        self.name = name
        self.capacity = parseCapacities(capacityStr=capacity)
        self.frm = frm
        self.to = to
        self.frmPort = frmPort
        self.toPort = toPort   
        self.load_direct  = 0
        self.load_reverse = 0
        
class Target:
    """ Target

        The Target class is used to handle targets
    """
    def __init__(self, flow, to):
        self.flow = flow
        self.to = to
        self.path = []
        self.path_link = []
        self.delay = 0

class Flow:
    """ Flow

        The Flow class is used to handle flows
    """
    def __init__(self, name, source, payload, overhead, period):
        self.name = name
        self.source = source
        self.payload = payload
        self.overhead = overhead
        self.period = period
        self.targets = []

################################################################@
""" Local methods """
################################################################@

def parseStations(root):
    """ parseStations

        Method to parse stations
        Args:
            root : the xml main root
    """
    for station in root.findall('station'):
        nodes.append (Station(station.get('name'), station.get('transmission-capacity')))

def parseSwitches(root):
    """ parseSwitches

        Method to parse switches
        Args:
            root : the xml main root
    """
    for sw in root.findall('switch'):
        nodes.append (Switch(sw.get('name'),float(sw.get('tech-latency'))*1e-6, sw.get('transmission-capacity')))

def parseEdges(root):
    """ parseEdges

        Method to parse edges
        Args:
            root : the xml main root
    """
    for link in root.findall('link'):
        edges.append (Edge(link.get('name'),link.get('from'),link.get('to'), link.get('fromPort'), link.get('toPort'), link.get('transmission-capacity')))

def parseFlows(root):
    """ parseFlows

        Method to parse flows
        Args:
            root : the xml main root
    """
    for sw in root.findall('flow'):
        flow = Flow (sw.get('name'),sw.get('source'),float(sw.get('max-payload')),67,float(sw.get('period'))*1e-3)
        flows.append (flow)
        for tg in sw.findall('target'): # for each multicasted target
            target = Target(flow,tg.get('name'))
            flow.targets.append(target)
            target.path.append(flow.source)
            for pt in tg.findall('path'):
                target.path.append(pt.get('node'))
                if len(target.path) > 1:
                    target.path_link.append(getEdge(target.path[-2], target.path[-1]))

def getEdge(frm, to):
    ''' getEdge
    
        get the edge name between two nodes
        Args:
            frm : the source node
            to  : the destination node
        Returns:
            The edge name is returned with the direction
    '''
    for edge in edges:
        if edge.frm == frm and edge.to == to:
            return edge.name + " Direct"
        elif edge.to == frm and edge.frm == to:
            return edge.name + " Reverse"
    return None

def parseCapacities(capacityStr: str) -> int:
    """ parseCapacities

        Method to automatically detect the unit of the capacity and convert it to bits/s
        Args:
            capacityStr : the capacity string
        Returns:
            The capacity in bits/s
    """
    if 'Gbps' in capacityStr:
        return int(capacityStr.replace('Gbps', '')) * 1e9
    elif 'Mbps' in capacityStr:
        return int(capacityStr.replace('Mbps', '')) * 1e6
    elif 'Kbps' in capacityStr:
        return int(capacityStr.replace('Kbps', '')) * 1e3
    else:
        return int(capacityStr)

def parseNetwork(xmlFile):
    """ parseNetwork

        Method to parse the whole network
        Args:
            xmlFile : the path to the xml file
    """
    if os.path.isfile(xmlFile):
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        parseStations(root)
        parseSwitches(root)
        parseEdges(root)
        parseFlows(root)
    else:
        print("File not found: "+xmlFile)

def traceNetwork():
    """ traceNetwork

        Method to trace the network to the console
    """
    print("Stations:")
    for node in nodes:
        if not node.isSwitch():
            print ("\t" + node.name)
            
    print("\nSwitches:")
    for node in nodes:
        if node.isSwitch():
            print ("\t" + node.name)
            
    print("\nEdges:")
    for edge in edges:
        print ("\t" + edge.name + ": " + edge.frm + " port: " + edge.frmPort + " => " + edge.to + " port: " + edge.toPort)
    
    print("\nFlows:")
    for flow in flows:
        print ("\t" + flow.name + ": " + flow.source + " (L=" + str(flow.payload) +", p=" + str(flow.period) + ")")
        for target in flow.targets:
            print ("\t\tPath to Target=" + target.to + ":\n")
            for node in target.path:
                print ("\t\t\t" + node)
                print ("\t\t\t| " + target.path_link[target.path.index(node)]) if target.path.index(node) < len(target.path)-1 else print("\n")

def createResultsFile (xmlFile):
    """ createResultsFile

        Method to create a result file
        Args:
            xmlFile : the path to the xml (output) file
    """
    posDot = xmlFile.rfind('.')
    if not (posDot == -1):
        resFile = xmlFile[0:posDot]+'_res_me.xml'
    else:
        resFile = xmlFile+'_res.xml'
    res = open(resFile,"w")
    res.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    res.write('<results>\n')
    res.write('\t<delays>\n')
    for flow in flows:
        res.write('\t\t<flow name="' + flow.name + '">\n')
        for target in flow.targets:
            res.write('\t\t\t<target name="' + target.to + '" value="'+str(ceil(target.delay*1E6))+'" />\n')
        res.write('\t\t</flow>\n')
    res.write('\t</delays>\n')

    # Write load results
    res.write('\t<loads>\n')
    for edge in edges:
        res.write('\t\t<edge name="' + edge.name + '">\n')
        res.write('\t\t\t<usage type="direct"  value="'+str(int(edge.load_direct))+'"  percent="'+str(round(edge.load_direct/edge.capacity*100, 1))+'%"/>\n')
        res.write('\t\t\t<usage type="reverse" value="'+str(int(edge.load_reverse))+'" percent="'+str(round(edge.load_reverse/edge.capacity*100, 1))+'%"/>\n')
        res.write('\t\t</edge>\n')
    res.write('\t</loads>\n')

    res.write('</results>\n')
    res.close()
    file2output(resFile)
    
def file2output (file):
    """ file2output
    
        Method to print a file to standard ouput
        Args:
            file : the path to the xml (input) file
    """
    hFile = open(file, "r")
    for line in hFile:
        print(line.rstrip())

################################################################@
""" Network analysis methods"""
################################################################@

class NetworkCalculus:
    def __init__(self, nodes, flows, edges):
        self.nodes = deepcopy(nodes)
        self.flows = deepcopy(flows)
        self.edges = deepcopy(edges)
        self.node_map = {node.name: node for node in self.nodes}  # Quick node lookup

    
    def computeNetworkLoads(self):
        """ Load calculus
            
            The aim of the load calculus is to verify the network stability condition.
            Return:
                is_overflow : boolean, True if the network is overloaded, False otherwise.
            
            **Assumption:**
                - The links are full-duplex and the load may be different on each direction.
        """
        is_overflow = False # overflow flag
        for edge in edges:  # for each edge
            for flow in flows:  # check all the flows in the network
                flow_is_computed = False    # initialize the multicasted flag
                for target in flow.targets: # for all the targets in the flow
                    if flow_is_computed:    # if one of the targets in this flow is already computed
                        break               # skip this flow
                    for pair_index in range(len(target.path)-1):    # check the path of the target
                        # if th path includes this edge (direct)
                        if edge.frm == target.path[pair_index] and edge.to == target.path[pair_index+1]:
                            edge.load_direct += (flow.payload + flow.overhead)/flow.period * 8
                            flow_is_computed = True
                        # if th path includes this edge (reverse)
                        elif edge.to == target.path[pair_index] and edge.frm == target.path[pair_index+1]:
                            edge.load_reverse += (flow.payload + flow.overhead)/flow.period * 8
                            flow_is_computed = True
                        # if the edge is overloaded
                        if edge.load_direct > edge.capacity or edge.load_reverse > edge.capacity:
                            is_overflow = True
                        # if the flow is already found in this edge, skip the other targets
                        if flow_is_computed:
                            break
        return is_overflow

    def computeNetworkDelays(self):
        """ Network delay calculus

            Main method to compute all network delays.
        """
        # calculate service curves for all nodes once
        self._computeAllServiceCurve()

        # calculate delays for all flow targets
        for flow in self.flows:
            for target in flow.targets:
                # reset target delay
                target.delay = 0
                # compute delays from Src to Dst
                for node in target.path:
                    if node == target.to:
                        break   # stop when reaching the target
                    else:
                        source_node = self.node_map[node]
                        rate, burst = self._computeNodeDelay(target, source_node)
                        if rate is not None and burst is not None:
                            source_node.arrivalCurve = (rate, burst)
                        else:
                            break
                # update the original flow data
                flows[self.flows.index(flow)].targets[flow.targets.index(target)] = deepcopy(target)
                print(f"Delay {target.flow.source} to target {target.to} = {target.delay*1E6:.2f} us")
    
    def _computeAllServiceCurve(self):
        """ Service curve calculus
        
        - The aim of the service curve (*S.C.*) calculus is to give the service curve
        of each network node based on *Rate-Latency* model. The service curve relay on
        the model of the network node only.

        - *S.C.* is defined by the tuple (c,l) where: c is the rate (capacity) and l is the latency.

        - **Assumption:** 
            - FIFO policy within the End-Systems and AFDX switches
            - Null technological latency within End-Systems and Switches
            - Cut-Through switching technique at the input port of switches
    """
        for node in self.nodes:
            # check if the node is a switch
            latency = node.latency if node.isSwitch() else 0
            node.serviceCurve = (node.capacity, latency)

    def _computeNodeDelay(self, target, node):
        """ Node delay calculus

            Recursive method to compute delay through a node
        """
        if node.name == target.to:
            return None, None
        
        # Theorem 1: Input arrival curve of a node
        if node.isSwitch():
            rate, burst = self._computeSwitchArrivalCurve(target, node)
        else:
            rate, burst = self._computeEndSystemArrivalCurve(node)
        
        # Delay
        node.arrivalCurve = (rate, burst)
        delay = self._computeDelayBound(node)

        # Theorem 2: Ouput arrival curve of a node
        if delay is not None:
            # add delay to target and update burst
            target.delay += delay
            burst += rate * delay
            node.arrivalCurve = (rate, burst)
            return rate, burst
        else:
            print("Service curve not defined for node: " + node.name)
            return None, None
    
    def _computeEndSystemArrivalCurve(self, node):
        """ End-System arrival curve calculus

            Method to compute the arrival curve of an End-System
        """
        rate, burst = 0, 0
        for flow in self.flows:
            if flow.source == node.name:
                packet_size = (flow.payload + flow.overhead) * 8
                rate += packet_size / flow.period
                burst += packet_size
        return rate, burst
    
    def _computeSwitchArrivalCurve(self, target, node):
        """ Switch arrival curve calculus

            Method to compute the arrival curve of a Switch
        """
        rate, burst = 0, 0
        # get current link for this node in the target's path
        node_index = target.path.index(node.name)
        current_link = target.path_link[node_index] # identify output port
        # get previous link for this node
        input_port_dict = {}
        for f in self.flows:
            for t in f.targets:
                if (node.name in t.path and
                    t.path_link[t.path.index(node.name)] == current_link):
                    pre_link = t.path_link[t.path.index(node.name) - 1]
                    if pre_link not in input_port_dict:
                        input_port_dict[pre_link] = [t]
                    else:
                        input_port_dict[pre_link].append(t)
        
        # iterate over all the input ports
        for input_port, t_list in input_port_dict.items():
            # get all the flows that pass through this switch and shares the same output link (i.e. same output port)
            for t in t_list:
                pre_node = self.node_map[t.path[t.path.index(node.name) - 1]]
                pre_rate, pre_burst = pre_node.arrivalCurve if pre_node.arrivalCurve is not None else (None, None)
                if pre_rate is not None:
                    rate += pre_rate
                    burst += pre_burst
                else:
                    pre_rate, pre_burst = self._computeNodeDelay(t, pre_node)
                    rate += pre_rate
                    burst += pre_burst
                break


        
        return rate, burst
    
    def _computeDelayBound(self, node: Node):
        if node.serviceCurve is None:
            return None

        arrival_rate, arrival_burst = node.arrivalCurve
        service_rate, service_latency = node.serviceCurve
        
        return arrival_burst / service_rate + service_latency
        

################################################################@
""" Main program """
################################################################@

if len(sys.argv)>=2:
    xmlFile=sys.argv[1]
else:
    xmlFile="./Samples/AFDX.xml"

parseNetwork(xmlFile)
traceNetwork()
nc = NetworkCalculus(nodes, flows, edges)
is_overflow = nc.computeNetworkLoads()
if not is_overflow:
    nc.computeNetworkDelays()
else:
    print("The network is overloaded.")
createResultsFile(xmlFile)