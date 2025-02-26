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

""" Node
    The Node class is used to handle any node if the network
    It's an abstract class
"""
class Node:
    def __init__(self, name):
        self.name = name
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)

""" Station
    The Station class is used to handle stations
"""
class Station(Node):
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = parseCapacities(capacityStr=capacity)
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)
    def isSwitch(self):
        return False

""" Switch
    The Switch class is used to handle switches
"""
class Switch(Node):
    def __init__(self, name, latency, capacity):
        self.name = name
        self.latency = latency
        self.capacity = parseCapacities(capacityStr=capacity)
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)
    def isSwitch(self):
        return True

""" Edge
    The Edge class is used to handle edges
"""
class Edge:
    def __init__(self, name, frm, to, frmPort, toPort, capacity):
        self.name = name
        self.capacity = parseCapacities(capacityStr=capacity)
        self.frm = frm
        self.to = to
        self.frmPort = frmPort
        self.toPort = toPort   
        self.load_direct  = 0
        self.load_reverse = 0
        
""" Target
    The Target class is used to handle targets
"""
class Target:
    def __init__(self, flow, to):
        self.flow = flow
        self.to = to
        self.path = []
        self.path_link = []
        self.delay = 0

""" Flow
    The Flow class is used to handle flows
"""
class Flow:
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
    
    def loadCalculus(self):
        """ Load calculus
            
            The aim of the load calculus is to verify the network stability condition.
            Return:
                is_overflow : boolean, True if the network is overloaded, False otherwise.
            
            **Assumption:**
                - The links are full-duplex and the load may be different on each direction.
        """
        is_overflow = False
        for edge in edges:
            for flow in flows:
                flow_is_computed = False
                for target in flow.targets:
                    if flow_is_computed:
                        break
                    for pair_index in range(len(target.path)-1):
                        if edge.frm == target.path[pair_index] and edge.to == target.path[pair_index+1]:
                            edge.load_direct += (flow.payload + flow.overhead)/flow.period * 8
                            flow_is_computed = True
                        elif edge.to == target.path[pair_index] and edge.frm == target.path[pair_index+1]:
                            edge.load_reverse += (flow.payload + flow.overhead)/flow.period * 8
                            flow_is_computed = True
                        if edge.load_direct > edge.capacity or edge.load_reverse > edge.capacity:
                            is_overflow = True
                        if flow_is_computed:
                            break
        return is_overflow

    def serviceCurve(self):
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
            capacity, latency = 0, 0 # initialize the service curve
            if node.isSwitch():
                latency = node.latency
                capacity = node.capacity
            else:
                latency = 0
                capacity = node.capacity
            node.serviceCurve = (capacity, latency)     # set the service curve

    def arrivalCurveES(self, node):
        rate, burst = 0, 0
        for flow in self.flows:
            if flow.source == node.name:
                packet_size = (flow.payload + flow.overhead)*8
                rate += packet_size/flow.period
                burst += packet_size
        return rate, burst
    
    def arrivalCurveSW(self, target, node):
        rate, burst = 0, 0
        target_link = target.path_link[target.path.index(node.name)]    # the output link of this target
        pre_nodes = {}  # the previous nodes of this node
        for flow in self.flows:
            for t in flow.targets:
                if (node.name in t.path and
                    t.path_link[t.path.index(node.name)] == target_link):
                    pre_node = self.getNodeByName(t.path[t.path.index(node.name)-1])
                    if pre_node not in pre_nodes.keys():
                        pre_nodes.update({pre_node: [t]})
                    else:
                        pre_nodes[pre_node].append(t)
        for pre_node in pre_nodes.keys():
            # prevent multiple calculation of multicasted flow
            if target in pre_nodes[pre_node]: 
                # if the target comes from the previous node 
                pre_rate, pre_burst = self.arrivalCurve(target, pre_node)
            else:
                # if the target doesn't come from the previous node
                pre_rate, pre_burst = self.arrivalCurve(pre_nodes[pre_node][0], pre_node)
            rate += pre_rate
            burst += pre_burst
        return rate, burst
    
    def arrivalCurve(self, target, node):
        if node.name == target.to:
            return None
        
        if node.isSwitch():
            rate, burst = self.arrivalCurveSW(target, node)
        else:
            rate, burst = self.arrivalCurveES(node)

        # apply service curve
        node.arrivalCurve = (rate, burst)
        delay = self.delayBound(node)

        if delay is not None:
            target.delay += delay
            burst += rate * delay
            node.arrivalCurve = (rate, burst)
            return rate, burst
        else:
            print("Service curve not defined for node: " + node.name)
            return None

    def delayBound(self, node: Node):
        if node.serviceCurve is not None:
            return node.arrivalCurve[1]/node.serviceCurve[0] + node.serviceCurve[1]
        return None

    def getNodeByName(self, name:str)->Node:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def resetCalculation(self):
        for flow in self.flows:
            for target in flow.targets:
                target.delay = 0

    def getNetworkDelay(self):
        self.serviceCurve()
        for flow in self.flows:
            for target in flow.targets:
                self.resetCalculation()
                self.arrivalCurve(target, self.getNodeByName(target.path[-2]))
                print("Delay " + target.flow.source +" to target " + target.to + " = " + str(target.delay*1E6) + " us")
                flows[self.flows.index(flow)].targets[flow.targets.index(target)] = deepcopy(target)
        

################################################################@
""" Main program """
################################################################@

if __name__ == '__main__':
    if len(sys.argv)>=2:
        xmlFile=sys.argv[1]
    else:
        xmlFile="./Samples/AFDX.xml"
    
    parseNetwork(xmlFile)
    traceNetwork()
    nc = NetworkCalculus(nodes, flows, edges)
    is_overflow = nc.loadCalculus()
    if not is_overflow:
        nc.getNetworkDelay()
    else:
        print("The network is overloaded.")
    createResultsFile(xmlFile)