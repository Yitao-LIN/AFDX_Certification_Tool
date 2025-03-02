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
        self.ports = {}            # dict  the ports of the node
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)

class Station(Node):
    """ Station

        The Station class is used to handle stations
    """
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = parseCapacities(capacityStr=capacity)
        self.ports = {}            # dict  the ports of the node
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
        self.ports = {}            # dict  the ports of the node
        self.arrivalCurve = None   # tuple (Rate, Burst)
        self.serviceCurve = None   # tuple (Capacity, Latency)
    def isSwitch(self):
        return True

class Port:
    """ Port
    
        The Port class is used to handle ports
    """
    def __init__(self, name):
        self.name = name
        self.edge = None
        self.isReverse = False
        self.load = 0

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
        self.arrival_curve_on_path = {}

################################################################@
""" Local methods """
################################################################@

def parseStations(root: ET.Element):
    """ parseStations

        Method to parse stations
        Args:
            root : the xml main root
    """
    for station in root.findall('station'):
        nodes.append (Station(station.get('name'), station.get('transmission-capacity')))

def parseSwitches(root: ET.Element):
    """ parseSwitches

        Method to parse switches
        Args:
            root : the xml main root
    """
    for sw in root.findall('switch'):
        nodes.append (Switch(sw.get('name'),float(sw.get('tech-latency'))*1e-6, sw.get('transmission-capacity')))

def parseEdges(root: ET.Element):
    """ parseEdges

        Method to parse edges
        Args:
            root : the xml main root
    """
    for link in root.findall('link'):
        edges.append (Edge(link.get('name'),link.get('from'),link.get('to'), link.get('fromPort'), link.get('toPort'), link.get('transmission-capacity')))

def parseFlows(root: ET.Element):
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
                    edge_name = getEdge(target.path[-2], target.path[-1])
                    target.path_link.append(edge_name)
                    flow.arrival_curve_on_path[edge_name] = (None, None)

def parsePort():
    """ getPort
    
        get all the ports of every node
    """
    for node in nodes:
        for edge in edges:
            if edge.frm == node.name:
                node.ports[edge.frmPort] = Port(edge.frmPort)
                node.ports[edge.frmPort].edge = edge
                node.ports[edge.frmPort].isReverse = False
            elif edge.to == node.name:
                node.ports[edge.toPort] = Port(edge.toPort)
                node.ports[edge.toPort].edge = edge
                node.ports[edge.toPort].isReverse = True

def getEdge(frm: str, to: str) -> str:
    ''' getEdge
    
        get the edge name between two nodes and identify the direction
        Args:
            frm : str, the source node
            to  : str, the destination node
        Returns:
            str, the edge name is returned with the direction
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

def parseNetwork(xmlFile: str):
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
        parsePort()
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

def createResultsFile (xmlFile: str):
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
            res.write('\t\t\t<target name="' + target.to + '" value="'+format(target.delay*1E6, '.1f')+'" />\n')
        res.write('\t\t</flow>\n')
    res.write('\t</delays>\n')

    # Write load results
    res.write('\t<loads>\n')
    for edge in edges:
        res.write('\t\t<edge name="' + edge.frm + " => " + edge.to + '">\n')
        res.write('\t\t\t<usage type="direct"  percent="'+ str(round(edge.load_direct/edge.capacity*100, 1))+'%" value="'+str(float(edge.load_direct))+'"/>\n')
        res.write('\t\t\t<usage type="reverse" percent="'+ str(round(edge.load_reverse/edge.capacity*100, 1))+'%" value="'+str(float(edge.load_reverse))+'"/>\n')
        res.write('\t\t</edge>\n')
    res.write('\t</loads>\n')

    res.write('</results>\n')
    res.close()
    file2output(resFile)
    
def file2output (file: str):
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
    """ NetworkCalculus
    
        This class is created to compute:
            - Network loads
            - Network delays
    """
    def __init__(self,
                 nodes: list,
                 flows: list,
                 edges: list) -> None:
        """ Class constructor

            Args:
                nodes : list, the list of nodes
                flows : list, the list of flows
                edges : list, the list of edges
        """
        # deep copy global data to avoid any modification
        self.nodes = deepcopy(nodes)
        self.flows = deepcopy(flows)
        self.edges = deepcopy(edges)

        # create quick lookup maps
        self.node_map = {node.name: node for node in self.nodes}
        self.edge_map = {edge.name: edge for edge in self.edges}
        self.flow_map = {flow.name: flow for flow in self.flows}
        self.port_map = {edge.name: (edge.frmPort, edge.toPort) for edge in self.edges}

        # Store paths for quicker lookup
        self.flow_paths = {}
        for flow in self.flows:
            self.flow_paths[flow.name] = {}
            for target in flow.targets:
                self.flow_paths[flow.name][target.to] = target.path

    def computeNetworkLoads(self) -> bool:
        """ Load calculus
            
            The aim of the load calculus is to verify the network stability condition.
            Return:
                boolean, True if the network is overloaded, False otherwise.
            
            **Assumption:**
                - The links are full-duplex and the load may be different on each direction.
        """
        # initialize is_overflow flag
        is_overflow = False

        # pre-compute flow-to-path mapping for efficiency
        flow_on_edges = {edge.name: {"direct": [], "reverse": []} for edge in self.edges}

        # build flow-to-edge mapping in one pass
        for flow in self.flows:
            for target in flow.targets:
                for i in range(len(target.path)-1):
                    src, dist = target.path[i], target.path[i+1]
                    # find matching edge
                    for edge in self.edges:
                        if edge.frm == src and edge.to == dist:
                            flow_on_edges[edge.name]["direct"].append(flow)
                            break
                        elif edge.frm == dist and edge.to == src:
                            flow_on_edges[edge.name]["reverse"].append(flow)
                            break
        
        # compute loads in one pass
        for edge in self.edges:
            # direct flows
            edge.load_direct = sum(
                (flow.payload + flow.overhead) * 8 / flow.period 
                for flow in set(flow_on_edges[edge.name]["direct"])
            )
            # reverse flows
            edge.load_reverse = sum(
                (flow.payload + flow.overhead) * 8 / flow.period 
                for flow in set(flow_on_edges[edge.name]["reverse"])
            )

            # deep copy the edge to the original list
            edges[self.edges.index(edge)] = deepcopy(edge)

            # check if the network is overloaded
            if edge.load_direct > edge.capacity or edge.load_reverse > edge.capacity:
                is_overflow = True
        
        return is_overflow

    def computeNetworkDelays(self) -> None:
        """ Network delay calculus

            Main method to compute all network delays.
            
            **Computation process:**
            1. Compute service curves for all nodes since the service curve is based on the node model only
            2. Go through all flows and targets to compute delays
                - Compute delays for each target along its path
                - Update the original flow data
        """
        # calculate service curves for all nodes once
        self._computeAllServiceCurve()

        # calculate delays for all flow targets
        for flow in self.flows:
            for target in flow.targets:
                # reset target delay
                target.delay = 0

                # compute delays along the target's path
                for node in target.path:
                    # stop when reaching the target
                    if node == target.to:
                        break              
                    else:
                        # get the node object
                        source_node = self.node_map[node]

                        # get the input and output port numbers
                        output_port = self._getPortNumber(target.path_link[target.path.index(node)])[0]

                        # compute delay through the node
                        self._computeNodeDelay(target, source_node, output_port)

                # update the original flow data
                flows[self.flows.index(flow)].targets[flow.targets.index(target)] = deepcopy(target)
    
    def _computeAllServiceCurve(self) -> None:
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

    def _computeNodeDelay(self,
                          target: Target,
                          node: Node,
                          output_port: str) -> tuple:
        """ Node delay calculus

            Method to compute the delay of a node.
            Args:
                target : Target, the target of interest
                node : Node, the target is passing through this node
                output_port : str, the output port number
            Returns:
                (rate, burst) : (float, float), the output arrival curve parameters

            **Computation process:**
            1. *THEOREM 1:* Compute the input arrival curve of the node
            2. *DELAY COMPUTATION:* Compute the delay of the node
            3. *THEOREM 2:* Compute the output arrival curve of the node
        """        
        ################################################################@
        """ THEOREM 1: Input arrival curve of a node """
        ################################################################@
        if node.isSwitch():
            rate, burst = self._computeArrivalCurveSW(node, output_port)
        else:
            rate, burst = self._computeArrivalCurveES(node)
        
        ################################################################@
        """ DELAY COMPUTATION & SERVICE CURVE APPLICATION """
        ################################################################@
        delay = self._computeDelayBound(node, burst)

        ################################################################@
        """ THEOREM 2: Output arrival curve of a node """
        ################################################################@
        if delay is not None:
            # add delay to target
            target.delay += delay

            # update the output link arrival curve
            output_link_index = target.path.index(node.name)
            output_link = target.path_link[output_link_index]

            if output_link_index > 0:
                # if there is a previous link, use previous link's arrival curve 
                pre_link = target.path_link[output_link_index-1]
                rate, burst = target.flow.arrival_curve_on_path[pre_link]
            else:
                rate = (target.flow.payload + target.flow.overhead) * 8 / target.flow.period
                burst = (target.flow.payload + target.flow.overhead) * 8
            
            # update the arrival curve of the output link (only burst changed - Theorem 2)
            burst += rate * delay

            # Store updated arrival curve for this link
            target.flow.arrival_curve_on_path[output_link] = (rate, burst)
            
            return rate, burst
        
        else:
            print("Service curve not defined for node: " + node.name)
            return None, None
    
    def _computeArrivalCurveES(self,
                               node: Node) -> tuple:
        """ End-System arrival curve calculus

            Method to compute the arrival curve of an End-System
        """
        rate, burst = 0, 0

        for f in flows:
            if f.source == node.name:
                rate += (f.payload + f.overhead) * 8 / f.period
                burst += (f.payload + f.overhead) * 8

        return rate, burst
    
    def _computeArrivalCurveSW(self,
                               node: Node,
                               output_port: str) -> tuple:
        """ Switch arrival curve calculus

            Calculate arrival curve for a Switch by aggregating flows.

            Args:
                node : Node, the switch node
                output_port : str, the output port number
            Returns:
                (rate, burst) : (float, float), the input arrival curve for this switch
        """
        # dict to track flow edges by input port
        input_port_flows = {}

        # identify all flows that use this output port
        for f in self.flows:
            for t in f.targets:
                # find this node in t's path
                node_index = t.path.index(node.name) if node.name in t.path else -1
                
                # check if t passes through this node
                if node_index >= 0 and node_index < len(t.path) - 1:
                    # get the output link and port
                    output_link = t.path_link[node_index]
                    t_output_port = self._getPortNumber(output_link)[0]

                    # if using the same output port as our target
                    if t_output_port == output_port:
                        # get the input link and port if it's not a source node
                        if node_index > 0:
                            pre_link = t.path_link[node_index - 1]
                            t_input_port = self._getPortNumber(pre_link)[1]

                            if t_input_port not in input_port_flows:
                                input_port_flows[t_input_port] = []

                            input_port_flows[t_input_port].append((f, t, pre_link))
                        break # skip the other targets since the flow is already included

        # calculate total arrival curve by summing per-port arrival curves
        total_rate, total_burst = 0, 0

        for _, flow_info in input_port_flows.items():
            for f, t, link in flow_info:
                # get arrival curve parameters for this flow on the previous link
                if link in f.arrival_curve_on_path and f.arrival_curve_on_path[link] != (None, None):
                    pre_rate, pre_burst = f.arrival_curve_on_path[link]
                else:
                    # compute recursively if not available
                    pre_node_index = t.path_link.index(link)
                    pre_node = self.node_map[t.path[pre_node_index]]
                    output_port = self._getPortNumber(link)[0]
                    pre_rate, pre_burst = self._computeNodeDelay(t, pre_node, output_port)

                total_rate += pre_rate
                total_burst += pre_burst
        
        return total_rate, total_burst
    
    def _computeDelayBound(self,
                           node: Node,
                           arrival_burst: float) -> float:
        """
        Calculate delay bound based on arrival burst and service curve.

        Args:
            node : Node, the node
            arrival_burst : float, the input arrival burst
        Returns:
            float, the delay bound
        
        **Remark:** Using rate-latency service curve: delay = burst/rate + latency
        """

        if node.serviceCurve is None:
            return None

        service_rate, service_latency = node.serviceCurve
        
        return arrival_burst / service_rate + service_latency
    
    def _getPortNumber(self,
                       link_name: str) -> tuple:
        """ Get port number

            Method to get the port number by the link name
            Args:
                link_name : str, the link name
            Returns:
                tuple, (from port, to port)
        """
        if link_name is not None:
            if "Direct" in link_name:
                return self.port_map[link_name.replace(" Direct", "")][0], self.port_map[link_name.replace(" Direct", "")][1]  # input port, output port
            elif "Reverse" in link_name:
                return self.port_map[link_name.replace(" Reverse", "")][1], self.port_map[link_name.replace(" Reverse", "")][0]  # input port, output port
        else:
            return None, None

################################################################@
""" Main program """
################################################################@

if __name__ == "__main__":
    if len(sys.argv)>=2:
        xmlFile=sys.argv[1]
    else:
        xmlFile="./Samples/AFDX.xml"

    parseNetwork(xmlFile)
    traceNetwork()
    nc = NetworkCalculus(nodes, flows, edges)
    if not nc.computeNetworkLoads():
        nc.computeNetworkDelays()
    else:
        print("The network is overloaded.")
    
    createResultsFile(xmlFile)