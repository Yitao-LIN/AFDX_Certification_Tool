# AFDX_Certification_Tool

This tool is part of the class of *Real-Time Networks* during my study at ISAE-SUPAERO.

## Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)

## Introduction
This tool is a software that calculates the end-to-end delay of a message transmitted through an AFDX (Avionic Full-DupleX Switched Ethernet) network.
#### Input and Output (.xml)
The input of the tool is a configuration file that describes the network topology and the traffic parameters of each message. The output is the end-to-end delay of each message, the load of each physical link and the required backlogs in each node.
#### Extended application with Open-Timaeus-Net(TMN)
To be specified.

## Requirements
The tool is written in Python 3.10.11

## Installation
To be specified.

## Usage
To run the tool, execute the following command:
```bash
python3 __base.py <input_file.xml>
```
The output will be saved in a file named `input_file_res_me.xml`.
