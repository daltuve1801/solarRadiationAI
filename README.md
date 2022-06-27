# TelemetryKrill
Krill energy development repository

This repository contains the telemetry developments that works as source for [Harvey](https://harvey.energy/), Krill Energy's monitoring system.

<p align="center">
  <img width="1024" height="400" src="images/krill1.png">
</p>

<p align="center">
  <small> (Harvey's landing page) </small>  
</p>

## Table of Contents

1. [Web services](#webServices)
    1. [Sacolar](#sacolar)
    2. [Studer](#studer)
2. [Krill Boxes](#krillBoxes)
     1. [Raspberry Pi](#raspberryPi)
     2. [Moxa](#Moxa)
     3. [Arduino](#Arduino)
     4. [Krill Box Manager](#krillBoxManager)
3. [Demo](#demo)
6. [Developers](#developers)

# Web services

The web services module is built-in using a EC2 AWS instance, runing an Ubuntu image. This module handles the data pooling through Http request, from provider's API.

<p align="center">
  <img width="400" height="200" src="images/krill5.png">
</p>

<p align="center">
  <small> (Amazon EC2 cloud platform) </small>  
</p>


**[Back to top](#table-of-contents)**

## Sacolar

One of the APIs that is fully integrated with Harvey's datalogin structure is from the provider [Sacolar](https://www.sacolar.com). In this case, the API call is done by a HTTP GET.


**[Back to top](#table-of-contents)**

## Studer

With the provider [Studer](https://www.studer-innotec.com/es/), the treatment is very similar, data pooling is done via HTTP GET.


**[Back to top](#table-of-contents)**

# Krill Boxes

The krill boxes are the datalogin unit provided by Krill Energy, capable of poll data through Modbus RTU, TCP and SNMP; from multiple certified industrial devices.

## Moxa

As main datalog device, Krill Energy can provide a solution-size Moxa industrial computer, selected for the requirements of the solution.

<p align="center">
  <img width="600" height="300" src="images/krill2.png">
</p>

<p align="center">
  <small> (Moxa industrial computers) </small>  
</p>


**[Back to top](#table-of-contents)**

## RaspberryPi

For smaller-size solutions, specially residential solutions, we count with the Raspberry Pi 4. It handles the communication over wife and Ethernet.

<p align="center">
  <img width="600" height="500" src="images/krill3.jpg">
</p>

<p align="center">
  <small> (Raspberry PI 4) </small>  
</p>

[Sacolar](https://www.sacolar.com)


**[Back to top](#table-of-contents)**

## Arduino

An industrial Arduino is used as remote measuring station.

<p align="center">
  <img width="600" height="600" src="images/krill4.png">
</p>

<p align="center">
  <small> (Arduino industrial PLC) </small>  
</p>

[Arduino](https://www.studer-innotec.com/es/).


**[Back to top](#table-of-contents)**

## Krill box manager

Using an EC2 instance the Krill box manager is a module that review the connection status of each krill box connected to an installation.

**[Back to top](#table-of-contents)**

# Demo

Finally, the demo module provides the data structure necessary to test visual features in the front end development

**[Back to top](#table-of-contents)**

# Developers

* **[David Altuve](https://github.com/Daltuve18)**
* **[Vincezo D'Argento](https://github.com/vincdargento)**
* **[Jaime Villegas](https://github.com/JMVI)**


**[Back to top](#table-of-contents)**
