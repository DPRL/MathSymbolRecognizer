# Author: Awelemdy Orakwue March 10, 2015
#!/usr/bin/python

import sys
import os
import csv
import subprocess
import SocketServer
import SimpleHTTPServer
import base64
import urllib
import httplib
import socket
from xml.dom.minidom import Document, parseString
import cPickle
from symbol_classifier import SymbolClassifier

classifier_filename = "best_full2013_SVMRBF_new.dat"

classifier = ''

class RecognitionServer(SimpleHTTPServer.SimpleHTTPRequestHandler):
    instance_id = 0

    def do_GET(self):
        """
        This will process a request which comes into the server.
        """
        #print classifier
        unquoted_url = urllib.unquote(self.path)
        unquoted_url = unquoted_url.replace("/?segmentList=", "")
        unquoted_url = unquoted_url.replace("&segment=false", "")
        
        dom = parseString(unquoted_url)

        Segments = dom.getElementsByTagName("Segment")
        numSegments = Segments.length
        segmentIDS = []
        classifierPoints = []
        for j in range(numSegments):
            segmentIDS.append(Segments[j].getAttribute("instanceID"))
            points = Segments[j].getAttribute("points")
            url_parts = points.split('|')
           
            translation = Segments[j].getAttribute("translation").split(",") 
            tempPoints = []
            for i in range(len(url_parts)):
                pt = url_parts[i].split(",");
                pt2 = (int(pt[0]) + int(translation[0]), int(pt[1]) + int(translation[1]))
                tempPoints.append(pt2)
            classifierPoints.append(tempPoints)
    
        print classifierPoints 
        results = classifier.classify_points_prob(classifierPoints, 30)
          
        doc = Document()
        root = doc.createElement("RecognitionResults")
        doc.appendChild(root)
        root.setAttribute("instanceIDs", ",".join(segmentIDS))
        
        for k in range(len(results)): 
            r = doc.createElement("Result")  
            s = str(results[k][0]).replace("\\", "")
            sym = dict.get(s)
            if(sym == None):
                sym = s 
            # special case due to CSV file
            if(sym.lower() == "comma"):
                sym = ","
            v = str(results[k][1])
            c = format(float(v), '.35f') 
            r.setAttribute("symbol", sym)
            r.setAttribute("certainty", c)
            root.appendChild(r)      
        
        xml_str = doc.toxml()  
        xml_str = xml_str.replace("amp;","") 
       
        self.send_response(httplib.OK, 'OK')
        self.send_header("Content-length", len(xml_str))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(xml_str)
        
####### UTILITY FUNCTIONS #######

                                   
if __name__ == "__main__":
    usage = "python PenStrokeServer <port number>"
    if(len(sys.argv) < 2):
        print usage
        sys.exit()

    HOST, PORT = "localhost", int(sys.argv[1])
    print("Loading classifier")
    
    global classifer
    global dict
    in_file = open(classifier_filename, 'rb')
    classifier = cPickle.load(in_file)
    in_file.close()

    if not isinstance(classifier, SymbolClassifier):
        print("Invalid classifier file!")
        #return
    print "Reading symbol code from generic_symbol_table.csv"
    dict = {}
    with open('generic_symbol_table.csv', 'rt') as csvfile2:
        reader2 = csv.reader(csvfile2, delimiter=',')
        for row in reader2: 
            dict[row[3]] = row[0]
    #print dict
    print "Starting server."
    
    try:
        server = SocketServer.TCPServer(("", PORT), RecognitionServer)
    except socket.error, e:
        print e
        exit(1)
    #proc = subprocess.Popen(['python','kdtreeServer.py'])    
    print "Serving"
    server.serve_forever()

