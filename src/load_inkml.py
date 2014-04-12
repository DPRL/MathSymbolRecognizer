from traceInfo import *
from mathSymbol import *
import xml.etree.ElementTree as ET

#=====================================================================
#  Load an INKML file using the mathSymbol and traceInfo classes
#  to represent the data
#
#  Created by:
#      - Kenny Davila (Oct, 2012)
#  Modified By:
#      - Kenny Davila (Oct 24, 2013)
#      - Kenny Davila (Nov 25, 2013)
#         - Added ID to symbol
#      - Kenny Davila (Jan 17, 2014)
#         - Fixed ID cases for CHROME 2013 data containing colon
#      - Kenny Davila (Feb 3, 2014)
#         - Fixed cases where symbol loaded has no traces
#      - Kenny Davila (Apr 11, 2014)
#         - Fixed cases where symbol loaded has no traces
#         - Added junk symbol compatibility
#
#=====================================================================


#debug flags
debug_raw = False
debug_added = False
debug_smoothing = False
debug_normalization = False

#the current XML namespace prefix...
INKML_NAMESPACE = '{http://www.w3.org/2003/InkML}'

def load_inkml_traces(file_name):
    #first load the tree...
    tree = ET.parse(file_name)
    root = tree.getroot()    
    
    #extract all the traces first...
    traces_objects = {}    
    for trace in root.findall(INKML_NAMESPACE + 'trace'):
        #text contains all points as string, parse them and put them
        #into a list of tuples...
        points_s = trace.text.split(",");
        points_f = []
        for p_s in points_s:
            #split again...
            coords_s = p_s.split()
            #add...
            points_f.append( (float(coords_s[0]), float(coords_s[1])) )
    
        trace_id = int(trace.attrib['id'])
        
        #now create the element
        object_trace = TraceInfo(trace_id, points_f )
        
        #add to the diccionary...
        traces_objects[trace_id] = object_trace
        
        #apply general trace pre processing...
        
        #1) first step of pre processing: Remove duplicated points        
        object_trace.removeDuplicatedPoints()

        if debug_raw:
            #output raw data
            file = open('out_raw_' + trace.attrib['id'] + '.txt', 'w')
            file.write( str(object_trace) )        
            file.close()        
            
        #Add points to the trace...
        object_trace.addMissingPoints()

        if debug_added:
            #output raw data
            file = open('out_added_' + trace.attrib["id"] + '.txt', 'w')
            file.write( str(object_trace) )        
            file.close()
                        
        #Apply smoothing to the trace...
        object_trace.applySmoothing()

        #it should not ... but .....
        if object_trace.hasDuplicatedPoints():
            #...remove them! ....
            object_trace.removeDuplicatedPoints()
            
        if debug_smoothing:
            #output data after smoothing
            file = open('out_smoothed_' + trace.attrib["id"] + '.txt', 'w')
            file.write( str(object_trace) )        
            file.close()
        
        
    
    return root, traces_objects

def extract_symbols( root, traces_objects, truth_available ):
    #put all the traces together with their corresponding symbols...
    #first, find the root of the trace groups...
    groups_root = root.find(INKML_NAMESPACE + 'traceGroup')            
    trace_groups = groups_root.findall(INKML_NAMESPACE + 'traceGroup')
    
    symbols = []
    avg_width = 0.0
    avg_height = 0.0
    
    for group in trace_groups:        
        if truth_available:
            #search for class label...
            symbol_class = group.find(INKML_NAMESPACE + 'annotation').text

            #search for id attribute...
            symbol_id = 0
            for id_att_name in group.attrib:
                if id_att_name[-2:] == "id":
                    try:
                        symbol_id = int(group.attrib[id_att_name])
                    except:
                        #could not convert to int, try spliting...
                        symbol_id = int( group.attrib[id_att_name].split(":")[0] )
                        
        else:
            #unknown
            symbol_class = '{Unknown}'
            symbol_id = 0
    
        #link with corresponding traces...
        group_traces = group.findall(INKML_NAMESPACE + 'traceView')
        symbol_list = []
        for trace in group_traces:
            object_trace = traces_objects[int(trace.attrib["traceDataRef"])]
            symbol_list.append(object_trace)
            
        #create the math symbol...
        try:
            new_symbol = MathSymbol(symbol_id, symbol_list, symbol_class)
        except Exception as e:
            print("Failed to load symbol!!")
            print(e)
            #skip this symbol...
            continue

        #capture statistics of relative size....
        symMinX, symMaxX, symMinY, symMaxY = new_symbol.original_box
        #...add...
        avg_width  += (symMaxX - symMinX)
        avg_height += (symMaxY - symMinY)
        
        #now normalize size and locations for traces in current symbol
        new_symbol.normalize()
        
        if debug_normalization:
            #output data after relocated
            for trace in group_traces:
                object_trace = traces_objects[int(trace.attrib["traceDataRef"])]
            
                file = open('out_reloc_' + trace.attrib["traceDataRef"] + '.txt', 'w')
                file.write( str(object_trace) )        
                file.close()
                
        symbols.append(new_symbol)

    if len(trace_groups) > 0:
        avg_width /= len(trace_groups)
        avg_height /= len(trace_groups)
    
        for s in symbols:
            s.setSizeRatio(avg_width, avg_height)
        
    return symbols

def load_inkml(file_name, truth_available):
    
    root, traces_objects = load_inkml_traces(file_name)
    
    symbols = extract_symbols(root, traces_objects, truth_available)
    
    return symbols

def extract_junk_symbol(traces_objects, junk_class_name):
    symbol_id = 0
    symbol_class = junk_class_name

    symbol_list = []
    for trace_id in traces_objects:
        symbol_list.append(traces_objects[trace_id])

    #create the math symbol...
    try:
        new_symbol = MathSymbol(symbol_id, symbol_list, symbol_class)
    except Exception as e:
        print("Failed to load symbol!!")
        print(e)
        return None

    #now normalize size and locations for traces in current symbol
    new_symbol.normalize()

    return new_symbol

def load_junk_inkml(file_name, junk_class_name):

    root, traces_objects = load_inkml_traces(file_name)

    symbols = [extract_junk_symbol(traces_objects, junk_class_name)]

    return symbols
