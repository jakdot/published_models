"""
ACT-R basic reader. This does not do any parsing, it just sees a word, retrieves it, moves on.

It checks found parameters.


"""

import warnings
import sys

import pandas as pd
import pyactr as actr
import math
from simpy.core import EmptySchedule
import numpy as np
import re
import scipy.stats as stats
import scipy

from pymc3 import Model, Gamma, Normal, HalfNormal, Uniform, find_MAP, Slice, sample, summary, Metropolis, traceplot
from pymc3.backends.base import merge_traces
import theano
import theano.tensor as T
from theano.compile.ops import as_op
from mpi4py import MPI
import matplotlib.pyplot as pp

warnings.filterwarnings(action="ignore", category=UserWarning)

MATERIALS = "materials_final.csv"
#DATA = "small_eyetracking.csv" #testing
DATA = "summed_eyetracking_not_search.csv" #this is a file which ignores sentences with ...

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR

USED_WORDS = 112.5

DECAY = 0.5

def load_texts(lfile=MATERIALS):
    """
    Loads the basic eye tracking text.
    """
    workbook = pd.ExcelFile(lfile)
    worksheet = workbook.parse()
    return worksheet

def load_file(lfile, index_col=None, sep=","):
    """
    Loads file as a list
    """
    csvfile = pd.read_csv(lfile, index_col=index_col, header=0, sep=sep)
    return csvfile

environment = actr.Environment(size=(1366, 768), focus_position=(0, 0))

actr.chunktype("read", "state word")
actr.chunktype("parsing", "top")
actr.chunktype("word", "form cat")

parser = actr.ACTRModel(environment, subsymbolic=True, optimized_learning=False, retrieval_threshold=-50, decay=DECAY, emma_noise=False, emma_landing_site_noise=False) #emma noise - True; to test word skipping

#parser.productionstring(name="find probe", string="""
#        =g>
#        isa     read
#        state   start
#        ?visual_location>
#        state   free
#        buffer  empty
#        ==>
#        =g>
#        isa     goal
#        state   start
#        ?visual_location>
#        attended False
#        +visual_location>
#        isa _visuallocation
#        screen_x closest""") #this rule is used if automatic visual search does not put anything in the buffer

parser.productionstring(name="attend word", string="""
        =g>
        isa     read
        state   start
        =visual_location>
        isa    _visuallocation
        ?visual>
        state   free
        buffer  empty
        ==>
        =g>
        isa     read
        state   start
        +visual>
        isa     _visual
        cmd     move_attention
        screen_pos =visual_location""") #this rule is used at the beginning and when a new line is started

parser.productionstring(name="retrieve word", string="""
        =g>
        isa     read
        state   start
        =visual>
        isa     _visual
        value   =val
        ?retrieval>
        state      free
        ==>
        =g>
        isa     read
        state   parse
        word    =val
        ~visual>
        +retrieval>
        isa         word
        form        =val""")
        
#pd.set_option('mode.chained_assignment',None) #we are setting a value in a copy so we'd get warnings; they don't matter (since we won't use grouped/worksheet later, so we can switch the warning off

def reading(grouped, rank, declchunks):
    """
    Main function, running reading. 
    """

    for name, group in grouped:

        #print(name)

        stimulus = {}
        stimuli = []
        old_utilitites = {}
     
        y_position = 0 #to check if some text appears on the same screen or not

        freq = {}

        positions = {}

        lastwords = {} #storing last words in each line

        for idx, i in enumerate(group.index):
            if group.IA_TOP[i] < y_position:
                stimuli.append(stimulus)
                stimulus = {}
            word = str(group.WORD[i])
            pos = str(group.PART_OF_SPEECH[i])
            stimulus[i] = {'text': word, 'position': (group.IA_CENTER[i], group.IA_TOP[i]), "vis_delay": len(word)}
            positions[(int(group.IA_CENTER[i]), int(group.IA_TOP[i]))] = [idx, 0] #dict to record RTs
            lastwords[str(group.IA_TOP[i])] = str(group.LAST_WORD[i]) #dict for last words in line
            last_position = (group.IA_CENTER[i], group.IA_TOP[i]) #last word in sentence; when reached, stop sim

            #add word into memory
            if not actr.makechunk("", typename="word", form=word, cat=pos) in declchunks:
                freq[word] = group.FREQUENCY[i]
                if freq[word] == 0:
                    freq[word] = 1
                word_freq = freq[word] * USED_WORDS/100 #BNC - 100 millions; estimated use by entering adulthood - 112.5 millions; we have to multiply by 1.125)
                time_interval = SEC_IN_TIME / word_freq
                chunk_times = np.arange(start=-time_interval, stop=-(time_interval*word_freq)-1, step=-time_interval)
                declchunks[actr.makechunk("", typename="word", form=word, cat=pos)] = math.log(np.sum((0-chunk_times) ** (-DECAY)))
            y_position = group.IA_TOP[i]

        for y in lastwords:
            tempstring = "\
        =g>\
        isa     read\
        state   parse\
        ?retrieval>\
        buffer  full\
        =visual_location>\
        isa _visuallocation\
        screen_y =ypos\
        screen_y " + y + "\
        screen_x ~ " + lastwords[y] + "\
        ==>\
        =g>\
        isa     read\
        state   start\
        ?visual_location>\
        attended False\
        +visual_location>\
        isa _visuallocation\
        screen_x lowest\
        screen_y =ypos\
        ~retrieval>"

            parser.productionstring(name="move eyes in the line " + y, string=tempstring)

            tempstring = "\
        =g>\
        isa     read\
        state   parse\
        ?retrieval>\
        buffer  full\
        =visual_location>\
        isa _visuallocation\
        screen_y =ypos\
        screen_y " + y + "\
        screen_x " + lastwords[y] + "\
        ==>\
        =g>\
        isa     read\
        state   start\
        ?visual_location>\
        attended False\
        +visual_location>\
        isa _visuallocation\
        screen_x lowest\
        screen_y onewayclosest\
        ~retrieval>"

            parser.productionstring(name="move eyes to a new line" + y, string=tempstring)

        stimuli.append(stimulus)

        parser.decmems = {}
        parser.set_decmem({x: np.array([]) for x in declchunks})
        
        parser.decmem.activations.update(declchunks)
    
        parser.retrievals = {}
        parser.set_retrieval("retrieval")
    
        parser.visbuffers = {}
    
        environment.current_focus = [stimuli[0][min(stimuli[0])]['position'][0],stimuli[0][min(stimuli[0])]['position'][1]]
    
        parser.visualBuffer("visual", "visual_location", parser.decmem, finst=80)
        parser.goals = {}
        parser.set_goal("g")
        parser.set_goal("g2")
        parser.goals["g"].add(actr.chunkstring(string="""
                isa     read
                state   start"""))
        parser.goals["g2"].add(actr.chunkstring(string="""
                isa     parsing
                top     None"""))
        parser.goals["g2"].delay = 0.2
    
        sim = parser.simulation(realtime=False, trace=False, gui=True, environment_process=environment.environment_process, stimuli=stimuli, triggers='A', times=100)
    
        last_time = 0
    
        cf = tuple(environment.current_focus)

        first_word = True

        generated_list = []

        while True:
                if sim.show_time() > 100:
                    generated_list = [len(group.WORD) - 2] + [0] * (len(group.WORD) - 2) #0 if getting stuck
                    break
                try:
                    sim.step()
                except (EmptySchedule, OverflowError):
                    generated_list = [len(group.WORD) - 2] + [10000] * (len(group.WORD) - 2) #10000 if not finishing or overflowing (too much time)
                    break
                if not positions:
                    break
                if cf[0] != environment.current_focus[0] or cf[1] != environment.current_focus[1]:
                    positions[cf][1] = 1000*(sim.show_time() - last_time) #time in milliseconds
                    last_time = sim.show_time()
                    cf = tuple(environment.current_focus)
                if cf == last_position:
                    break
        if not generated_list:
            ordered_keys = sorted(list(positions), key=lambda x:positions[x][0])
            generated_list = [len(group.WORD)-2] + [positions[x][1] for x in ordered_keys][1:-1] #first and last words ignored
        assert len(generated_list) == len(group.WORD) - 1, "In %s, the length of generated RTs would be %s, expected number of words is %s. This is illegal mismatch" %(name, len(generated_list) + 1, len(group.WORD))
        comm.Send(bytearray(name, 'utf-8'), dest=0, tag=0)
        sent_list = np.array(generated_list, dtype=np.float)
        comm.Send([sent_list, MPI.FLOAT], dest=0, tag=1)

    comm.Send(bytearray('DONE', 'utf-8'), dest=0, tag=0)

    return declchunks

def repeated_reading(grouped, rank):
    """
    For slaves: when receiving True from master, start reading.
    """

    declchunks = {} #storing chunks, moved later to decl. mem

    while True:
        received_list = np.empty(3, dtype=np.float)
        comm.Recv([received_list, MPI.FLOAT], 0, rank)
        if received_list[0] == -1:
            break
        parser.model_parameters["latency_factor"] = received_list[0]
        parser.model_parameters["latency_exponent"] = received_list[1]
        parser.model_parameters["eye_mvt_angle_parameter"] = received_list[2]
        declchunks = reading(grouped, rank, declchunks)

def model(lf, le, emap):

    sent_list = np.array([lf, le, emap], dtype = np.float)

    #get slaves to work
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)
    
    value_dict = {}
    skipped = set()
    #collect from slaves
    while True:
        if len(skipped) == comm.Get_size() - 1:
            break
        for i in range(1, comm.Get_size()):
            if i in skipped:
                continue
            #receive string -- name
            buf = bytearray(100)
            m = comm.Irecv(buf, i, 0)
            status = MPI.Status()
            m.Wait(status)
            length_buffer = status.Get_count(MPI.CHAR)
            name = buf[:length_buffer].decode('utf-8')
            if name == "DONE":
                skipped.add(i)
                continue
            #receive list -- RTs
            received_list = np.empty(200, dtype=np.float)
            comm.Recv([received_list, MPI.FLOAT], i, 1)
            value_dict[name] = received_list[1:int(received_list[0])+1]

    print("SIMULATED SENTENCES", len(value_dict))

    return np.array([val for x in sorted(sorted(value_dict.keys(), key=lambda x:int(x[2:])), key=lambda x:int(x[0])) for val in value_dict[x] ])

worksheet = load_file(MATERIALS)
data = load_file(DATA)
data_sen_id = list(data.SENTENCE_ID)

Y = data.RT

worksheet = worksheet.groupby('SENTENCE_ID', sort=False).filter(lambda x:x['SENTENCE_ID'].iloc[0] in data_sen_id) #delete sentences that are not used later

sentences = [x for x, _ in worksheet.groupby('SENTENCE_ID', sort=False)]

n_sentences = len(sentences)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N_GROUPS = comm.Get_size() - 1 #Groups used for simulation - one less than used cores

if rank == 0: #master
    #following values come from parametersearch done doing Metropolis -- see file parametersearch_MPI.py
    lf = 0.0001
    le = 0.36
    emap = 1.17

    for i in range(14):

        rt = model(lf, le, emap)

        value = str(i) + "LF" +  str(lf) + "LE" + str(le)  + "EMAP" + str(emap)

        data[value] = pd.Series(rt)

        data.to_csv("NONRANDoutput_simulation.csv", sep=",", encoding="utf-8", index=False)

    #stop slaves from work
    sent_list = np.array([-1, -1, -1], dtype = np.float)
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)

else: #slave
    left = round((rank-1)*n_sentences/N_GROUPS)
    right = round(rank*n_sentences/N_GROUPS)
    if right != n_sentences:
        worksheet = worksheet.groupby('SENTENCE_ID', sort=False).filter(lambda x:x['SENTENCE_ID'].iloc[0] in sentences[left:right]).groupby('SENTENCE_ID', sort=False)
    else:
        worksheet = worksheet.groupby('SENTENCE_ID', sort=False).filter(lambda x:x['SENTENCE_ID'].iloc[0] in sentences[left:]).groupby('SENTENCE_ID', sort=False)
    repeated_reading(worksheet, rank)

print(rank, "STOPPED")
MPI.Finalize()
