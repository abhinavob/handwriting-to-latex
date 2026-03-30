from torch import topk
from decoder import model,attention_wt as decoder,atwt
from encoder import source_len
from math import log


temp_const=0 ## change to some very samll non zero val
decoder_output=decoder()
beam_width=1 #change now its in greedy
eos=0 #change the definiions here for whats end of line
token_score={'tokens':[],'score':0}
completed = []

'''left to define
def max_len():
    passs
def max_threshhold_gap: #for when max beam and other beams have high gap
    pass
def repetion_penalty: #doubtuful
    pass
def min_len(): # debatble check by accuracy
    pass
def early_stop: ## the prob for some beam is really high
    pass
def divereseity(): ## if many beams have lot of common words penalise
    pass
'''
def lenp(length,k=0,a=0): #define before test
    return ((k+length)/(k+1))**a

def conv_penalty(conv,convpenalty=0,sum=0): #define before test should not be zero
    for i in range(source_len):
        sum+=log(min(conv[i],1.0)+temp_const)  # define temp_constane samll so if conv[i]=o still no issues 
    return sum*convpenalty; 
         
def normalised_beam_score(beam_score,length,conv):
    return beam_score/lenp(length)+conv_penalty(conv)

def beam_score(decoder_output,beam,coverage):
    
    topb,indices=topk(decoder_output,beam_width)
    for score, tok_id in zip(topb,indices):
        new_token = beam['tokens'] + [tok_id.item()]  
        new_score  = beam['score']  + score.item()

        new_coverage = coverage.copy()
        attn = atwt(tok_id.item())
        for i in range(source_len):
            new_coverage[i] += attn[i]

        if(tok_id.item()==eos):
            normalised_score=normalised_beam_score(new_score,len(new_token),new_coverage)
            completed.append({'tokens': new_token, 'score': normalised_score})
            return;
        new_decoder_output=decoder(new_token)
        new_beam={'tokens':new_token,'score':new_score}
        beam_score(new_decoder_output,new_beam,new_coverage)

def output(): ## the main function 
    conv = [0.0] * source_len
    beam_score(decoder_output,token_score,conv)
    best=max(completed,key=lambda x: x['score'])
    return best['tokens']

print(output()) ## temp remove before test