import torch
from math import log

def max_len(length,a=150): #define the max len at which you recive max beifit (Changed to 150 for memory safety)
    if length<a:
        return True
    else: 
        return False

def get_repetition_penalty(tokens, current_id, penalty=-0.7):
    if len(tokens) > 1 and tokens[-1] == current_id and tokens[-2]==current_id:
        return penalty
    return 0.0

'''left to define
def min_len(): # debatble check by accuracy
    pass
def early_stop: ## the prob for some beam is really high
    pass
def divereseity(): ## if many beams have lot of common words penalise
    pass
'''

def lenp(length,k=6,a=0.5): #define before test
    return ((k+length)/(k+1))**a

def conv_penalty(conv, convpenalty=0.7, penalty_sum=0):
    from math import log
    temp_const = 1e-6
    for i in range(len(conv)):
        penalty_sum += log(min(conv[i], 1.0) + temp_const)
    return penalty_sum * convpenalty

def bracket_penalty(new_token, open_bracket_id, close_bracket_id, penalty=-10): ##change as per nedeed
    sum_open_bracket=0
    sum_close_bracket=0
    if open_bracket_id == -1 or close_bracket_id == -1: return 0 # Safety check
    
    for i in new_token:
        if(i==open_bracket_id):
            sum_open_bracket+=1
        elif(i==close_bracket_id):
            sum_close_bracket+=1
        if(sum_open_bracket-sum_close_bracket<0):
            return penalty;
    if(sum_open_bracket!=sum_close_bracket):
        return penalty;
    else:
        return 0

def normalised_beam_score(beam_score,length,conv,new_token, open_bracket_id, close_bracket_id):
    return beam_score/lenp(length)+conv_penalty(conv)+bracket_penalty(new_token, open_bracket_id, close_bracket_id)


def beam_decode(decoder, encoder, image, tokenizer, device):
    
    # 1. Dynamically fetch IDs
    eos = tokenizer.token_to_id.get("<end>")
    start_id = tokenizer.token_to_id.get("<start>")
    open_bracket_id = tokenizer.token_to_id.get("{", -1)
    close_bracket_id = tokenizer.token_to_id.get("}", -1)
    
    # 2. Person 1's Original Hyperparameters
    min_score_token=-11
    temp_const=1e-6 
    beam_width=9
    min_length=4 
    
    token_score={'tokens':[start_id],'score':0.0}
    global completed
    completed = []
    
    # 3. Extract visual features exactly ONCE
    with torch.no_grad():
        enc_out, src_lengths = encoder(image)

    src_len = enc_out.shape[1]  # 49 for ResNet34 7x7
    active_beams = [(token_score['tokens'], token_score['score'], [0.0] * src_len)]
    
    while active_beams:
        all_candidates = [] 
        
        for b_tokens, b_score, b_conv in active_beams:
            tgt_tensor = torch.tensor([b_tokens], dtype=torch.long).to(device)
            
            with torch.no_grad():
                
                logits, attn = decoder.decode_step(tgt_tensor, enc_out, src_lengths)
                
                # logits is shape [1, vocab_size]
                next_token_logits = logits[0]
                log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                
                topb, indices = torch.topk(log_probs, beam_width)
                best_score = topb[0].item()

            attn_weights = attn[0].tolist()  # (src_len,)

            for score, tok_id in zip(topb, indices):
                t_id = tok_id.item()
            
                if (best_score - score.item()) > 3.2:
                    break
                
                if score.item() < min_score_token:
                    continue

                new_token = b_tokens + [t_id]
                
                if new_token.count(close_bracket_id) > new_token.count(open_bracket_id):
                    continue

                rep_penalty = get_repetition_penalty(b_tokens, t_id)
                new_score = b_score + score.item() + rep_penalty

                
                new_coverage = b_conv.copy()
                for i in range(src_len):
                    new_coverage[i] += attn_weights[i]

                if not(max_len(len(new_token))):
                    continue

            
                if(t_id == eos and len(new_token) > min_length):
                    norm_score = normalised_beam_score(new_score, len(new_token), new_coverage, new_token, open_bracket_id, close_bracket_id)
                    completed.append({'tokens': new_token, 'score': norm_score})
                    continue
                
                all_candidates.append((new_token, new_score, new_coverage))

        all_candidates.sort(key=lambda x: x[1] / lenp(len(x[0])), reverse=True)
        active_beams = all_candidates[:beam_width]
        
        if not active_beams or len(active_beams[0][0]) > 150:
            break

    # 4. Final output selection
    if not completed:
        if active_beams:
            completed = [{'tokens': b[0], 'score': normalised_beam_score(b[1], len(b[0]), b[2], b[0], open_bracket_id, close_bracket_id)} for b in active_beams]
        else:
            return ""

    best = max(completed, key=lambda x: x['score'])
    best_tokens = best['tokens']
    
    if best_tokens[0] == start_id: best_tokens = best_tokens[1:]
    if best_tokens and best_tokens[-1] == eos: best_tokens = best_tokens[:-1]
        
    return tokenizer.decode(best_tokens)
