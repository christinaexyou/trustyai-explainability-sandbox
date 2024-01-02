from IPython.display import HTML
import matplotlib.pyplot as plt

def get_color(score):
    if score > 0:
        g = int(128*score) + 127
        b = 128 - int(64*score)
        r = 128 - int(64*score)
    else:
        g = 128 + int(64*score)
        b = 128 + int(64*score)
        r = int(-128*score) + 127
    return r,g,b

def visualize_token_scores(tokens, scores):
    html_text = ""
    for i, tok in enumerate(tokens):
        r, g, b = get_color(scores[i])
        html_text += " <span style='color:rgb(%d,%d,%d)'>%s</span>" % \
                     (r, g, b, tok)

    return HTML(html_text)

def plot_topk_scores(tokens, scores, top_k):
     # order tokens by descending scores and select top k
    sorted_tokens_scores = sorted(list(zip(tokens, scores)),
                                  key=lambda x: x[1],
                                  reverse=True)[:top_k]

    tokens, scores = zip(*sorted_tokens_scores)

    plt.figure(figsize=(21,3))
    xvals = [ x + str(i) for i,x in enumerate(tokens)]
    colors =  [ (0,0,1, c) for c in (scores) ]

    plt.tick_params(axis='both', which='minor', labelsize=29)
    p = plt.bar(xvals, scores, color=colors, linewidth=1 )
    p = plt.xticks(ticks=[i for i in range(len(tokens))], labels=tokens, fontsize=12,rotation=90)
