import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

maxquery = 5e2

def attack_targeted(model, x0, y0 , pair, target, alpha = 0.1, beta = 0.001, iterations = 300,stop_g=0.5,filter_g=0.8):
    """ Attack the original image and return adversarial example of target t
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        t: target
    """
    device=x0.device
    if (model.predict(x0) != y0):
        # print("Fail to classify the image. No need to attack.")
        raise Exception("Fail to classify the image. No need to attack.")

    if(model.predict(pair)!=target):
        # print("Fail to classify pair")
        raise Exception('Fail to classify pair')

    # STEP I: find initial direction (theta, g_theta)

    # num_samples = 100
    # best_theta, g_theta = None, float('inf')
    query_count = 0

    # print("Searching for the initial direction on %d samples: " % (num_samples))
    # timestart = time.time()
    # samples = set(random.sample(range(len(train_dataset)), num_samples))
    # for i, (xi, yi) in enumerate(train_dataset):
    #     if i not in samples:
    #         continue
    #     query_count += 1
    #     xi=xi[0].to(device)
    #     yi=yi[0].to(device)
    #     if model.predict(xi) == target:
    #         theta = xi - x0
    #         initial_lbd = torch.norm(theta)
    #         theta = theta/torch.norm(theta)
    #         lbd, count = fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd)
    #         query_count += count
    #         if lbd < g_theta:
    #             best_theta, g_theta = theta, lbd
    #             print("--------> Found distortion %.4f" % g_theta)

    # timeend = time.time()
    # print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))


    # STEP II: seach for optimal
    timestart = time.time()
    theta = pair - x0
    initial_lbd = torch.norm(theta)
    theta = theta/torch.norm(theta)
    g_theta,count = fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd)
    best_theta=theta
    query_count += count

    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    opt_count = 0

    for i in range(iterations):
        gradient = torch.zeros(theta.size()).to(device)
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor)
            u = u/torch.norm(u)
            u = u.to(device)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))

        min_theta = theta
        min_g2 = g2
    
        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2
        
        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = model.predict(x0 + g_theta*best_theta)
    timeend = time.time()
    
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
    count=query_count+opt_count
    return x0 + g_theta*best_theta,count

def fine_grained_binary_search_local_targeted(model, x0, y0, t, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd
   
    if model.predict(x0+lbd*theta) != t:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) != t and nquery < maxquery:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 100: 
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while model.predict(x0+lbd_lo*theta) == t and nquery < maxquery:
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol and nquery < maxquery:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search_targeted(model, x0, y0, t, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd

    while model.predict(x0 + lbd*theta) != t and nquery < maxquery:
        lbd *= 1.05
        nquery += 1
        if lbd > 100: 
            return float('inf'), nquery

    num_intervals = 100

    lambdas = np.linspace(0.0, lbd.to('cpu'), num_intervals)[1:]
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        nquery += 1
        if model.predict(x0 + lbd*theta) == t:
            lbd_hi = lbd
            lbd_hi_index = i
            break

    lbd_lo = lambdas[lbd_hi_index - 1]

    while (lbd_hi - lbd_lo) > 1e-7 and nquery < maxquery:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    return lbd_hi, nquery



def attack_untargeted(model, x0, y0,pair, alpha = 0.2, beta = 0.001, iterations = 300, stop_g=0.5,filter_g=0.8):
    """ Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
    """
    device=x0.device
    if (model.predict(x0) != y0):
        # print("Fail to classify the image. No need to attack.")
        raise Exception("Fail to classify the image. No need to attack.")

    if(model.predict(pair)==y0):
        # print("Fail to classify pair")
        raise Exception('Fail to classify pair')

    # num_samples = 1000
    # best_theta, g_theta = None, float('inf')
    query_count = 0

    # print("Searching for the initial direction on %d samples: " % (num_samples))
    # timestart = time.time()
    # samples = set(random.sample(range(len(train_dataset)), num_samples))
    # for i, (xi, yi) in enumerate(train_dataset):
    #     if i not in samples:
    #         continue
    #     query_count += 1
    #     xi=xi[0].to(device)
    #     yi=yi[0].to(device)
    #     if model.predict(xi) != y0:
    #         theta = xi - x0
    #         initial_lbd = torch.norm(theta)
    #         theta = theta/torch.norm(theta)
    #         lbd, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
    #         query_count += count
    #         if lbd < g_theta:
    #             best_theta, g_theta = theta, lbd
    #             print("--------> Found distortion %.4f" % g_theta)

    # timeend = time.time()
    # print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))

    
    
    timestart = time.time()

    theta = pair - x0
    initial_lbd = torch.norm(theta)
    theta = theta/torch.norm(theta)
    g_theta,count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd,float('inf'))
    best_theta=theta
    query_count += count

    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    torch.manual_seed(0)
    opt_count = 0
    stopping = 0.01
    prev_obj = 100000
    for i in range(iterations):
        gradient = torch.zeros(theta.size()).to(device)
        q = 10
        min_g1 = float('inf')
        # print(f'--------------------stage 1(iter={i})-------------------')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor)
            u = u/torch.norm(u)
            u = u.to(device)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/100)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
            if g2 > prev_obj-stopping:
                break
            prev_obj = g2

        min_theta = theta
        min_g2 = g2
    
        # print(f'--------------------stage 2(iter={i})-------------------')
        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/100)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
            else:
                break

        # print(f'--------------------stage 3(iter={i})-------------------')
        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/100)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2
        
        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break
        
        if g_theta< 0.5:
            break

    target = model.predict(x0 + g_theta*best_theta)
    timeend = time.time()
    if g_theta<0.8:
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        count=query_count+opt_count
        return x0 + g_theta*best_theta,count
    else:
        print("\nAdversarial Example Found Failed: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        count=query_count+opt_count
        return x0,count

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd
     
    if model.predict(x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) == y0 and nquery < maxquery:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            # print('binary_search_local 1: query={}'.format(nquery))
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while_cnt=0
        while model.predict(x0+lbd_lo*theta) != y0 and nquery < maxquery:
            lbd_lo = lbd_lo*0.99
            nquery += 1
            while_cnt+=1
            if while_cnt>100000:
                break
            # print('binary_search_local 2: query={}'.format(nquery))
    

    while (lbd_hi - lbd_lo) > tol and nquery < maxquery:
        # print(lbd_hi-lbd_lo,tol)
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        # print('binary_search_local 3: query={}'.format(nquery))
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best: 
        if model.predict(x0+current_best*theta) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd
    
    ## original version
    #lbd = initial_lbd
    #while model.predict(x0 + lbd*theta) == y0:
    #    lbd *= 2
    #    nquery += 1
    #    if lbd > 100:
    #        return float('inf'), nquery
    
    #num_intervals = 100

    # lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    # lbd_hi = lbd
    # lbd_hi_index = 0
    # for i, lbd in enumerate(lambdas):
    #     nquery += 1
    #     if model.predict(x0 + lbd*theta) != y0:
    #         lbd_hi = lbd
    #         lbd_hi_index = i
    #         break

    # lbd_lo = lambdas[lbd_hi_index - 1]
    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5 and nquery < maxquery:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        # print('binary_search: query={}'.format(nquery))
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    print(f'fine_grained_binary_search nquery = {nquery}')
    return lbd_hi, nquery


def optim(opt , model, images, labels, pairs,targets=None):
    device=opt.device
    ret=[]
    count=0
    iterations = opt.optim_iter
    if targets is None:
        for (image,label,pair) in zip(images,labels,pairs):
            preturbed_image,count_tmp=attack_untargeted(model, image, label,pair,iterations=iterations,alpha=opt.optim_alpha,beta=opt.optim_beta,stop_g=opt.stop_g,filter_g=opt.filter_g)
            count+=count_tmp
            shape=preturbed_image.shape
            ret.append(preturbed_image.view(1,shape[0],shape[1],shape[2]))
    else:
        for (image,label,pair,target) in zip(images,labels,pairs,targets):
            preturbed_image,count_tmp=attack_targeted(model, image, label,pair,target,iterations=iterations,alpha=opt.optim_alpha,beta=opt.optim_beta,stop_g=opt.stop_g,filter_g=opt.filter_g)
            count+=count_tmp
            shape=preturbed_image.shape
            ret.append(preturbed_image.view(1,shape[0],shape[1],shape[2]))
    ret=torch.cat(ret,0)
    return ret,count

class inner_model():
    def __init__(self,model):
        self.model=model

    def predict(self,x0):
        # with torch.no_grad():
        shape=x0.shape
        x0=x0.view(1,shape[0],shape[1],shape[2])
        ret = self.model(x0).max(1)[1]
        return ret[0]

class Optim_attacker:
    def __init__(self,opt,model):
        istargeted=opt.attack.endswith('T')
        self.istargeted=istargeted
        num_classes=opt.classifier_classes
        self.model=inner_model(model)
        self.opt=opt
        self.count=0

    def __call__(self,org,org_l,pair,pair_l):
        if self.istargeted:
            image,count = optim(self.opt,self.model,org,org_l,pair,target=pair_l)
        else:
            image,count = optim(self.opt,self.model,org,org_l,pair)
        self.count=count
        return image

    def get_count(self):
        return self.count