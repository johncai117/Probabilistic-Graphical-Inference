import pickle
import numpy as np
import argparse
import networkx as nx
#from datetime import datetime ## time


def load_graph(filename):
    'load the graphical model (DO NOT MODIFY)'
    return pickle.load(open(filename, 'rb'))


def inference_brute_force(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input:
        G:
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K:
            G.graph['K']
        unary potentials:
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials:
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients:
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']:
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']:
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']:
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']:
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    assignment_l = [] ## stores assignment
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        #print(G.nodes[v]['unary_potential'])
        assignment_l.append(G.nodes[v]['assignment'])
        #print(v)
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
        print(e)
        #print(G.edges[e]['binary_potential'])
    G.graph['v_map'] = np.zeros(len(G.nodes))

    # YOUR CODE STARTS
    K = G.graph['K']
    num_nodes = len(G.nodes)


    def recurse_brute(num_calls, temp_list_l):
        if num_calls == 0:
            for k in range(K):
                temp_list_l.append([k])
            return recurse_brute(num_calls + 1, temp_list_l)
        elif num_calls < num_nodes:
            new_temp_list = []
            for v in temp_list_l:
                for k in range(K):
                    v2 = v.copy()
                    v2.append(k)
                    new_temp_list.append(v2)
            temp_list_l = new_temp_list
            return recurse_brute(num_calls + 1, temp_list_l)
        elif num_calls == num_nodes:
            output = temp_list_l
            return output

    possibility_list = recurse_brute(0, [])

    Z = 0
    prob_list = []
    for liz in possibility_list:
        prod = 1
        for v in G.nodes:
            prod *= G.nodes[v]['unary_potential'][liz[v]]
        for e in G.edges:
            prod *= G.edges[e]['binary_potential'][liz[e[0]], liz[e[1]]]
        prob_list.append(prod)
        Z += prod

    prob_list = prob_list / Z

    #print(len(prob_list))

    for v in G.nodes:
        for k in range(K):
            for pr, liz in zip(prob_list, possibility_list):
                if liz[v] == k:
                    G.nodes[v]['marginal_prob'][k] += pr
        print(G.nodes[v]['marginal_prob'])

    G.graph['v_map'] = np.asarray(possibility_list[np.argmax(np.asarray(prob_list))])

    #print("MAP")
    #print(G.graph['v_map'])

    #print(assignment_l)
    ## first term in unary
    for v, a in zip(G.nodes, assignment_l):
        G.nodes[v]["gradient_unary_potential"][a] += 1/ G.nodes[v]['unary_potential'][a]
        #print(G.nodes[v]["gradient_unary_potential"])

    ## second term in unary
    for v in G.nodes:
        for k in range(K):
            dZ = 0
            for liz in possibility_list:
                if liz[v] == k: ## check that the possibilities only have the restricted sum
                    prod = 1
                    for v2 in G.nodes:
                        if v2 != v:
                            prod *= G.nodes[v2]['unary_potential'][liz[v2]]
                    for e in G.edges:
                        prod *= G.edges[e]['binary_potential'][liz[e[0]], liz[e[1]]]
                    dZ += prod
            G.nodes[v]["gradient_unary_potential"][k] -= dZ/Z
        #print(G.nodes[v]["gradient_unary_potential"])

    for e in G.edges:
        for k in range(K):
            for k2 in range(K):
                if k == assignment_l[e[0]] and k2 == assignment_l[e[1]]:
                    G.edges[e]['gradient_binary_potential'][k,k2] += 1/ G.edges[e]['binary_potential'][k,k2]
                dZ = 0
                for liz in possibility_list:
                    if liz[e[0]] == k and liz[e[1]] == k2:
                        prod = 1
                        for v in G.nodes:
                            prod *= G.nodes[v]['unary_potential'][liz[v]]
                        for e2 in G.edges:
                            if e2 != e:
                                prod *= G.edges[e2]['binary_potential'][liz[e2[0]], liz[e2[1]]]
                        dZ += prod
                G.edges[e]['gradient_binary_potential'][k,k2] -= dZ/Z

    #for e in G.edges:
        #print(G.edges[e]['gradient_binary_potential'])
    print(G.graph['v_map'])

    return G


def inference(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input:
        G:
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K:
            G.graph['K']
        unary potentials:
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials:
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients:
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']:
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']:
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']:
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']:
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    assignment_l = [] ## stores assignment
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        #print(G.nodes[v]['unary_potential'])
        assignment_l.append(G.nodes[v]['assignment'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
        #print(e)
    G.graph['v_map'] = np.zeros(len(G.nodes))

    # YOUR CODE STARTS
    K = G.graph['K']
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)

    print(assignment_l)

    degree_l = [G.degree(x) for x in G.nodes()]
    #print(degree_l)

    leaf_nodes = [x for x in G.nodes() if G.degree(x)==1] ## this is possible because we have the requirement that it is a tree - connected and without loops
    #print(leaf_nodes)
    neighbour_l = [list(G.neighbors(x)) for x in G.nodes()] ##get neighbors
    #print(neighbour_l)

    ## Messages
    for d,v in zip(degree_l,G.nodes):
        G.nodes[v]['Messages'] = np.zeros((d,K))
        G.nodes[v]['Num_Messages'] = np.zeros(1).astype(int)
        G.nodes[v]['Messages_tracker'] = []
        G.nodes[v]['Messages_max'] = np.zeros((d,K))
        G.nodes[v]['Back_pointer'] = np.zeros((d,K)).astype(int)
        G.nodes[v]['Back_pointer_idx'] = np.zeros((d)).astype(int)
        #G.nodes[v]['Back_pointer_idx'][:] = np.nan ## ensure it fills up


    def belief_prop(message_count, nodes_eval, full_nodes, deepest_node):
        #instantiate variables
        new_nodes = []
        new_full_nodes = []
        new_deepest_node = deepest_node


        ##terminal case
        if message_count == num_edges * 2:
            #print("CHECK")
            #print(nodes_eval)
            #print(full_nodes)
            #print(nodes_eval)
            #print(deepest_node)
            for l in nodes_eval:
                if l in full_nodes and not l in deepest_node: ##fix for 2 cases
                    #print(l)
                    ##calculate marginals
                    grad_term2 = np.prod(G.nodes[l]['Messages'], axis = 0)
                    #print(grad_term2)

                    G.nodes[l]['marginal_prob'] = np.multiply(grad_term2, G.nodes[l]['unary_potential'])
                    Z = sum(G.nodes[l]['marginal_prob']) ##normalizer
                    #print(Z)
                    G.nodes[l]['marginal_prob'] = G.nodes[l]['marginal_prob'] / Z
                    G.nodes[l]['gradient_unary_potential'][assignment_l[l]] += (1 / G.nodes[l]['unary_potential'][assignment_l[l]])
                    G.nodes[l]['gradient_unary_potential'] -= grad_term2/ Z
                if l in deepest_node:
                    temp_message_prod_max_all = np.prod(G.nodes[l]['Messages_max'], axis = 0)
                    argmax_idx = np.argmax(np.multiply(temp_message_prod_max_all, G.nodes[l]['unary_potential']))
                    G.graph['v_map'][l] = argmax_idx
                    for idx, backpointer_l in zip(G.nodes[l]['Back_pointer_idx'], G.nodes[l]['Back_pointer']):
                        if not G.nodes[int(idx)]['Num_Messages'] == degree_l[int(idx)]:
                            G.graph['v_map'][int(idx)] = backpointer_l[int(argmax_idx)]

            return ##exit function and return Z

        ## cases in between
        if message_count < num_edges*2:
            for l in nodes_eval:
                if l in deepest_node or l in full_nodes:
                    ##calculate marginals
                    grad_term2 = np.prod(G.nodes[l]['Messages'], axis = 0)
                    G.nodes[l]['marginal_prob'] = np.multiply(grad_term2, G.nodes[l]['unary_potential'])
                    Z = sum(G.nodes[l]['marginal_prob']) ##normalizer
                    G.nodes[l]['marginal_prob'] = G.nodes[l]['marginal_prob'] / Z
                    G.nodes[l]['gradient_unary_potential'][assignment_l[l]] += (1 / G.nodes[l]['unary_potential'][assignment_l[l]])
                    G.nodes[l]['gradient_unary_potential'] -= grad_term2/ Z

                if l in deepest_node:
                    temp_message_prod_max_all = np.prod(G.nodes[l]['Messages_max'], axis = 0)
                    argmax_idx = np.argmax(np.multiply(temp_message_prod_max_all, G.nodes[l]['unary_potential']))
                    G.graph['v_map'][l] = argmax_idx
                    for idx, backpointer_l in zip(G.nodes[l]['Back_pointer_idx'], G.nodes[l]['Back_pointer']):
                        G.graph['v_map'][int(idx)] = backpointer_l[int(argmax_idx)]
                        #new_deepest_node.append(idx)
                elif l in full_nodes:
                    argmax_idx = G.graph['v_map'][l]
                    for idx, backpointer_l in zip(G.nodes[l]['Back_pointer_idx'], G.nodes[l]['Back_pointer']):
                        if not G.nodes[int(idx)]['Num_Messages'] == degree_l[int(idx)]:
                            G.graph['v_map'][int(idx)] = backpointer_l[int(argmax_idx)]

                for e in G.edges:
                    if e[0] == l and not G.nodes[e[1]]['Num_Messages'] == degree_l[e[1]]: ## don't double compute
                        ## Either the node is full and needs to start firing back or the edge is pointing to a new node
                        if G.nodes[e[0]]['Num_Messages'] == degree_l[e[0]] or e[1] not in G.nodes[e[0]]['Messages_tracker']:
                            receive_idx = neighbour_l[e[1]].index(l) ## find the index of the origin node within the node_list of the en
                            send_idx = neighbour_l[l].index(e[1])
                            temp_message_prod_max = np.ones(K)
                            temp_message_prod = np.ones(K)
                            if not deepest_node:
                                for i,(m, maxm) in enumerate(zip(G.nodes[l]['Messages'], G.nodes[l]['Messages_max'])):
                                    if i != send_idx:
                                        temp_message_prod = np.multiply(temp_message_prod, m)
                                        temp_message_prod_max = np.multiply(temp_message_prod_max, maxm)
                            else:
                                for i,m  in enumerate(G.nodes[l]['Messages']):
                                    if i != send_idx:
                                        temp_message_prod = np.multiply(temp_message_prod, m)

                            ## get unary multipled by binary and messages
                            G.nodes[e[1]]['Messages'][receive_idx] = np.matmul(G.edges[e]['binary_potential'].T, np.multiply(G.nodes[e[0]]['unary_potential'], temp_message_prod))
                            ## Normalize for numerical stability
                            G.nodes[e[1]]['Messages'][receive_idx] = G.nodes[e[1]]['Messages'][receive_idx] / max(G.nodes[e[1]]['Messages'][receive_idx]) ## normalization for numerical stability

                            ## max values
                            if not deepest_node:
                                temp_max = np.multiply(G.edges[e]['binary_potential'].T ,np.multiply(G.nodes[e[0]]['unary_potential'], temp_message_prod_max))
                                max_val = np.amax(temp_max, axis = 1) ##max value
                                max_backpoint = np.argmax(temp_max, axis = 1) ##max index
                                G.nodes[e[1]]['Messages_max'][receive_idx] = max_val / max(max_val)##max value - prevent overflow
                                G.nodes[e[1]]['Back_pointer'][receive_idx] = max_backpoint ##back pointer
                            G.nodes[e[1]]['Back_pointer_idx'][receive_idx] = e[0]

                            message_count +=1
                            ## check if the number of elements fulfills the requirement for the message
                            G.nodes[e[1]]['Num_Messages'] += 1
                            G.nodes[e[1]]['Messages_tracker'].append(e[0]) ##tracks that it has received a message
                            if G.nodes[e[1]]['Num_Messages'] >= degree_l[e[1]] - 1 and e[1] not in new_nodes:
                                new_nodes.append(e[1])

                            if G.nodes[e[1]]['Num_Messages'] == degree_l[e[1]]:
                                new_full_nodes.append(e[1])
                                if not full_nodes and len(new_full_nodes) == 1: ##get the first one
                                    new_deepest_node.append(e[1])


                    if e[1] == l and not G.nodes[e[0]]['Num_Messages'] == degree_l[e[0]]: ## don't double compute
                        ## Either the node is full and needs to start firing back or the edge is pointing to a new node
                        if G.nodes[e[1]]['Num_Messages'] == degree_l[e[1]] or e[0] not in G.nodes[e[1]]['Messages_tracker']:
                            receive_idx = neighbour_l[e[0]].index(l) ## find the index of the labour within the node_list
                            send_idx = neighbour_l[l].index(e[0])
                            temp_message_prod = np.ones(K)
                            temp_message_prod_max = np.ones(K)
                            if not deepest_node:
                                for i,(m, maxm) in enumerate(zip(G.nodes[l]['Messages'], G.nodes[l]['Messages_max'])):
                                    if i != send_idx:
                                        temp_message_prod = np.multiply(temp_message_prod, m)
                                        temp_message_prod_max = np.multiply(temp_message_prod_max, maxm)
                            else:
                                for i, m in enumerate(G.nodes[l]['Messages']):
                                    if i != send_idx:
                                        temp_message_prod = np.multiply(temp_message_prod, m)

                            ## get unary multipled by binary and messages
                            G.nodes[e[0]]['Messages'][receive_idx] = np.matmul(G.edges[e]['binary_potential'], np.multiply(G.nodes[e[1]]['unary_potential'], temp_message_prod))
                            ## Normalize for numerical stability
                            G.nodes[e[0]]['Messages'][receive_idx] = G.nodes[e[0]]['Messages'][receive_idx] / max(G.nodes[e[0]]['Messages'][receive_idx]) ## normalization for numerical stability

                            ## max values
                            if not deepest_node:
                                temp_max = np.multiply(G.edges[e]['binary_potential'] ,np.multiply(G.nodes[e[1]]['unary_potential'], temp_message_prod_max))
                                max_val = np.amax(temp_max, axis = 1) ##max value
                                max_backpoint = np.argmax(temp_max, axis = 1) ##max index
                                G.nodes[e[0]]['Messages_max'][receive_idx] = max_val / max(max_val) ##max value - prevent overflow
                                G.nodes[e[0]]['Back_pointer'][receive_idx] = max_backpoint ##back pointer

                            G.nodes[e[0]]['Back_pointer_idx'][receive_idx] = e[1]

                            ## get unary multipled by binary
                            message_count +=1
                            ## check if the number of elements fulfills the requirement for the message
                            G.nodes[e[0]]['Num_Messages'] += 1
                            G.nodes[e[0]]['Messages_tracker'].append(e[1])
                            if G.nodes[e[0]]['Num_Messages'] >= degree_l[e[0]] - 1 and e[0] not in new_nodes:
                                new_nodes.append(e[0])
                            if G.nodes[e[0]]['Num_Messages'] == degree_l[e[0]]:
                                new_full_nodes.append(e[0])
                                if not full_nodes and len(new_full_nodes) == 1: ## get the first one
                                    new_deepest_node.append(e[0])

            return (belief_prop(message_count, new_nodes, new_full_nodes, new_deepest_node))

    emp_list = []
    emp_list2 = []



    belief_prop(0, leaf_nodes, emp_list, emp_list2)


    #for v in G.nodes:
        #print("hi")
        #print(G.nodes[v]['Messages'])
        #G.nodes[v]['marginal_prob'] = np.multiply(np.prod(G.nodes[v]['Messages'], axis = 0), G.nodes[v]['unary_potential'])
        #G.nodes[v]['marginal_prob'] = G.nodes[v]['marginal_prob'] / sum(G.nodes[v]['marginal_prob'])
        #print(G.nodes[v]['marginal_prob'])
    #for v in G.nodes:
        #print(G.nodes[v]['gradient_unary_potential'])



    for e in G.edges:
        matrix_term2 = np.ones((G.graph['K'], G.graph['K']))
        assign_e0 = assignment_l[e[0]] ##find the class assigned
        assign_e1 = assignment_l[e[1]]
        edge_idx_e0 = neighbour_l[e[0]].index(e[1])
        edge_idx_e1 = neighbour_l[e[1]].index(e[0])

        ## calculate value of messages without including the edge
        temp_message_prod_e0 = np.ones(K)
        temp_message_prod_e1 = np.ones(K)
        for i, m in enumerate(G.nodes[e[0]]['Messages']):
            if i != edge_idx_e0:
                temp_message_prod_e0 = np.multiply(temp_message_prod_e0, m)

        for i, m2 in enumerate(G.nodes[e[1]]['Messages']):
            if i != edge_idx_e1:
                temp_message_prod_e1 = np.multiply(temp_message_prod_e1, m2)

        #mutliply unary and messages together

        unary_messages_e0 = np.multiply(temp_message_prod_e0, G.nodes[e[0]]['unary_potential'])
        unary_messages_e0 = np.expand_dims(unary_messages_e0, 1)
        unary_messages_e1 = np.multiply(temp_message_prod_e1, G.nodes[e[1]]['unary_potential'])

        ## multuply and broadcast terms
        matrix_term2 = np.multiply(matrix_term2, unary_messages_e0)
        matrix_term2 = np.multiply(matrix_term2, unary_messages_e1)

        ##calculate normalization term
        Z = np.sum(np.multiply(matrix_term2, G.edges[e]['binary_potential']))

        ## Add the entry of the gradient based on the assignment
        G.edges[e]['gradient_binary_potential'][assign_e0, assign_e1] = 1 / G.edges[e]['binary_potential'][assign_e0, assign_e1]

        #minus (matrix_term/Z) from the gradient
        G.edges[e]['gradient_binary_potential'] -= matrix_term2 / Z
        #print(G.edges[e]['gradient_binary_potential'])


    print(G.graph['v_map'])

    return G



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The input graph')
    args = parser.parse_args()
    G = load_graph(args.input)
    inference(G)
    pickle.dump(G, open('results_' + args.input, 'wb'))


    G2 = nx.generators.trees.random_tree(2000, seed = 1170)

    np.random.seed(1170)
    G2.graph['K'] = 5
    b = 5
    a = 1
    for v in G2.nodes:
        G2.nodes[v]['assignment'] = np.random.randint(0, G2.graph['K'])
        G2.nodes[v]['unary_potential'] = (b - a) * np.random.random_sample(G2.graph['K']) + a

    for e in G2.edges:
        G2.edges[e]['binary_potential'] = (b - a) * np.random.random_sample((G2.graph['K'], G2.graph['K'])) + a

    inference(G2)
    #print(len(G2))
