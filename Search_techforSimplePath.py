
def bfs(graph, source, destination):
    frontier_queue= []
    explored_set = []
    child_and_parent = []
    path = []

    node = source
    frontier_queue.append(node)

    if frontier_queue == destination:           #child is source

        out_file.write(str(source))             #write output file
        out_file.write(" ")
        out_file.write(str(0) + '\n')
    else:
        child = frontier_queue[0]

        while child != 0:                               #run loop till child is found
            explored_node = frontier_queue.pop(0)       #frontier queue has all nodes which need to be searched for a child
                                                        #BFS follows Queue implementation so all noded are added at the end and its FIFO

            explored_node = explored_node.split('\n')[0]

            explored_set.append(explored_node)          #explored set to keep track of already explored nodes, to avoid loops.
            for child in graph[explored_node]:
                if ((child not in explored_set) and (child not in frontier_queue)): #check to see if child in not already travelled, to avoid loop


                    if (child == destination):

                        child_and_parent.insert(0,[child,explored_node])
                        travelled_branch=[child]
                        """Each node is stored as [child,parent] and the list has an example of [[D,C],[D,B],[C,A]].
                        the part below would find the path as D->C->A by searching through the list. Start is set as
                        child and then when child is found in the list, start is upadated as parent and then it searches the list
                        and then it keeps on backtracking to the root node"""
                        for lists in child_and_parent:

                            if (lists[0] == child):
                                travelled_branch.insert(0,lists[1])
                                child = lists[1]
                                if child == source:

                                    for item in range(len(travelled_branch)):

                                        out_file.write(str(travelled_branch[item]))
                                        out_file.write(" ")
                                        out_file.write(str(item)+'\n')



                                    child = 0
                                    return (len(travelled_branch)-1)

                    else:
                        # if node is not child then we should insert is as a list in a child_parent list so that backtracking can be easier later

                        frontier_queue.append(child)
                        child_and_parent.insert(0,[child,explored_node])


def dfs(graph, source, destination,parent):
    frontier_queue = []
    explored_set = []
    path = OrderedDict()
    child = source
    copy_graph=graph
    path[source]=source

    if child == destination:

        out_file.write(str(source))
        out_file.write(" ")
        out_file.write(str(0) + '\n')

    else:
        while child != 0:
            #child = explored_node
            if ((child in frontier_queue) or (child in explored_set)):
                child = frontier_queue.pop(0)                           #DFS follows Stack Implementation. LIFO

            elif ((child not in explored_set) and (child not in frontier_queue)):
                if child == destination:
                    c=[]
                    while (child != source):
                        """Another approch to find parent is to store the child as the key of a dictionary and its value as the parent.
                        Backtracking would help find the appropriate path"""
                        if child in path.keys():

                            a = path[child]
                            b=a[0]
                            c.insert(0,child)
                            child = b
                    c.insert(0,source)

                    for item in range(len(c)):
                        out_file.write(str(c[item]))
                        out_file.write(" ")
                        out_file.write(str(item) + '\n')
                    child = 0

                    return len(path),"L"

                else:
                    if child in graph.keys():

                        explored_set.insert(0,child)
                        parent = child
                        """This is to update the path dictionary, which contains info of where the node came came from"""
                        for i in copy_graph[child]:

                            if i not in path.keys():
                                path[i] = [parent]
                            else:
                                path[i].insert(len(path[i]), parent)
                        for item in graph[child][::-1]:
                            frontier_queue.insert(0, item)

                        child = frontier_queue.pop(0)

                    else:
                        child = frontier_queue.pop(0)

def ucs(graph_ucs ,source, destination, tie_breaker):
    frontier = []
    frontier_values= []
    frontier_queue= []
    frontier_nodes = []
    explored_nodes = []
    explored_set = []
    came_from = []
    path = OrderedDict()
    copy_graph={}
    copy_graph = graph_ucs
    child = source
    weight = 0

    frontier_queue.append([child,weight])
    parent = child
    path[source]=[[source,0]]
    c=[]
    if child == destination:
        out_file.write(str(source))
        out_file.write(" ")
        out_file.write(str(0) + '\n')
    else:
        while len(frontier_queue) != 0:     #loop runs till queue is empty so that no path is undiscovered

            frontier_queue = sorted(frontier_queue,key=itemgetter(1))   # sort using weights
            frontier_nodes = list(map(itemgetter(0),frontier_queue))    # to get a list of frontier nodes
            child_and_weight = frontier_queue.pop(0)                    #should return first child & weight
            child = child_and_weight.pop(0)                             #got child
            weight = int(child_and_weight.pop(0))                       #got weight
            frontier_nodes.pop(0)                                       # to remove child from frontier, this helps to check if another path is present
            if ((child in explored_nodes) or (child in frontier_nodes)):

                if child in explored_nodes:                             #does not make a difference
                    remove_child = frontier_queue.pop(0)
                elif child in frontier_nodes and child not in explored_nodes :

                    while (child in frontier_nodes):
                        for item_check in frontier_queue:

                            #to update weight of the child if a shorter path has been found.
                            if item_check[0] == child:

                                if weight <= int(item_check[1]):
                                    item_check[1] = weight

                                elif (weight > int(item_check[1])):
                                    weight = int(item_check[1])

                        frontier_queue = sorted(frontier_queue, key=itemgetter(1))
                        frontier_nodes = list(map(itemgetter(0), frontier_queue))
                        child_and_weight = frontier_queue.pop(0)
                        child = child_and_weight.pop(0)
                        weight = int(child_and_weight.pop(0))
                        frontier_nodes.pop(0)
                    if child == destination:
                        #path dictionary to keep a track if which child came from which parent
                        c=[]
                        while child!=source:
                            if child in path.keys():
                                a =path[child]
                                b=a[0]
                                c.insert(0,[child,b[1]])

                                #out_file.write(str)
                                child = b[0]
                        c.insert(0,[source,0])

                        child =0
                        for item in c:
                            out_file.write(str(item[0]))
                            out_file.write(" ")
                            out_file.write(str(item[1])+'\n')
                        return weight
                    else:
                        # if child was not destination but another node had to update its value because it was found through a shorter path
                        if child in graph_ucs.keys():

                            for item_z in copy_graph[child]:
                                item_z[1] = int(item_z[1]) + weight

                            frontier_queue = copy_graph[child]+ frontier_queue
                            #frontier_queue = sorted(frontier_queue, key=itemgetter(1))
                            explored_set.insert(0, [child, weight])
                            explored_nodes.append(child)
                            parent = child
                            for item in copy_graph[child]:

                                came_from.append([item[0], parent])
                                if item[0] not in path.keys():
                                    path[item[0]] = [[parent, item[1]]]
                                else:
                                    path[item[0]].insert(len(path[item[0]]), [parent, int(item[1])])

                        else:
                            explored_set.insert(0,[child,weight])
                            explored_nodes.append(child)
                            frontier_queue = sorted(frontier_queue, key=itemgetter(1))


            elif ((child not in explored_nodes) and (child not in frontier_nodes)):

                if child == destination:
                    c = []
                    while child != source:
                        if child in path.keys():
                            a = path[child]
                            b = a[0]
                            c.insert(0, [child, b[1]])

                            # out_file.write(str)
                            child = b[0]
                    c.insert(0, [source, 0])
                    for item in c:
                        out_file.write(str(item[0]))            #writing output.txt
                        out_file.write(" ")
                        out_file.write(str(item[1]) + '\n')

                    child = 0
                    return weight
                else:
                    # to avoid leaf nodes being explanded inside the frontier queue and updating the path and explored set of nodes to avoid loops
                    if child in graph_ucs.keys():
                        for itemz in copy_graph[child]:
                            itemz[1] = int(itemz[1]) + weight

                        frontier_values = list(map(itemgetter(1), frontier_queue))
                        """Loop here is to determine that if two same cost nodes are reached then which node should be visited first
                        the convention chosen is according to which node was first received in input.txt. It check which node was
                        obttained first and then updates the values"""
                        for item in copy_graph[child]:
                            check_child = item[0]
                            check_weight = item[1]
                            count = 0
                            for k in frontier_values:
                                if k == check_weight:
                                    count = count +1

                            if check_weight in frontier_values:

                                value = []
                                open_check = []
                                parent = child

                                for item in copy_graph[child]:

                                    if item[0] not in path.keys():
                                        path[item[0]] = [[parent, int(item[1])]]


                                    elif item[0] in path.keys():

                                        path[item[0]].insert(len(path[item[0]]), [parent, int(item[1])])

                                    came_from.append([item[0], parent])


                                open_check = copy_graph[child] + frontier_queue

                                for item in open_check:


                                    if item[0] in path.keys():

                                        for lst10 in path[item[0]]:

                                            if (lst10[1] == check_weight):

                                                check_parent = lst10[0]
                                                for i in tie_breaker[item[0]]:
                                                    if i[0] == check_parent:
                                                        tbv = i[1]
                                                        value.append([item[0],i[1]])
                                            value = sorted(value,key=itemgetter(1))
                                            appending_order=[]
                                            for abc in value:
                                                appending_order.append([abc[0],check_weight])

                                for val in range(count):
                                    frontier_queue.pop(0)
                                frontier_queue = appending_order + frontier_queue
                                frontier_queue = frontier_queue = sorted(frontier_queue,key=itemgetter(1))

                            elif check_weight not in frontier_values:
                                parent = child

                                for item in copy_graph[child]:
                                    came_from.append([item[0], parent])
                                    if item[0] not in path.keys():
                                        path[item[0]] = [[parent, item[1]]]
                                    else:
                                        path[item[0]].insert(len(path[item[0]]), [parent, int(item[1])])

                                frontier_queue = copy_graph[child] + frontier_queue

                                frontier_queue = sorted(frontier_queue, key=itemgetter(1))

                                explored_set.insert(0,[child,weight])
                                explored_nodes.append(child)

                                break



                    else:
                        explored_set.insert(0,[child,weight])
                        frontier_queue = sorted(frontier_queue, key=itemgetter(1))






def a_star(graph_ucs ,source, destination, heurisitcs,tie_breaker):
    frontier = []
    frontier_values = []
    frontier_queue= []
    frontier_nodes = []
    explored_nodes = []
    explored_set = []

    copy_graph={}
    copy_graph = graph_ucs

    child = source
    weight = 0
    came_from = []
    path = OrderedDict()

    frontier_queue.append([child, weight,0])

    if child == destination:
        out_file.write(str(source))
        out_file.write(" ")
        out_file.write(str(0) + '\n')

    else:

        while len(frontier_queue) != 0:

            frontier_queue = sorted(frontier_queue,key=itemgetter(2)) # sort using weights

            frontier_nodes = map(itemgetter(0),frontier_queue) # to get a list of frontier nodes
            frontier_nodes = list(frontier_nodes)
            child_weight_heuristic = frontier_queue.pop(0)        #should return first child & weight

            child = child_weight_heuristic.pop(0) #got child
            weight = int(child_weight_heuristic.pop(0))    #got weight

            sum_heuristic_weight = child_weight_heuristic.pop(0)

            frontier_nodes.pop(0) # to remove child from frontier, this helps to check if another path is present

            if ((child in explored_nodes) or (child in frontier_nodes)):

                if child in frontier_nodes and child not in explored_nodes :

                    while (child in frontier_nodes):

                        for item_check in frontier_queue:
                            val=[sum_heuristic_weight]

                        #updating cost to minimum value. If cost is already min no action is taken. If cost is not min, it is updated
                            if item_check[0] == child:
                                val.append(item_check[2])
                                if weight <= int(item_check[1]):

                                    item_check[1] = weight

                                elif (weight > int(item_check[1])):
                                    weight = int(item_check[1])

                                item_check[2] = min(val)

                        #Updating queue as per new cost
                        frontier_queue = sorted(frontier_queue, key=itemgetter(2))
                        frontier_nodes = map(itemgetter(0), frontier_queue)
                        frontier_nodes = list(frontier_nodes)

                        child_and_weight = frontier_queue.pop(0)
                        child = child_and_weight.pop(0)
                        weight = int(child_and_weight.pop(0))

                        frontier_nodes.pop(0)

                        break
                    if child == destination:
                        c = []
                        while child != source:
                            if child in path.keys():
                                a = path[child]             #path updation for backtracking
                                b = a[0]
                                c.insert(0, [child, b[1]])
                                # out_file.write(str)
                                child = b[0]
                        c.insert(0, [source, 0])
                        child = 0

                    else:
                        #to get total cost from source
                        if child in graph_ucs.keys():

                            for item_z in copy_graph[child]:
                                item_z[1] = int(item_z[1]) + weight
                            for check in copy_graph[child]:
                                if check[0] in heuristics.keys():
                                    value = heuristics[check[0]] + check[1]

                                    check.append(value)


                            frontier_queue = copy_graph[child] + frontier_queue

                            explored_set.insert(0, [child, weight])
                            explored_nodes.append(child)

                            parent = child
                            for item in copy_graph[child]:

                                came_from.append([item[0], parent])
                                if item[0] not in path.keys():
                                    path[item[0]] = [[parent, item[1]]]
                                else:
                                    path[item[0]].insert(len(path[item[0]]), [parent, int(item[1])])

                        else:
                            explored_set.insert(0,[child,weight])
                            explored_nodes.append(child)
                            frontier_queue = sorted(frontier_queue, key=itemgetter(2))


            elif ((child not in explored_nodes) and (child not in frontier_nodes)):
                if child == destination:

                    c = []
                    while child != source:
                        if child in path.keys():
                            a = path[child]             #Updating path for backtracking
                            b = a[0]
                            c.insert(0, [child, b[1]])

                            # out_file.write(str)
                            child = b[0]
                    c.insert(0, [source, 0])



                    child = 0


                else:

                    if child in graph_ucs.keys():

                        for itemz in copy_graph[child]:

                            itemz[1] = int(itemz[1]) + weight


                        for check in copy_graph[child]:
                            if check[0] in heuristics.keys():
                                value = heuristics[check[0]] + check[1]

                                check.append(value)

                        #frontier_queue = copy_graph[child] + frontier_queue
                        frontier_value = list(map(itemgetter(2),frontier_queue))

                        for item in copy_graph[child]:
                            check_child = item[0]
                            check_weight = item[1]
                            check_heuristic = item[2]

                            count = 0
                            for k in frontier_values:
                                if k == check_heuristic:
                                    count = count + 1


                            if check_heuristic in frontier_values:

                                value = []
                                open_check = []
                                parent = child

                                for item in copy_graph[child]:

                                    if item[0] not in path.keys():
                                        path[item[0]] = [[parent, int(item[1])]]


                                    elif item[0] in path.keys():

                                        path[item[0]].insert(len(path[item[0]]), [parent, int(item[1])])

                                    came_from.append([item[0], parent])

                                open_check = copy_graph[child] + frontier_queue
                                """Loop here is to determine that if two same cost nodes are reached then which node should be visited first
                                the convention chosen is according to which node was first received in input.txt. It check which node was
                                obtained first and then updates the values"""

                                for item in open_check:

                                    if item[0] in path.keys():

                                        for lst10 in path[item[0]]:

                                            if (lst10[2] == check_heuristic):

                                                check_parent = lst10[0]
                                                for i in tie_breaker[item[0]]:
                                                    if i[0] == check_parent:
                                                        tbv = i[2]
                                                        value.append([item[0], i[1]])

                                            value = sorted(value, key=itemgetter(1))
                                            appending_order = []
                                            for abc in value:
                                                appending_order.append([abc[0], check_weight ,check_heuristic])

                                for i in range(count):
                                    frontier_queue.pop(0)

                                frontier_queue = appending_order + frontier_queue
                                frontier_queue = frontier_queue = sorted(frontier_queue, key=itemgetter(2))

                            elif check_heuristic not in frontier_values:
                                parent = child

                                for item in copy_graph[child]:
                                    came_from.append([item[0], parent])
                                    if item[0] not in path.keys():
                                        path[item[0]] = [[parent, item[1]]]         #path updation for backtracking
                                    else:
                                        path[item[0]].insert(len(path[item[0]]), [parent, int(item[1])])

                                frontier_queue = copy_graph[child] + frontier_queue

                                frontier_queue = sorted(frontier_queue, key=itemgetter(2))

                                explored_set.insert(0, [child, weight])
                                explored_nodes.append(child)

                                break



                        # updated non equal nodes as per heuristics
                    else:
                        explored_set.insert(0,[child,weight])
                        frontier_queue = sorted(frontier_queue, key=itemgetter(2))

        for item in c:
            out_file.write(str(item[0]))
            out_file.write(" ")
            out_file.write(str(item[1]) + '\n')

#*** Reading input file****
in_file = open("input.txt", "r")
out_file = open("output.txt", "w")

ALGO = in_file.readline().split('\n')[0]    #which search technique to follow
START = in_file.readline().split('\n')[0]   #start node
GOAL = in_file.readline().split('\n')[0]    #goal Node

Active_TrafficLines = int(in_file.readline())   #Traffic data between nodes

active_states = []
heuristic_states = []
count = 0
for item1 in range(Active_TrafficLines):
    active_states.append(in_file.readline().split('\n')[0])
    count = count + 1
    active_states.append(str(count))

#if (ALGO == "A*"):
Sunday_TrafficLines = int(in_file.readline())   #Heuristic data for A* search
for b in range(Sunday_TrafficLines):
    heuristic_states.append(in_file.readline().split('\n')[0])

characters = []
distance=[]

for j in range(len(active_states)):
    characters.append(active_states[j].split(" "))

for a in range(0,len(characters),2):
    characters[a] = characters[a] + characters[a+1]

new_characters = [characters[i] for i in range(0,len(characters)) if (i) % 2 == 0]

for k in range(len(heuristic_states)):
    distance.append(heuristic_states[k].split(" "))

#ordered dictionary to keep track of which node came first
from collections import OrderedDict

class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)
"""This is to obtain graphs as dictionaries -> A:{B,C} this means A has an edge to B and A has an edge to C or A
is parent of both B and C. A:(10,5) means A->B distance is 10 and A->C distance is 5. All the work done below
is to get meaningful graph out of the input file. Form A B 10 in input.txt ->"AB10 -> A,B,10 -> A:B and A:10. Similar
work is done for heuristics"""

graph = OrderedDict()
# for bfs and dfs graph
for lst1 in new_characters:

    if lst1[0] not in graph.keys():
        graph[lst1[0]] = [lst1[1]]


    elif lst1[0] in graph.keys():
        graph[lst1[0]].append(lst1[1])

parent = OrderedDict()

# parent for dfs
for lst2 in new_characters:
    if lst2[1] not in parent.keys():
        parent[lst2[1]]=[lst2[0]]

    elif lst2[1] in parent.keys():
        parent[lst2[1]].append(lst2[0])

heuristics = {}

for lst4 in distance:
    heuristics[lst4[0]] = int(lst4[1])

#for UCS weights
graph_ucs= OrderedDict()
for lst3 in new_characters:
    if lst3[0] not in graph_ucs.keys():
        graph_ucs[lst3[0]] = [[lst3[1],int(lst3[2])]]

    elif lst3[0] in graph_ucs.keys():
        graph_ucs[lst3[0]].insert(len(graph_ucs[lst3[0]]),[lst3[1],int(lst3[2])])


tie_breaker = OrderedDict()

for lst5 in new_characters:
    if lst5[1] not in tie_breaker.keys():
        tie_breaker[lst5[1]] = [[lst5[0],int(lst5[3])]]

    elif lst5[1] in tie_breaker.keys():
        tie_breaker[lst5[1]].insert(len(tie_breaker[lst5[1]]),[lst5[0],int(lst5[3])])

from operator import itemgetter


if (ALGO =="BFS"):
    bfs(graph, START, GOAL)

elif (ALGO == "DFS"):
    dfs(graph, START, GOAL, parent)

elif (ALGO == "UCS"):
    ucs(graph_ucs, START, GOAL, tie_breaker)

elif (ALGO == "A*"):
    a_star(graph_ucs, START, GOAL, heuristics,tie_breaker)


#**closing files**
in_file.close()
out_file.close()

