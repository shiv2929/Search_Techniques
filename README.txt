Search Techniques-
This involves finding the lowest cost or the shortest path from one point to another. 
It involves a start point and an end point. The program reads from an input.txt file and writes the output to an output.txt file.
for example:-

UCS	-Search Technique
A	- Start
E	-End
4	-# of Active traffic Lines 
A B 1	-1 unit time to reach from A to B
A C 4
A D 10	-10 unit time to reach from A to D
B E 1
5	-# of Heuristic(used for A* search only) 
A 1	-value at intersection
B 1
C 1
D 1
E 1

Based on the input file provided, an output file has to be generated which gives which gives path from start to end with cost/time at each intersection.

A 0
B 1
E 2

would mean, that it reached at 2 cost from A to E. For BFS and DFS step costs are 1 and for USC and A* step costs are as given.

The input file is taken and a graph is made as {parent: childen} and {parent:distane}. Each time a child is compared as destination and if it not the destination it is updated in a path list {child : parent} which helps in backtracking. Also, child is updated in an explored set to avoid loops. Tie Breaker is used at points where there is a tie between two or more nodes. Such as they have same distance or cost and in such a case they are explored depending upon which path came first in the input.txt file.


